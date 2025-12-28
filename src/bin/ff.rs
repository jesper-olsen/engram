use engram::Mat;
use mnist::{Mnist, error::MnistError};
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

const DROPOUT: f32 = 0.10; // 10% of neurons
const USE_AUGMENTATION: bool = true;

//const TINY: f32 = 1e-20;
const TINY: f32 = 1e-10;
const NUMLAB: usize = 10;
const LAMBDAMEAN: f32 = 0.03;
const TEMP: f32 = 1.0;
const LABELSTRENGTH: f32 = 1.0;
const MINLEVELSUP: usize = 2;
//const MINLEVELENERGY: usize = 2;
const WC: f32 = 0.002;
const SUPWC: f32 = 0.003;
const EPSILON: f32 = 0.01;
const EPSILONSUP: f32 = 0.1;
const DELAY: f32 = 0.9;
const LAYERS: [usize; 4] = [784, 1000, 1000, 1000];
const BATCH_SIZE: usize = 100;
//const MAX_EPOCH: usize = 125;
const MAX_EPOCH: usize = 200;

struct Layer {
    weights: Mat,
    biases: Vec<f32>,
    supweights: Option<Mat>,

    weights_grad: Mat,
    biases_grad: Vec<f32>,
    supweights_grad: Option<Mat>,

    mean_states: Vec<f32>,
}

struct BatchWorkspace {
    data: Mat,
    targets: Mat,
    lab_data: Mat,
    labin: Mat,
    dc_din_sup: Mat,
    neg_data: Mat,
    pos_st: Vec<Mat>,
    pos_nst: Vec<Mat>,
    neg_st: Vec<Mat>,
    neg_nst: Vec<Mat>,
    softmax_nst: Vec<Mat>,
    pos_probs: Vec<Vec<f32>>,
    pos_dc_din: Vec<Mat>,
    neg_dc_din: Vec<Mat>,
    // Gradient buffers to avoid allocations
    pos_dw: Vec<Mat>,
    neg_dw: Vec<Mat>,
    sup_contrib: Mat,
    softmax_st: Vec<Mat>,
    sw_grad_tmp: Mat,
}

impl BatchWorkspace {
    fn new(layers: &[usize], batch_size: usize) -> Self {
        BatchWorkspace {
            data: Mat::zeros(batch_size, layers[0]),
            targets: Mat::zeros(batch_size, NUMLAB),
            lab_data: Mat::zeros(batch_size, layers[0]),
            labin: Mat::zeros(batch_size, NUMLAB),
            dc_din_sup: Mat::zeros(batch_size, NUMLAB),
            neg_data: Mat::zeros(batch_size, layers[0]),
            pos_st: layers[1..]
                .iter()
                .map(|&cols| Mat::zeros(batch_size, cols))
                .collect(),
            pos_nst: (0..layers.len())
                .map(|i| Mat::zeros(batch_size, layers[i]))
                .collect(),
            neg_st: layers[1..]
                .iter()
                .map(|&cols| Mat::zeros(batch_size, cols))
                .collect(),
            neg_nst: (0..layers.len())
                .map(|i| Mat::zeros(batch_size, layers[i]))
                .collect(),
            softmax_st: layers[1..]
                .iter()
                .map(|&cols| Mat::zeros(batch_size, cols))
                .collect(),
            softmax_nst: (0..layers.len())
                .map(|i| Mat::zeros(batch_size, layers[i]))
                .collect(),
            pos_probs: vec![vec![0.0; batch_size]; layers.len() - 1],
            pos_dc_din: layers[1..]
                .iter()
                .map(|&cols| Mat::zeros(batch_size, cols))
                .collect(),
            neg_dc_din: layers[1..]
                .iter()
                .map(|&cols| Mat::zeros(batch_size, cols))
                .collect(),
            pos_dw: (0..layers.len() - 1)
                .map(|i| Mat::zeros(layers[i], layers[i + 1]))
                .collect(),
            neg_dw: (0..layers.len() - 1)
                .map(|i| Mat::zeros(layers[i], layers[i + 1]))
                .collect(),
            sup_contrib: Mat::zeros(batch_size, NUMLAB),
            sw_grad_tmp: Mat::zeros(layers.iter().max().cloned().unwrap_or(NUMLAB), NUMLAB),
        }
    }
}

fn layer_io_into(vin: &Mat, layer: &Layer, st: &mut Mat, nst: &mut Mat, orng: Option<&mut StdRng>) {
    vin.matmul_into(&layer.weights, st);

    let cols = st.cols;
    st.data.par_chunks_mut(cols).for_each(|row| {
        for (val, &bias) in row.iter_mut().zip(layer.biases.iter()) {
            *val = (*val + bias).max(0.0);
        }
    });

    // dropout - for training only
    if let Some(rng) = orng {
        for val in st.data.iter_mut() {
            if rng.random::<f32>() < DROPOUT {
                *val = 0.0;
            } else {
                // Rescale to keep the expected sum of squares (Goodness) consistent
                *val /= (1.0 - DROPOUT).sqrt();
            }
        }
    }

    nst.data.copy_from_slice(&st.data);
    nst.norm_rows();
}

fn embed_label(buffer: &mut [f32], label_idx: usize, strength: f32, num_labels: usize) {
    // Clear the first N pixels
    buffer[..num_labels].fill(0.0);
    // Set the specific pixel for the label
    if label_idx < num_labels {
        buffer[label_idx] = strength;
    }
}

fn apply_random_shift(
    src_image: &[f32; mnist::NPIXELS],
    target_buffer: &mut [f32],
    rng: &mut StdRng,
) {
    let shift_x = rng.random_range(-1..=1);
    let shift_y = rng.random_range(-1..=1);

    for y in 0..28 {
        for x in 0..28 {
            let src_x = x as i32 + shift_x;
            let src_y = y as i32 + shift_y;

            let val = if src_x >= 0 && src_x < 28 && src_y >= 0 && src_y < 28 {
                src_image[(src_y * 28 + src_x) as usize]
            } else {
                0.0
            };
            target_buffer[y * 28 + x] = val;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Applies the "goodness" function: sum of squares of activities
fn calc_prob(row: &[f32], temp: f32) -> f32 {
    let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
    let cols = row.len() as f32;
    // Logistic function on the goodness minus threshold (threshold is number of neurons)
    let logits = (sum_sq - cols) / temp;
    let logits = logits.clamp(-80.0,80.0); // prevent overflow
    sigmoid(logits)
}

/// Clamps values to prevent NaN or Infinity propagation
fn sanitise_slice(data: &mut [f32]) {
    for x in data.iter_mut() {
        if x.is_nan() { *x = 0.0; }
        *x = x.clamp(-1e10, 1e10);
    }
}

fn train_epoch(
    model: &mut [Layer],
    images: &[[f32; mnist::NPIXELS]],
    labels: &[u8],
    epoch: usize,
    rng: &mut StdRng,
    ws: &mut BatchWorkspace,
) -> f32 {
    let num_batches = images.len() / BATCH_SIZE;
    let epsgain = if epoch < MAX_EPOCH / 2 {
        1.0
    } else {
        (1.0 + 2.0 * (MAX_EPOCH - epoch) as f32) / MAX_EPOCH as f32
    };
    let mut total_cost = 0.0;

    let mut indices: Vec<usize> = (0..images.len()).collect();
    indices.shuffle(rng);

    for b in 0..num_batches {
        // prepare positive batch - data + correct label
        for i in 0..BATCH_SIZE {
            let idx = indices[b * BATCH_SIZE + i];
            let start = i * mnist::NPIXELS;

            if USE_AUGMENTATION {
                apply_random_shift(
                    &images[idx],
                    &mut ws.data.data[start..start + mnist::NPIXELS],
                    rng,
                );
            } else {
                ws.data.data[start..start + mnist::NPIXELS].copy_from_slice(&images[idx]);
            }

            // embed label
            let lab = labels[idx] as usize;
            ws.targets.data[i * NUMLAB..(i + 1) * NUMLAB].fill(0.0);
            ws.targets.data[i * NUMLAB + lab] = 1.0;

            ws.data.data[start..start + NUMLAB].fill(0.0);
            ws.data.data[start + lab] = LABELSTRENGTH;
        }
        ws.pos_nst[0].data.copy_from_slice(&ws.data.data);
        ws.pos_nst[0].norm_rows();

        // -- forward pass (positive)
        for l in 0..model.len() {
            // Split the slice so we can borrow l and l+1 simultaneously
            let (prev_nst, next_nst) = ws.pos_nst.split_at_mut(l + 1);
            layer_io_into(
                &prev_nst[l],
                &model[l],
                &mut ws.pos_st[l],
                &mut next_nst[0],
                Some(rng),
            );
            sanitise_slice(&mut ws.pos_st[l].data);

            let cols = ws.pos_st[l].cols;
            for r in 0..BATCH_SIZE {
                let row = &ws.pos_st[l].data[r * cols..(r + 1) * cols];
                ws.pos_probs[l][r] = calc_prob(row, TEMP);
            }
        }

        // --- 3. SOFTMAX PREDICTOR & SUPERVISED WEIGHTS ---
        ws.lab_data.data.copy_from_slice(&ws.data.data);
        ws.lab_data
            .data
            .chunks_exact_mut(784)
            .for_each(|img| img[..NUMLAB].fill(LABELSTRENGTH / NUMLAB as f32));

        ws.softmax_nst[0].data.copy_from_slice(&ws.lab_data.data);
        ws.softmax_nst[0].norm_rows();
        for l in 0..model.len() {
            let (prev_nst, next_nst) = ws.softmax_nst.split_at_mut(l + 1);
            layer_io_into(
                &prev_nst[l],
                &model[l],
                &mut ws.softmax_st[l],
                &mut next_nst[0],
                Some(rng),
            );
        }

        ws.labin.data.fill(0.0);
        for l in MINLEVELSUP - 1..model.len() {
            if let Some(sw) = &model[l].supweights {
                ws.softmax_nst[l + 1].matmul_into(sw, &mut ws.sup_contrib);
                for i in 0..ws.labin.data.len() {
                    ws.labin.data[i] += ws.sup_contrib.data[i];
                }
            }
        }

        sanitise_slice(&mut ws.labin.data);

        // Softmax & Cost Calculation
        for r in 0..BATCH_SIZE {
            let row = &mut ws.labin.data[r * NUMLAB..(r + 1) * NUMLAB];
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;
            for x in row.iter_mut() {
                *x = (*x - max_val).exp();
                sum_exp += *x;
            }
            row.iter_mut().for_each(|x| *x /= sum_exp);

            let mut correct_p = 0.0;
            for c in 0..NUMLAB {
                correct_p += row[c] * ws.targets.data[r * NUMLAB + c];
            }
            total_cost += -(correct_p + TINY).ln();
        }

        // Supervised Weight Update
        for i in 0..ws.dc_din_sup.data.len() {
            ws.dc_din_sup.data[i] = ws.targets.data[i] - ws.labin.data[i];
        }
        for l in MINLEVELSUP - 1..model.len() {
            if let Some(sw) = &mut model[l].supweights {
                ws.softmax_nst[l + 1].t_matmul_into(&ws.dc_din_sup, &mut ws.sw_grad_tmp);
                let g_buf = model[l].supweights_grad.as_mut().unwrap();
                for i in 0..sw.data.len() {
                    g_buf.data[i] = DELAY * g_buf.data[i]
                        + (1.0 - DELAY) * ws.sw_grad_tmp.data[i] / BATCH_SIZE as f32;
                    sw.data[i] += epsgain * EPSILONSUP * (g_buf.data[i] - SUPWC * sw.data[i]);
                }
            }
        }

        // --- 4. NEGATIVE PASS ---

        for r in 0..BATCH_SIZE {
            let mut probs = [0.0f32; NUMLAB];
            let start_idx = r * NUMLAB;
            probs.copy_from_slice(&ws.labin.data[start_idx..start_idx + NUMLAB]);
            for c in 0..NUMLAB {
                if ws.targets.data[r * NUMLAB + c] > 0.0 {
                    probs[c] = 0.0;
                }
            }
            let sum: f32 = probs.iter().sum();

            let mut cum = 0.0;
            let rv: f32 = rng.random();
            let mut sel = 0;
            for (c, &p) in probs.iter().enumerate() {
                cum += p / (sum + TINY);
                if rv < cum {
                    sel = c;
                    break;
                }
            }

            let start = r * 784;
            // Copy the already shifted/prepared image from pos buffer to neg buffer
            ws.neg_data.data[start..start + 784].copy_from_slice(&ws.data.data[start..start + 784]);

            // Overwrite with a WRONG label for the negative pass
            ws.neg_data.data[start..start + NUMLAB].fill(0.0);
            ws.neg_data.data[start + sel] = LABELSTRENGTH;
        }

        ws.neg_nst[0].data.copy_from_slice(&ws.neg_data.data);
        ws.neg_nst[0].norm_rows();

        // --- 5. LAYER UPDATES (WEIGHTS & BIASES) ---
        for l in 0..model.len() {
            // Pos Grad preparation
            let cols = model[l].weights.cols;
            let layer_mean: f32 = model[l].mean_states.iter().sum::<f32>() / cols as f32;

            for r in 0..BATCH_SIZE {
                let p = ws.pos_probs[l][r];
                for c in 0..cols {
                    let st = ws.pos_st[l].data[r * cols + c];
                    model[l].mean_states[c] =
                        0.9 * model[l].mean_states[c] + 0.1 * (st / BATCH_SIZE as f32);
                    let reg = LAMBDAMEAN * (layer_mean - model[l].mean_states[c]);
                    let mut grad_val = (1.0 - p) * st;
                    if grad_val.is_nan() { grad_val = 0.0; } // NaN shield
                    ws.pos_dc_din[l].data[r * cols + c] = grad_val + reg;
                }
            }
            ws.pos_nst[l].t_matmul_into(&ws.pos_dc_din[l], &mut ws.pos_dw[l]);

            // Neg pass
            let (prev_nst, next_nst) = ws.neg_nst.split_at_mut(l + 1);
            layer_io_into(
                &prev_nst[l],
                &model[l],
                &mut ws.neg_st[l],
                &mut next_nst[0],
                Some(rng),
            );

            for r in 0..BATCH_SIZE {
                let row = &ws.neg_st[l].data[r * cols..(r + 1) * cols];
                let p_neg = calc_prob(row, TEMP);

                for c in 0..cols {
                    ws.neg_dc_din[l].data[r * cols + c] = -p_neg * row[c];
                }
            }
            ws.neg_nst[l].t_matmul_into(&ws.neg_dc_din[l], &mut ws.neg_dw[l]);

            // Apply Updates to Weights
            for i in 0..model[l].weights.data.len() {
                let g = (ws.pos_dw[l].data[i] + ws.neg_dw[l].data[i]) / BATCH_SIZE as f32;
                model[l].weights_grad.data[i] =
                    DELAY * model[l].weights_grad.data[i] + (1.0 - DELAY) * g;
                model[l].weights.data[i] += epsgain
                    * EPSILON
                    * (model[l].weights_grad.data[i] - WC * model[l].weights.data[i]);
            }

            // Apply Updates to Biases
            for c in 0..model[l].biases.len() {
                let mut g = 0.0;
                for r in 0..BATCH_SIZE {
                    g += ws.pos_dc_din[l].data[r * cols + c] + ws.neg_dc_din[l].data[r * cols + c];
                }
                model[l].biases_grad[c] =
                    DELAY * model[l].biases_grad[c] + (1.0 - DELAY) * (g / BATCH_SIZE as f32);
                model[l].biases[c] += epsgain * EPSILON * model[l].biases_grad[c];
            }
        }
    }
    total_cost / num_batches as f32
}

fn predict(model: &[Layer], image: &[f32], ws: &mut BatchWorkspace) -> usize {
    // 1. Prepare input: Copy image to workspace data buffer
    ws.data.data[..784].copy_from_slice(image);
    ws.data.data[..NUMLAB].fill(LABELSTRENGTH / NUMLAB as f32);

    // 2. Initial normalization
    let input_len = ws.pos_nst[0].data.len();
    ws.pos_nst[0]
        .data
        .copy_from_slice(&ws.data.data[..input_len]);
    //ffnormrows(&mut ws.pos_nst[0]);
    ws.pos_nst[0].norm_rows();

    // 3. Forward Pass: Traverse layers
    for l in 0..model.len() {
        let (prev_nst, next_nst) = ws.pos_nst.split_at_mut(l + 1);
        layer_io_into(
            &prev_nst[l],
            &model[l],
            &mut ws.pos_st[l],
            &mut next_nst[0],
            None,
        );
    }

    // 4. Sum scores from all supervised layers
    let mut scores = [0.0f32; NUMLAB];
    for l in MINLEVELSUP - 1..model.len() {
        if let Some(sw) = &model[l].supweights {
            // Matmul into sup_contrib (which is 1x10 in this workspace)
            ws.pos_nst[l + 1].matmul_into(sw, &mut ws.sup_contrib);
            for c in 0..NUMLAB {
                scores[c] += ws.sup_contrib.data[c];
            }
        }
    }

    // 5. Return index of the highest score
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap()
}

fn fftest(model: &[Layer], images: &[[f32; mnist::NPIXELS]], labels: &[u8]) -> (usize, usize) {
    let errors: usize = images
        .par_iter()
        .zip(labels)
        .map_init(
            || BatchWorkspace::new(&LAYERS, 1), // Initialize 1 workspace per thread
            |ws, (img, &label)| {
                if predict(model, img, ws) != label as usize {
                    1
                } else {
                    0
                }
            },
        )
        .sum();
    (errors, images.len())
}

fn save_model(model: &[Layer], path: &str) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write number of layers
    writer.write_all(&(model.len() as u64).to_le_bytes())?;

    for layer in model {
        // 1. Weights
        layer.weights.write_raw(&mut writer)?;

        // 2. Biases (Vec<f32>)
        writer.write_all(&(layer.biases.len() as u64).to_le_bytes())?;
        let b_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(layer.biases.as_ptr() as *const u8, layer.biases.len() * 4)
        };
        writer.write_all(b_bytes)?;

        // 3. Supervised Weights
        if let Some(sw) = &layer.supweights {
            writer.write_all(&[1u8])?; // Flag for Some
            sw.write_raw(&mut writer)?;
        } else {
            writer.write_all(&[0u8])?; // Flag for None
        }
    }
    Ok(())
}

fn load_model(path: &str) -> std::io::Result<Vec<Layer>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut b8 = [0u8; 8];
    reader.read_exact(&mut b8)?;
    let num_layers = u64::from_le_bytes(b8) as usize;

    let mut model = Vec::with_capacity(num_layers);

    for _ in 0..num_layers {
        let weights = Mat::read_raw(&mut reader)?;

        reader.read_exact(&mut b8)?;
        let b_len = u64::from_le_bytes(b8) as usize;
        let mut biases = vec![0.0f32; b_len];
        let b_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(biases.as_mut_ptr() as *mut u8, b_len * 4) };
        reader.read_exact(b_bytes)?;

        let mut opt_flag = [0u8; 1];
        reader.read_exact(&mut opt_flag)?;
        let supweights = if opt_flag[0] == 1 {
            Some(Mat::read_raw(&mut reader)?)
        } else {
            None
        };

        // Reconstruct gradients and states as zeros
        let (rows, cols) = (weights.rows, weights.cols);
        model.push(Layer {
            weights_grad: Mat::zeros(rows, cols),
            biases_grad: vec![0.0; cols],
            supweights_grad: supweights.as_ref().map(|sw| Mat::zeros(sw.rows, sw.cols)),
            mean_states: vec![0.5; cols],
            weights,
            biases,
            supweights,
        });
    }
    Ok(model)
}

fn train_model() -> Result<(), MnistError> {
    let data = Mnist::load("MNIST")?;
    let mut rng = StdRng::seed_from_u64(1234);

    let mut model: Vec<Layer> = (0..LAYERS.len() - 1)
        .map(|i| {
            let fanin = LAYERS[i];
            let fanout = LAYERS[i + 1];
            Layer {
                weights: Mat::new_randn(fanin, fanout, 1.0 / (fanin as f32).sqrt(), &mut rng),
                biases: vec![0.0; fanout],
                supweights: Some(Mat::zeros(fanout, NUMLAB)),
                weights_grad: Mat::zeros(fanin, fanout),
                biases_grad: vec![0.0; fanout],
                supweights_grad: Some(Mat::zeros(fanout, NUMLAB)),
                mean_states: vec![0.5; fanout],
            }
        })
        .collect();

    let train_imgs: Vec<[f32; mnist::NPIXELS]> = data
        .train_images
        .iter()
        .map(|img| img.as_f32_array())
        .collect();
    let test_imgs: Vec<[f32; mnist::NPIXELS]> = data
        .test_images
        .iter()
        .map(|img| img.as_f32_array())
        .collect();

    let mut ws = BatchWorkspace::new(&LAYERS, BATCH_SIZE);

    println!("Training Forward-Forward Model...");
    for epoch in 0..MAX_EPOCH {
        const RTRAIN: std::ops::Range<usize> = 0..50000;
        const RVAL: std::ops::Range<usize> = 50000..60000;
        let cost = train_epoch(
            &mut model,
            &train_imgs[RTRAIN],
            &data.train_labels[RTRAIN],
            epoch,
            &mut rng,
            &mut ws,
        );
        if (epoch > 0 && epoch % 5 == 0) || epoch == MAX_EPOCH - 1 {
            let (errors0, total0) = fftest(&model, &train_imgs[RTRAIN], &data.train_labels[RTRAIN]);
            let (errors1, total1) = fftest(&model, &train_imgs[RVAL], &data.train_labels[RVAL]);
            println!(
                "Epoch {epoch:3} | Cost: {cost:8.4} | Errors; Train: ({errors0}/{total0}), Valid: ({errors1}/{total1})"
            );
        } else {
            println!("Epoch {epoch:3} | Cost: {cost:8.4}");
        }
    }

    let (errors, total) = fftest(&model, &test_imgs, &data.test_labels);
    println!("Test Errors: ({errors}/{total})");

    let fname = "model_ff.bin";
    println!("Saving model to: {fname}");
    save_model(&model, fname)?;

    Ok(())
}

fn calc_confusions(model: &[Layer], images: &[[f32; 784]], labels: &[u8]) -> [[usize; 10]; 10] {
    let mut matrix = [[0usize; 10]; 10];
    let mut ws = BatchWorkspace::new(&LAYERS, 1);

    println!("Calculating confusion matrix...");
    for (img, &label) in images.iter().zip(labels) {
        let pred = predict(model, img, &mut ws);
        matrix[label as usize][pred] += 1;
    }
    return matrix;
}

fn print_confusions(matrix: &[[usize; 10]; 10]) {
    println!("\nConfusion Matrix (Actual vs Predicted):");
    println!("");

    // Header
    print!("Actual |");
    for i in 0..10 {
        print!("  P{}  ", i);
    }
    println!("\n-------|------------------------------------------------------------");

    for i in 0..10 {
        // Row label
        print!("  A{}   |", i);

        for j in 0..10 {
            let count = matrix[i][j];
            if i == j {
                // Correct: Just the number
                print!("{:>5} ", count);
            } else if count > 0 {
                // Error: Just the number
                print!("{:>5} ", count);
            } else {
                // Zero: Just a dot, padded to match the 5-space width
                print!("{:>5} ", ".");
            }
        }
        println!("");
    }
    println!("--------------------------------------------------------------------");
}

fn main() -> Result<(), MnistError> {
    train_model()?;

    if true {
        let data = Mnist::load("MNIST")?;
        let train_imgs: Vec<[f32; mnist::NPIXELS]> = data
            .train_images
            .iter()
            .map(|img| img.as_f32_array())
            .collect();
        let test_imgs: Vec<[f32; mnist::NPIXELS]> = data
            .test_images
            .iter()
            .map(|img| img.as_f32_array())
            .collect();

        let fname = "model_ff.bin";
        let model = load_model(fname)?;
        let (errors, total) = fftest(&model, &test_imgs, &data.test_labels);
        println!("Test Errors: ({errors}/{total})");

        let (errors, total) = fftest(&model, &train_imgs, &data.train_labels);
        println!("Train Errors: ({errors}/{total})");

        let matrix = calc_confusions(&model, &test_imgs, &data.test_labels);
        print_confusions(&matrix);
    }

    Ok(())
}
