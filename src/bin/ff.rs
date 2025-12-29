use engram::Mat;
use mnist::{IMAGE_HEIGHT, IMAGE_WIDTH, Mnist, NPIXELS, error::MnistError};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

// --- Hyperparameters ---
const DROPOUT: f32 = 0.10;
const USE_DROPOUT: bool = true;
const USE_AUGMENTATION: bool = true;
const SANITISE: bool = false;
const TINY: f32 = 1e-10;
const NUMLAB: usize = 10;
const LAMBDAMEAN: f32 = 0.03;
const TEMP: f32 = 1.0;
const LABELSTRENGTH: f32 = 1.0;
const MINLEVELSUP: usize = 2;
const WC: f32 = 0.002;
const SUPWC: f32 = 0.003;
const EPSILON: f32 = 0.01;
const EPSILONSUP: f32 = 0.1;
const DELAY: f32 = 0.9;
const LAYERS: [usize; 4] = [784, 1000, 1000, 1000];
const BATCH_SIZE: usize = 100;
const MAX_EPOCH: usize = 200;

// --- Data Structures ---

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
                .map(|&c| Mat::zeros(batch_size, c))
                .collect(),
            pos_nst: layers.iter().map(|&c| Mat::zeros(batch_size, c)).collect(),
            neg_st: layers[1..]
                .iter()
                .map(|&c| Mat::zeros(batch_size, c))
                .collect(),
            neg_nst: layers.iter().map(|&c| Mat::zeros(batch_size, c)).collect(),
            softmax_st: layers[1..]
                .iter()
                .map(|&c| Mat::zeros(batch_size, c))
                .collect(),
            softmax_nst: layers.iter().map(|&c| Mat::zeros(batch_size, c)).collect(),
            pos_probs: vec![vec![0.0; batch_size]; layers.len() - 1],
            pos_dc_din: layers[1..]
                .iter()
                .map(|&c| Mat::zeros(batch_size, c))
                .collect(),
            neg_dc_din: layers[1..]
                .iter()
                .map(|&c| Mat::zeros(batch_size, c))
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

// --- Helper Functions ---

#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline(always)]
fn goodness(row: &[f32], temp: f32) -> f32 {
    let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
    let cols = row.len() as f32;
    sigmoid((sum_sq - cols) / temp)
}

#[inline(always)]
fn sanitise_slice(data: &mut [f32]) {
    if SANITISE {
        for x in data.iter_mut() {
            if x.is_nan() {
                *x = 0.0;
            }
            *x = x.clamp(-1e10, 1e10);
        }
    }
}

/// Applies MatMul -> Bias -> ReLU -> Dropout -> Normalization
fn layer_io_into(
    vin: &Mat,
    layer: &Layer,
    st: &mut Mat,
    nst: &mut Mat,
    orng: Option<&mut SmallRng>,
) {
    vin.matmul_into(&layer.weights, st);
    let cols = st.cols;

    // Process Bias + ReLU + Dropout
    if let Some(rng) = orng
        && USE_DROPOUT
    {
        let dropout_scale = 1.0 / (1.0 - DROPOUT).sqrt(); // TODO: no sqrt?
        st.data.chunks_exact_mut(cols).for_each(|row| {
            for (val, &bias) in row.iter_mut().zip(layer.biases.iter()) {
                *val = (*val + bias).max(0.0); // ReLU
                if rng.random::<f32>() < DROPOUT {
                    *val = 0.0;
                } else {
                    *val *= dropout_scale;
                }
            }
        });
    } else {
        // Inference path (Parallelizable)
        st.data.par_chunks_mut(cols).for_each(|row| {
            for (val, &bias) in row.iter_mut().zip(layer.biases.iter()) {
                *val = (*val + bias).max(0.0);
            }
        });
    }
    nst.data.copy_from_slice(&st.data);
    nst.norm_rows();
}

fn apply_random_shift(src_image: &[f32; NPIXELS], target_buffer: &mut [f32], rng: &mut SmallRng) {
    let shift_x = rng.random_range(-1..=1);
    let shift_y = rng.random_range(-1..=1);

    if shift_x == 0 && shift_y == 0 {
        target_buffer.copy_from_slice(src_image);
        return;
    }

    const W: i32 = IMAGE_WIDTH as i32;
    const H: i32 = IMAGE_HEIGHT as i32;
    let y_start = 0.max(-shift_y) as usize;
    let y_end = H.min(H - shift_y) as usize;
    let x_start = 0.max(-shift_x) as usize;
    let x_end = W.min(W - shift_x) as usize;
    let copy_width = x_end - x_start;

    target_buffer.fill(0.0);

    for y in y_start..y_end {
        let src_y = (y as i32 + shift_y) as usize;
        let src_x = (x_start as i32 + shift_x) as usize;
        let dst_idx = y * IMAGE_WIDTH + x_start;
        let src_idx = src_y * IMAGE_WIDTH + src_x;
        target_buffer[dst_idx..dst_idx + copy_width]
            .copy_from_slice(&src_image[src_idx..src_idx + copy_width]);
    }
}

/// Prepares the batch in parallel.
/// Uses a seed vector to ensure deterministic behavior across threads.
fn prepare_positive_batch(
    images: &[[f32; NPIXELS]],
    labels: &[u8],
    indices: &[usize],
    batch_idx: usize,
    ws: &mut BatchWorkspace,
    seeds: &[u64], // Pre-generated seeds for determinism
) {
    let batch_start = batch_idx * BATCH_SIZE;
    let chunk_indices = &indices[batch_start..batch_start + BATCH_SIZE];

    ws.data
        .data
        .par_chunks_exact_mut(NPIXELS)
        .zip(ws.targets.data.par_chunks_exact_mut(NUMLAB))
        .zip(chunk_indices)
        .zip(seeds)
        .for_each(|(((img_buf, target_buf), &sample_idx), &seed)| {
            let mut local_rng = SmallRng::seed_from_u64(seed);

            let img_buf: &mut [f32] = img_buf;
            let target_buf: &mut [f32] = target_buf;
            let label = labels[sample_idx] as usize;

            // 1. Augment / Copy
            if USE_AUGMENTATION {
                apply_random_shift(&images[sample_idx], img_buf, &mut local_rng);
            } else {
                img_buf.copy_from_slice(&images[sample_idx][..]);
            }

            // 2. Set Targets (One Hot)
            target_buf.fill(0.0);
            target_buf[label] = 1.0;

            // 3. Embed Label
            img_buf[..NUMLAB].fill(0.0);
            img_buf[label] = LABELSTRENGTH;
        });

    // Initialize first layer
    ws.pos_nst[0].data.copy_from_slice(&ws.data.data);
    ws.pos_nst[0].norm_rows();
}

fn train_epoch(
    model: &mut [Layer],
    images: &[[f32; NPIXELS]],
    labels: &[u8],
    epoch: usize,
    rng: &mut SmallRng,
    ws: &mut BatchWorkspace,
    indices: &mut [usize],
    seed_buffer: &mut Vec<u64>,
) -> f32 {
    let num_batches = images.len() / BATCH_SIZE;
    let epsgain = if epoch < MAX_EPOCH / 2 {
        1.0
    } else {
        (1.0 + 2.0 * (MAX_EPOCH - epoch) as f32) / MAX_EPOCH as f32
    };

    let mut total_cost = 0.0;
    indices.shuffle(rng);

    for b in 0..num_batches {
        // Generate seeds for this batch on the main thread for determinism
        for seed in seed_buffer.iter_mut() {
            *seed = rng.next_u64();
        }

        prepare_positive_batch(images, labels, indices, b, ws, seed_buffer);

        // --- 1. FORWARD PASS (POSITIVE) ---
        for l in 0..model.len() {
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
                ws.pos_probs[l][r] = goodness(&ws.pos_st[l].data[r * cols..(r + 1) * cols], TEMP);
            }
        }

        // --- 2. SOFTMAX ---
        ws.lab_data.data.copy_from_slice(&ws.data.data);
        for r in 0..BATCH_SIZE {
            ws.lab_data.data[r * 784..r * 784 + NUMLAB].fill(LABELSTRENGTH / NUMLAB as f32);
        }
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

        // Supervised Contributions
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

        // Softmax & Gradients
        for r in 0..BATCH_SIZE {
            let row = &mut ws.labin.data[r * NUMLAB..(r + 1) * NUMLAB];
            let target_row = &ws.targets.data[r * NUMLAB..(r + 1) * NUMLAB];

            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;
            for x in row.iter_mut() {
                *x = (*x - max_val).exp();
                sum_exp += *x;
            }

            let mut correct_p = 0.0;
            for c in 0..NUMLAB {
                row[c] /= sum_exp;
                correct_p += row[c] * target_row[c];
                ws.dc_din_sup.data[r * NUMLAB + c] = target_row[c] - row[c];
            }
            total_cost += -(correct_p + TINY).ln();
        }

        for l in MINLEVELSUP - 1..model.len() {
            if let Some(sw) = &mut model[l].supweights {
                ws.softmax_nst[l + 1].t_matmul_into(&ws.dc_din_sup, &mut ws.sw_grad_tmp);
                let g_buf = model[l].supweights_grad.as_mut().unwrap();
                let scale = epsgain * EPSILONSUP;
                for i in 0..sw.data.len() {
                    g_buf.data[i] = DELAY * g_buf.data[i]
                        + (1.0 - DELAY) * ws.sw_grad_tmp.data[i] / BATCH_SIZE as f32;
                    sw.data[i] += scale * (g_buf.data[i] - SUPWC * sw.data[i]);
                }
            }
        }

        // --- 3. NEGATIVE PASS ---
        ws.neg_data.data.copy_from_slice(&ws.data.data);
        for r in 0..BATCH_SIZE {
            let start_idx = r * NUMLAB;
            let probs = &ws.labin.data[start_idx..start_idx + NUMLAB];
            let targets = &ws.targets.data[start_idx..start_idx + NUMLAB];

            let mut sel = 0;
            let rv: f32 = rng.random();
            let mut cum = 0.0;
            let sum_other: f32 = probs
                .iter()
                .zip(targets)
                .filter(|&(_, &t)| t == 0.0)
                .map(|(&p, _)| p)
                .sum();

            for (c, (&p, &t)) in probs.iter().zip(targets).enumerate() {
                if t == 0.0 {
                    cum += p / (sum_other + TINY);
                    if rv < cum {
                        sel = c;
                        break;
                    }
                }
            }
            let img_start = r * 784;
            let label_slice = &mut ws.neg_data.data[img_start..img_start + NUMLAB];
            label_slice.fill(0.0);
            label_slice[sel] = LABELSTRENGTH;
        }

        ws.neg_nst[0].data.copy_from_slice(&ws.neg_data.data);
        ws.neg_nst[0].norm_rows();

        // --- 4. WEIGHT UPDATES ---
        for l in 0..model.len() {
            let cols = model[l].weights.cols;
            let layer_mean: f32 = model[l].mean_states.iter().sum::<f32>() / cols as f32;
            let inv_bs = 1.0 / BATCH_SIZE as f32;

            for r in 0..BATCH_SIZE {
                let p = ws.pos_probs[l][r];
                let row_offset = r * cols;
                for c in 0..cols {
                    let st = ws.pos_st[l].data[row_offset + c];
                    model[l].mean_states[c] = 0.9 * model[l].mean_states[c] + 0.1 * (st * inv_bs);
                    let reg = LAMBDAMEAN * (layer_mean - model[l].mean_states[c]);
                    ws.pos_dc_din[l].data[row_offset + c] = (1.0 - p) * st + reg;
                }
            }
            ws.pos_nst[l].t_matmul_into(&ws.pos_dc_din[l], &mut ws.pos_dw[l]);

            let (prev_nst, next_nst) = ws.neg_nst.split_at_mut(l + 1);
            layer_io_into(
                &prev_nst[l],
                &model[l],
                &mut ws.neg_st[l],
                &mut next_nst[0],
                Some(rng),
            );

            for r in 0..BATCH_SIZE {
                let row_offset = r * cols;
                let row = &ws.neg_st[l].data[row_offset..row_offset + cols];
                let p_neg = goodness(row, TEMP);
                for c in 0..cols {
                    ws.neg_dc_din[l].data[row_offset + c] = -p_neg * row[c];
                }
            }
            ws.neg_nst[l].t_matmul_into(&ws.neg_dc_din[l], &mut ws.neg_dw[l]);

            let w_scale = epsgain * EPSILON;
            let wg = &mut model[l].weights_grad.data;
            let w = &mut model[l].weights.data;
            let pdw = &ws.pos_dw[l].data;
            let ndw = &ws.neg_dw[l].data;

            for i in 0..w.len() {
                let g = (pdw[i] + ndw[i]) * inv_bs;
                wg[i] = DELAY * wg[i] + (1.0 - DELAY) * g;
                w[i] += w_scale * (wg[i] - WC * w[i]);
            }

            let bg = &mut model[l].biases_grad;
            let b = &mut model[l].biases;
            for c in 0..b.len() {
                let mut g = 0.0;
                for r in 0..BATCH_SIZE {
                    g += ws.pos_dc_din[l].data[r * cols + c] + ws.neg_dc_din[l].data[r * cols + c];
                }
                bg[c] = DELAY * bg[c] + (1.0 - DELAY) * (g * inv_bs);
                b[c] += w_scale * bg[c];
            }
        }
    }
    total_cost / num_batches as f32
}

fn predict(model: &[Layer], image: &[f32], ws: &mut BatchWorkspace) -> usize {
    ws.data.data[..784].copy_from_slice(image);
    ws.data.data[..NUMLAB].fill(LABELSTRENGTH / NUMLAB as f32);
    let input_len = ws.pos_nst[0].data.len();
    ws.pos_nst[0]
        .data
        .copy_from_slice(&ws.data.data[..input_len]);
    ws.pos_nst[0].norm_rows();

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

    let mut scores = [0.0f32; NUMLAB];
    for l in MINLEVELSUP - 1..model.len() {
        if let Some(sw) = &model[l].supweights {
            ws.pos_nst[l + 1].matmul_into(sw, &mut ws.sup_contrib);
            for c in 0..NUMLAB {
                scores[c] += ws.sup_contrib.data[c];
            }
        }
    }
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn fftest(model: &[Layer], images: &[[f32; NPIXELS]], labels: &[u8]) -> (usize, usize) {
    let errors: usize = images
        .par_iter()
        .zip(labels)
        .map_init(
            || BatchWorkspace::new(&LAYERS, 1),
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
    writer.write_all(&(model.len() as u64).to_le_bytes())?;
    for layer in model {
        layer.weights.write_raw(&mut writer)?;
        writer.write_all(&(layer.biases.len() as u64).to_le_bytes())?;
        let b_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(layer.biases.as_ptr() as *const u8, layer.biases.len() * 4)
        };
        writer.write_all(b_bytes)?;
        if let Some(sw) = &layer.supweights {
            writer.write_all(&[1u8])?;
            sw.write_raw(&mut writer)?;
        } else {
            writer.write_all(&[0u8])?;
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
    // UNIFIED RNG: Single SmallRng used for everything.
    let mut rng = SmallRng::seed_from_u64(1234);

    let mut model: Vec<Layer> = (0..LAYERS.len() - 1)
        .map(|i| {
            let fanin = LAYERS[i];
            let fanout = LAYERS[i + 1];
            // Requires updated Mat::new_randn that accepts generic Rng
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

    let train_imgs: Vec<[f32; NPIXELS]> = data
        .train_images
        .iter()
        .map(|img| img.as_f32_array())
        .collect();
    let mut ws = BatchWorkspace::new(&LAYERS, BATCH_SIZE);

    // TRAINING RANGES
    const RTRAIN: std::ops::Range<usize> = 0..50000;
    const RVAL: std::ops::Range<usize> = 50000..60000;

    // Initialize indices specifically for the training slice size (50,000)
    let mut indices: Vec<usize> = (0..RTRAIN.len()).collect();
    let mut seed_buffer: Vec<u64> = vec![0; BATCH_SIZE];

    println!("Training Forward-Forward Model...");
    for epoch in 0..MAX_EPOCH {
        let cost = train_epoch(
            &mut model,
            &train_imgs[RTRAIN],
            &data.train_labels[RTRAIN],
            epoch,
            &mut rng,
            &mut ws,
            &mut indices,
            &mut seed_buffer,
        );

        if (epoch > 0 && epoch % 5 == 0) || epoch == MAX_EPOCH - 1 {
            let (errors0, total0) = fftest(&model, &train_imgs[RTRAIN], &data.train_labels[RTRAIN]);
            let (errors1, total1) = fftest(&model, &train_imgs[RVAL], &data.train_labels[RVAL]);
            println!(
                "Epoch {epoch:3} | Cost: {cost:8.4} | Err Train: ({errors0}/{total0}), Err Val: ({errors1}/{total1})"
            );
        } else {
            println!("Epoch {epoch:3} | Cost: {cost:8.4}");
        }
    }
    save_model(&model, "model_ff.bin")?;
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
    matrix
}

fn print_confusions(matrix: &[[usize; 10]; 10]) {
    println!("\nConfusion Matrix (Actual vs Predicted):\n");
    print!("Actual |");
    for i in 0..10 {
        print!("  P{i}  ");
    }
    println!("\n-------|------------------------------------------------------------");
    for i in 0..10 {
        print!("  A{}   |", i);
        for j in 0..10 {
            let count = matrix[i][j];
            if count == 0 {
                print!("{:>5} ", ".");
            } else {
                print!("{count:>5} ");
            }
        }
        println!();
    }
    println!("--------------------------------------------------------------------");
}

fn main() -> Result<(), MnistError> {
    train_model()?;
    if true {
        let data = Mnist::load("MNIST")?;
        let train_imgs: Vec<[f32; NPIXELS]> = data
            .train_images
            .iter()
            .map(|img| img.as_f32_array())
            .collect();
        let test_imgs: Vec<[f32; NPIXELS]> = data
            .test_images
            .iter()
            .map(|img| img.as_f32_array())
            .collect();
        let model = load_model("model_ff.bin")?;
        let (errors, total) = fftest(&model, &test_imgs, &data.test_labels);
        println!("Test Errors: ({errors}/{total})");
        let (errors, total) = fftest(&model, &train_imgs, &data.train_labels);
        println!("Train Errors: ({errors}/{total})");
        let matrix = calc_confusions(&model, &test_imgs, &data.test_labels);
        print_confusions(&matrix);
    }
    Ok(())
}
