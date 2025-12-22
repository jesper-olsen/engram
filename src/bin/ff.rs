use mnist::{Mnist, error::MnistError};
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;

const TINY: f32 = 1e-20;
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
const MAX_EPOCH: usize = 125;
//const MAX_EPOCH: usize = 25;

#[derive(Clone)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    fn new_randn(rows: usize, cols: usize, scale: f32, rng: &mut StdRng) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            let u: f32 = rng.random();
            let v: f32 = rng.random();
            let z = (-2.0 * u.ln()).sqrt() * (2.0 * std::f32::consts::PI * v).cos();
            data.push(z * scale);
        }
        Mat { rows, cols, data }
    }

    fn matmul(&self, other: &Mat) -> Mat {
        let mut res = Mat::zeros(self.rows, other.cols);
        let a_data = &self.data;
        let b_data = &other.data;

        res.data
            .par_chunks_mut(other.cols)
            .enumerate()
            .for_each(|(i, res_row)| {
                let a_row_offset = i * self.cols;
                for k in 0..self.cols {
                    let a_val = a_data[a_row_offset + k];
                    let b_row_offset = k * other.cols;
                    // contiguous slice - allows the compiler to use SIMD (AVX/SSE) instructions.
                    let b_row = &b_data[b_row_offset..b_row_offset + other.cols];
                    for j in 0..other.cols {
                        res_row[j] += a_val * b_row[j];
                    }
                }
            });
        res
    }

    // Optimized in-place matmul: self * other -> out
    fn matmul_into(&self, other: &Mat, out: &mut Mat) {
        out.data.fill(0.0);
        out.data
            .par_chunks_mut(other.cols)
            .enumerate()
            .for_each(|(i, out_row)| {
                let a_row_offset = i * self.cols;
                for k in 0..self.cols {
                    let a_val = self.data[a_row_offset + k];
                    if a_val == 0.0 {
                        continue;
                    } // Skip zeros
                    let b_row_offset = k * other.cols;
                    let b_row = &other.data[b_row_offset..b_row_offset + other.cols];
                    for j in 0..other.cols {
                        out_row[j] += a_val * b_row[j];
                    }
                }
            });
    }

    // In-place transposed matmul: self^T * other -> out
    fn t_matmul_into(&self, other: &Mat, out: &mut Mat) {
        out.data.fill(0.0);
        let self_cols = self.cols;
        let other_cols = other.cols;

        out.data
            .par_chunks_mut(other_cols)
            .enumerate()
            .for_each(|(i, out_row)| {
                for k in 0..self.rows {
                    let a_val = self.data[k * self_cols + i];
                    if a_val == 0.0 {
                        continue;
                    }
                    let b_row_offset = k * other_cols;
                    let b_row = &other.data[b_row_offset..b_row_offset + other_cols];
                    for j in 0..other_cols {
                        out_row[j] += a_val * b_row[j];
                    }
                }
            });
    }
}

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

fn ffnormrows(a: &mut Mat) {
    // Makes every 'a' have a sum of squared activities that averages 1 per neuron.
    a.data.par_chunks_mut(a.cols).for_each(|row| {
        let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
        let scale = 1.0 / (TINY + (sum_sq / row.len() as f32).sqrt());
        row.iter_mut().for_each(|x| *x *= scale);
    });
}

fn layer_io(vin: &Mat, layer: &Layer) -> (Mat, Mat) {
    let mut states = vin.matmul(&layer.weights);
    for r in 0..states.rows {
        for c in 0..states.cols {
            let idx = r * states.cols + c;
            states.data[idx] = (states.data[idx] + layer.biases[c]).max(0.0);
        }
    }
    let mut norm = states.clone();
    ffnormrows(&mut norm);
    (states, norm)
}

fn layer_io_into(vin: &Mat, layer: &Layer, st: &mut Mat, nst: &mut Mat) {
    vin.matmul_into(&layer.weights, st);
    st.data.par_chunks_mut(st.cols).for_each(|row| {
        for c in 0..row.len() {
            row[c] = (row[c] + layer.biases[c]).max(0.0);
        }
    });
    nst.data.copy_from_slice(&st.data);
    ffnormrows(nst);
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
    let mut total_train_cost = 0.0;

    let mut indices: Vec<usize> = (0..images.len()).collect();
    indices.shuffle(rng);

    for b in 0..num_batches {
        // --- 1. DATA PREPARATION ---
        for i in 0..BATCH_SIZE {
            let idx = indices[b * BATCH_SIZE + i];
            let start = i * mnist::NPIXELS;
            ws.data.data[start..start + mnist::NPIXELS].copy_from_slice(&images[idx]);

            let lab = labels[idx] as usize;
            ws.targets.data[i * NUMLAB..(i + 1) * NUMLAB].fill(0.0);
            ws.targets.data[i * NUMLAB + lab] = 1.0;

            ws.data.data[start..start + NUMLAB].fill(0.0);
            ws.data.data[start + lab] = LABELSTRENGTH;
        }

        // --- 2. POSITIVE PASS ---
        ws.pos_nst[0].data.copy_from_slice(&ws.data.data);
        ffnormrows(&mut ws.pos_nst[0]);

        for l in 0..model.len() {
            // Split the slice so we can borrow l and l+1 simultaneously
            let (prev_nst, next_nst) = ws.pos_nst.split_at_mut(l + 1);
            layer_io_into(&prev_nst[l], &model[l], &mut ws.pos_st[l], &mut next_nst[0]);
            //layer_io_into(&ws.pos_nst[l], &model[l], &mut ws.pos_st[l], &mut ws.pos_nst[l+1]);

            let cols = ws.pos_st[l].cols;
            for r in 0..BATCH_SIZE {
                let row = &ws.pos_st[l].data[r * cols..(r + 1) * cols];
                let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
                ws.pos_probs[l][r] = 1.0 / (1.0 + (-(sum_sq - cols as f32) / TEMP).exp());
            }
        }

        // --- 3. SOFTMAX PREDICTOR & SUPERVISED WEIGHTS ---
        ws.lab_data.data.copy_from_slice(&ws.data.data);
        ws.lab_data
            .data
            .chunks_exact_mut(784)
            .for_each(|img| img[..NUMLAB].fill(LABELSTRENGTH / NUMLAB as f32));

        ws.softmax_nst[0].data.copy_from_slice(&ws.lab_data.data);
        ffnormrows(&mut ws.softmax_nst[0]);
        //for l in 0..model.len() {
        //    layer_io_into(&ws.softmax_nst[l], &model[l], &mut ws.pos_st[l], &mut ws.softmax_nst[l+1]);
        //}
        for l in 0..model.len() {
            let (prev_nst, next_nst) = ws.softmax_nst.split_at_mut(l + 1);
            layer_io_into(
                &prev_nst[l],
                &model[l],
                &mut ws.softmax_st[l],
                &mut next_nst[0],
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
            total_train_cost += -(correct_p + TINY).ln();
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
            let mut probs = ws.labin.data[r * NUMLAB..(r + 1) * NUMLAB].to_vec();
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
            ws.neg_data.data[start..start + 784]
                .copy_from_slice(&images[indices[b * BATCH_SIZE + r]]);
            ws.neg_data.data[start..start + NUMLAB].fill(0.0);
            ws.neg_data.data[start + sel] = LABELSTRENGTH;
        }

        ws.neg_nst[0].data.copy_from_slice(&ws.neg_data.data);
        ffnormrows(&mut ws.neg_nst[0]);

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
                    ws.pos_dc_din[l].data[r * cols + c] = (1.0 - p) * st + reg;
                }
            }
            ws.pos_nst[l].t_matmul_into(&ws.pos_dc_din[l], &mut ws.pos_dw[l]);

            // Neg pass
            //layer_io_into(&ws.neg_nst[l], &model[l], &mut ws.neg_st[l], &mut ws.neg_nst[l+1]);
            let (prev_nst, next_nst) = ws.neg_nst.split_at_mut(l + 1);
            layer_io_into(&prev_nst[l], &model[l], &mut ws.neg_st[l], &mut next_nst[0]);

            for r in 0..BATCH_SIZE {
                let row = &ws.neg_st[l].data[r * cols..(r + 1) * cols];
                let sum_sq: f32 = row.iter().map(|&x| x * x).sum();
                let p = 1.0 / (1.0 + (-(sum_sq - cols as f32) / TEMP).exp());
                for c in 0..cols {
                    ws.neg_dc_din[l].data[r * cols + c] = -p * row[c];
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
    total_train_cost / num_batches as f32
}

//fn predict(model: &[Layer], image: &[f32]) -> usize {
//    // energy test - pic max energy
//    // 1. Prepare neutral input (average label strength)
//    let mut input = Mat::zeros(1, image.len());
//    input.data.copy_from_slice(image);
//    for j in 0..NUMLAB {
//        input.data[j] = LABELSTRENGTH / NUMLAB as f32;
//    }
//
//    let mut n_st = input;
//    ffnormrows(&mut n_st);
//
//    // 2. Forward pass to get normalized states
//    let mut softmax_norms = vec![n_st];
//    for l in 0..model.len() {
//        let (_, nst) = layer_io(&softmax_norms[l], &model[l]);
//        softmax_norms.push(nst);
//    }
//
//    // 3. Accumulate scores from supweights
//    let mut scores = [0.0f32; NUMLAB];
//    for l in MINLEVELSUP - 1..model.len() {
//        if let Some(sw) = &model[l].supweights {
//            let contrib = softmax_norms[l + 1].matmul(sw);
//            for c in 0..NUMLAB {
//                scores[c] += contrib.data[c];
//            }
//        }
//    }
//
//    // 4. Return index of max score
//    scores
//        .iter()
//        .enumerate()
//        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
//        .map(|(index, _)| index)
//        .unwrap()
//}

fn predict(model: &[Layer], image: &[f32], ws: &mut BatchWorkspace) -> usize {
    // 1. Prepare input: Copy image to workspace data buffer
    ws.data.data[..784].copy_from_slice(image);
    ws.data.data[..NUMLAB].fill(LABELSTRENGTH / NUMLAB as f32);

    // 2. Initial normalization
    let input_len = ws.pos_nst[0].data.len();
    ws.pos_nst[0]
        .data
        .copy_from_slice(&ws.data.data[..input_len]);
    ffnormrows(&mut ws.pos_nst[0]);

    // 3. Forward Pass: Traverse layers
    for l in 0..model.len() {
        let (prev_nst, next_nst) = ws.pos_nst.split_at_mut(l + 1);
        layer_io_into(&prev_nst[l], &model[l], &mut ws.pos_st[l], &mut next_nst[0]);
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

//fn fftest(model: &[Layer], images: &[[f32; mnist::NPIXELS]], labels: &[u8]) -> (usize, usize) {
//    let errors = images
//        .par_iter()
//        .zip(labels)
//        .filter(|&(img, &label)| predict(model, img) != label as usize)
//        .count();
//    (errors, images.len())
//}

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

fn main() -> Result<(), MnistError> {
    let data = Mnist::load("MNIST")?;
    let mut rng = StdRng::seed_from_u64(1234);

    let mut model: Vec<Layer> = (0..LAYERS.len() - 1)
        .map(|i| {
            let fanin = LAYERS[i];
            let fanout = LAYERS[i + 1];
            Layer {
                weights: Mat::new_randn(fanin, fanout, 1.0 / (fanin as f32).sqrt(), &mut rng),
                biases: vec![0.0; fanout],
                supweights: Some(Mat::zeros(fanout, NUMLAB)), // Simplified for all layers
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
        if (epoch > 0 && epoch % 5 == 0) || epoch == MAX_EPOCH {
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
    Ok(())
}
