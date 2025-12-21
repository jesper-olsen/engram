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
        res.data
            .par_chunks_mut(other.cols)
            .enumerate()
            .for_each(|(i, row)| {
                for k in 0..self.cols {
                    let a_val = self.data[i * self.cols + k];
                    for j in 0..other.cols {
                        row[j] += a_val * other.data[k * other.cols + j];
                    }
                }
            });
        res
    }

    fn t_matmul(&self, other: &Mat) -> Mat {
        let mut res = Mat::zeros(self.cols, other.cols);
        for i in 0..self.cols {
            for k in 0..self.rows {
                let a_val = self.data[k * self.cols + i];
                for j in 0..other.cols {
                    res.data[i * other.cols + j] += a_val * other.data[k * other.cols + j];
                }
            }
        }
        res
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

fn train_epoch(
    model: &mut [Layer],
    images: &[[f32; mnist::NPIXELS]],
    labels: &[u8],
    epoch: usize,
    rng: &mut StdRng,
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
        let mut data = Mat::zeros(BATCH_SIZE, 784);
        let mut targets = Mat::zeros(BATCH_SIZE, NUMLAB);
        for i in 0..BATCH_SIZE {
            let idx = indices[b * BATCH_SIZE + i];
            //for j in 0..784 {
            //    data.data[i * 784 + j] = images[idx][j];
            //}
            data.data[i*mnist::NPIXELS..].copy_from_slice(&images[idx]);
            let lab = labels[idx] as usize;
            targets.data[i * NUMLAB + lab] = 1.0;
            for j in 0..NUMLAB {
                data.data[i * mnist::NPIXELS + j] = 0.0;
            }
            data.data[i * mnist::NPIXELS + lab] = LABELSTRENGTH;
        }

        // --- POSITIVE PASS ---
        let mut norm_states = vec![data.clone()];
        ffnormrows(&mut norm_states[0]);
        let mut pos_probs = Vec::new();
        let mut layer_states = Vec::new();

        for l in 0..model.len() {
            let (st, nst) = layer_io(&norm_states[l], &model[l]);
            let mut p = Vec::with_capacity(BATCH_SIZE);
            for r in 0..BATCH_SIZE {
                let mut sum_sq = 0.0;
                for c in 0..st.cols {
                    sum_sq += st.data[r * st.cols + c].powi(2);
                }
                p.push(1.0 / (1.0 + (-(sum_sq - st.cols as f32) / TEMP).exp()));
            }
            pos_probs.push(p);
            layer_states.push(st);
            norm_states.push(nst);
        }

        // --- SOFTMAX PREDICTOR & NEGATIVE LABEL SELECTION ---
        let mut lab_data = data.clone();
        for i in 0..BATCH_SIZE {
            for j in 0..NUMLAB {
                lab_data.data[i * 784 + j] = LABELSTRENGTH / NUMLAB as f32;
            }
        }
        let mut n_st = lab_data;
        ffnormrows(&mut n_st);
        let mut softmax_norms = vec![n_st.clone()];
        for l in 0..model.len() {
            let (_, nst) = layer_io(&softmax_norms[l], &model[l]);
            softmax_norms.push(nst);
        }

        let mut labin = Mat::zeros(BATCH_SIZE, NUMLAB);
        for l in MINLEVELSUP - 1..model.len() {
            if let Some(sw) = &model[l].supweights {
                let contrib = softmax_norms[l + 1].matmul(sw);
                for i in 0..labin.data.len() {
                    labin.data[i] += contrib.data[i];
                }
            }
        }

        // Softmax for predictions and cost
        for r in 0..BATCH_SIZE {
            let row = &mut labin.data[r * NUMLAB..(r + 1) * NUMLAB];
            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut sum_exp = 0.0;
            for x in row.iter_mut() {
                *x = (*x - max_val).exp();
                sum_exp += *x;
            }
            for x in row.iter_mut() {
                *x /= sum_exp;
            }
            let mut correct_p = 0.0;
            for c in 0..NUMLAB {
                correct_p += row[c] * targets.data[r * NUMLAB + c];
            }
            total_train_cost += -(correct_p + TINY).ln();
        }

        // Update Supervised Weights
        let mut dc_din_sup = Mat::zeros(BATCH_SIZE, NUMLAB);
        for i in 0..dc_din_sup.data.len() {
            dc_din_sup.data[i] = targets.data[i] - labin.data[i];
        }
        for l in MINLEVELSUP - 1..model.len() {
            if let Some(sw) = &mut model[l].supweights {
                let grad = softmax_norms[l + 1].t_matmul(&dc_din_sup);
                let g_buf = model[l].supweights_grad.as_mut().unwrap();
                for i in 0..g_buf.data.len() {
                    g_buf.data[i] =
                        DELAY * g_buf.data[i] + (1.0 - DELAY) * grad.data[i] / BATCH_SIZE as f32;
                    sw.data[i] += epsgain * EPSILONSUP * (g_buf.data[i] - SUPWC * sw.data[i]);
                }
            }
        }

        // --- NEGATIVE PASS ---
        let mut neg_data = data.clone();
        for r in 0..BATCH_SIZE {
            let mut probs = labin.data[r * NUMLAB..(r + 1) * NUMLAB].to_vec();
            for c in 0..NUMLAB {
                if targets.data[r * NUMLAB + c] > 0.0 {
                    probs[c] = 0.0;
                }
            }
            let sum: f32 = probs.iter().sum();
            let mut cum = 0.0;
            let rv: f32 = rng.random();
            let mut sel = 0;
            for (c, &p) in probs.iter().enumerate() {
                cum += p / sum;
                if rv < cum {
                    sel = c;
                    break;
                }
            }
            for j in 0..NUMLAB {
                neg_data.data[r * 784 + j] = 0.0;
            }
            neg_data.data[r * 784 + sel] = LABELSTRENGTH;
        }
        let mut neg_norm_states = vec![neg_data];
        ffnormrows(&mut neg_norm_states[0]);

        for l in 0..model.len() {
            // Pos Grad
            let mut pos_dc_din = Mat::zeros(BATCH_SIZE, model[l].weights.cols);
            let layer_mean: f32 =
                model[l].mean_states.iter().sum::<f32>() / model[l].mean_states.len() as f32;
            for r in 0..BATCH_SIZE {
                let p = pos_probs[l][r];
                for c in 0..model[l].weights.cols {
                    let st = layer_states[l].data[r * model[l].weights.cols + c];
                    model[l].mean_states[c] =
                        0.9 * model[l].mean_states[c] + 0.1 * (st / BATCH_SIZE as f32);
                    let reg = LAMBDAMEAN * (layer_mean - model[l].mean_states[c]);
                    pos_dc_din.data[r * model[l].weights.cols + c] = (1.0 - p) * st + reg;
                }
            }
            let pos_dw = norm_states[l].t_matmul(&pos_dc_din);

            // Neg pass and Grad
            let (neg_st, neg_nst) = layer_io(&neg_norm_states[l], &model[l]);
            let mut neg_dc_din = Mat::zeros(BATCH_SIZE, model[l].weights.cols);
            for r in 0..BATCH_SIZE {
                let mut sum_sq = 0.0;
                for c in 0..neg_st.cols {
                    sum_sq += neg_st.data[r * neg_st.cols + c].powi(2);
                }
                let p = 1.0 / (1.0 + (-(sum_sq - neg_st.cols as f32) / TEMP).exp());
                for c in 0..neg_st.cols {
                    neg_dc_din.data[r * neg_st.cols + c] = -p * neg_st.data[r * neg_st.cols + c];
                }
            }
            let neg_dw = neg_norm_states[l].t_matmul(&neg_dc_din);
            neg_norm_states.push(neg_nst);

            // Weight and Bias Updates
            for i in 0..model[l].weights.data.len() {
                let g = (pos_dw.data[i] + neg_dw.data[i]) / BATCH_SIZE as f32;
                model[l].weights_grad.data[i] =
                    DELAY * model[l].weights_grad.data[i] + (1.0 - DELAY) * g;
                model[l].weights.data[i] += epsgain
                    * EPSILON
                    * (model[l].weights_grad.data[i] - WC * model[l].weights.data[i]);
            }
            for c in 0..model[l].biases.len() {
                let mut g = 0.0;
                for r in 0..BATCH_SIZE {
                    g += pos_dc_din.data[r * model[l].biases.len() + c]
                        + neg_dc_din.data[r * model[l].biases.len() + c];
                }
                model[l].biases_grad[c] =
                    DELAY * model[l].biases_grad[c] + (1.0 - DELAY) * (g / BATCH_SIZE as f32);
                model[l].biases[c] += epsgain * EPSILON * model[l].biases_grad[c];
            }
        }
    }
    total_train_cost / num_batches as f32
}

fn predict(model: &[Layer], image: &[f32]) -> usize {
    // energy test - pic max energy
    // 1. Prepare neutral input (average label strength)
    let mut input = Mat::zeros(1, image.len());
    input.data.copy_from_slice(image);
    for j in 0..NUMLAB {
        input.data[j] = LABELSTRENGTH / NUMLAB as f32;
    }

    let mut n_st = input;
    ffnormrows(&mut n_st);

    // 2. Forward pass to get normalized states
    let mut softmax_norms = vec![n_st];
    for l in 0..model.len() {
        let (_, nst) = layer_io(&softmax_norms[l], &model[l]);
        softmax_norms.push(nst);
    }

    // 3. Accumulate scores from supweights
    let mut scores = [0.0f32; NUMLAB];
    for l in MINLEVELSUP - 1..model.len() {
        if let Some(sw) = &model[l].supweights {
            let contrib = softmax_norms[l + 1].matmul(sw);
            for c in 0..NUMLAB {
                scores[c] += contrib.data[c];
            }
        }
    }

    // 4. Return index of max score
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

fn fftest(model: &[Layer], images: &[[f32; mnist::NPIXELS]], labels: &[u8]) -> (usize, usize) {
    let errors = images
        .par_iter()
        .zip(labels)
        .filter(|&(img, &label)| predict(model, img) != label as usize)
        .count();
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
        );
        let (errors0, total0) = fftest(&model, &train_imgs[RTRAIN], &data.train_labels[RTRAIN]);
        let (errors1, total1) = fftest(&model, &train_imgs[RVAL], &data.train_labels[RVAL]);
        println!(
            "Epoch {epoch:3} | Cost: {cost:8.4} | Errors; Train: ({errors0}/{total0}), Valid: ({errors1}/{total1})"
        );
    }

    let (errors, total) = fftest(&model, &test_imgs, &data.test_labels);
    println!("Test Errors: ({errors}/{total})");
    Ok(())
}
