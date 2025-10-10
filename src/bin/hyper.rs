use engram::{ItemMemory, MultiChannelAccumulator, MultiChannelHDV};
use mnist::error::MnistError;
use mnist::{self, Mnist};
use rand::rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;

fn calc_channel_weights<const N: usize, const M: usize>(
    models: &[MultiChannelHDV<N, M>],
) -> [f32; M] {
    let mut spread = [0f32; M];
    for channel in 0..M {
        let mut dist = 0;
        for d1 in 0..10 {
            for d2 in (d1 + 1)..10 {
                dist += models[d1].hdvs[channel].hamming_distance(&models[d2].hdvs[channel]);
            }
        }
        spread[channel] = dist as f32;
        //println!("Channel {channel}, Spread {dist}");
    }
    let total: f32 = spread.iter().sum();
    if total > 0.0 {
        spread.iter_mut().for_each(|e| *e /= total);
    }
    spread
}

fn calc_test_accuracy<const N: usize, const M: usize>(
    test_hvs: &[MultiChannelHDV<N, M>],
    test_labels: &[u8],
    models: &[MultiChannelHDV<N, M>; 10],
) -> (usize, usize, f64) {
    let weights = calc_channel_weights(models);
    let correct: usize = test_hvs
        .iter()
        .zip(test_labels)
        .map(|(hdv, &label)| {
            let predicted = hdv.predict(models, &weights);
            (predicted == label) as usize
        })
        .sum();

    let total = test_hvs.len();
    let acc = if total > 0 {
        100.0 * correct as f64 / total as f64
    } else {
        0.0
    };
    (correct, total, acc)
}

fn main() -> Result<(), MnistError> {
    const N: usize = 200;
    const M: usize = 4;
    let imem = ItemMemory::<N>::new();
    let data = Mnist::load("MNIST")?;
    println!("Read {} training labels", data.train_labels.len());

    println!("Encoding training images...");
    //let train_hvs: Vec<BinaryHDV<N>> = data
    let train_hvs: Vec<MultiChannelHDV<N, M>> = data
        .train_images
        .par_iter()
        .map(|im| MultiChannelHDV::<N, M>::encode_image4(im.as_u8_array(), &imem))
        .collect();

    println!("Encoding test images...");
    //let test_hvs: Vec<BinaryHDV<N>> = data
    let test_hvs: Vec<MultiChannelHDV<N, M>> = data
        .test_images
        .par_iter()
        .map(|im| MultiChannelHDV::<N, M>::encode_image4(im.as_u8_array(), &imem))
        .collect();

    let n_epochs = 5000;
    let mut accumulators: [MultiChannelAccumulator<N, M>; 10] =
        core::array::from_fn(|_| MultiChannelAccumulator::<N, M>::new());

    // --- Initial Bundling (Epoch 1) ---
    println!("Epoch:   1 (Bundling)");
    for (i, hdv) in train_hvs.iter().enumerate() {
        accumulators[data.train_labels[i] as usize].add(hdv, 1.0);
    }

    let weights = [1.0; M];
    // --- Iterative Correction ---
    if n_epochs > 1 {
        // Shuffle data for each epoch
        let mut train_indices: Vec<usize> = (0..train_hvs.len()).collect();

        for epoch in 2..=n_epochs {
            train_indices.shuffle(&mut rng());

            let models: [MultiChannelHDV<N, M>; 10] =
                core::array::from_fn(|i| accumulators[i].finalize());
            let mut errors = 0;
            let lr = 1.0 / (epoch as f64).sqrt();

            for &i in &train_indices {
                let img_hdv = &train_hvs[i];
                let true_label = data.train_labels[i];
                let predicted = img_hdv.predict(&models, &weights);

                if predicted != true_label {
                    errors += 1;
                    accumulators[true_label as usize].add(img_hdv, lr);
                    accumulators[predicted as usize].add(img_hdv, -lr);
                }
            }
            let total = train_hvs.len();
            let correct = total - errors;
            let acc = 100.0 * correct as f64 / total as f64;

            let (_test_correct, _test_total, test_acc) =
                calc_test_accuracy(&test_hvs, &data.test_labels, &models);
            //println!("Epoch: {epoch:3} Training Accuracy: {correct:5}/{total} = {acc:.2}%");
            println!(
                "Epoch: {epoch:3} Training Accuracy: {correct:5}/{total} = {acc:.2}% test {test_acc:.2}%"
            );
            if errors == 0 {
                break;
            }
        }
    }

    // --- Final Evaluation ---
    let models: [MultiChannelHDV<N, M>; 10] = core::array::from_fn(|i| accumulators[i].finalize());

    let (test_correct, test_total, test_acc) =
        calc_test_accuracy(&test_hvs, &data.test_labels, &models);
    println!("Test Accuracy {test_correct}/{test_total} = {test_acc:.2}%");

    // for (channel, sp) in weights.iter().enumerate() {
    //     println!("Channel {channel}, Spread {:.4}", sp);
    // }

    Ok(())
}
