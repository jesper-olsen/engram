use engram::{ItemMemory, encode_image};
use hypervector::{
    Accumulator,
    binary_hdv::{BinaryAccumulator, BinaryHDV},
};
use mnist::{self, Mnist, error::MnistError};
use rand::{rng, seq::SliceRandom};
use rayon::prelude::*;

fn calc_test_accuracy<const N: usize>(
    test_hvs: &[BinaryHDV<N>],
    test_labels: &[u8],
    models: &[BinaryHDV<N>; 10],
) -> (usize, usize, f64) {
    let correct: usize = test_hvs
        .iter()
        .zip(test_labels)
        .map(|(hdv, &label)| {
            let predicted = engram::predict(hdv, models);
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
    const N: usize = 1600;
    let imem = ItemMemory::<N>::new();
    let data = Mnist::load("MNIST")?;
    println!("Read {} training labels", data.train_labels.len());

    println!("Encoding training images...");
    let train_hvs: Vec<BinaryHDV<N>> = data
        .train_images
        .par_iter()
        .map(|im| encode_image(im.as_u8_array(), &imem))
        .collect();

    println!("Encoding test images...");
    let test_hvs: Vec<BinaryHDV<N>> = data
        .test_images
        .par_iter()
        .map(|im| encode_image(im.as_u8_array(), &imem))
        .collect();

    let n_epochs = 5000;
    let mut accumulators: [BinaryAccumulator<N>; 10] =
        core::array::from_fn(|_| BinaryAccumulator::<N>::new());

    // --- Initial Bundling (Epoch 1) ---
    println!("Epoch:   1 (Bundling)");
    for (i, hdv) in train_hvs.iter().enumerate() {
        let digit = data.train_labels[i] as usize;
        accumulators[digit].add(hdv, 1.0);
    }

    // --- Iterative Correction ---
    if n_epochs > 1 {
        // Shuffle data for each epoch
        let mut train_indices: Vec<usize> = (0..train_hvs.len()).collect();

        for epoch in 2..=n_epochs {
            train_indices.shuffle(&mut rng());

            let models: [BinaryHDV<N>; 10] = core::array::from_fn(|i| accumulators[i].finalize());
            let mut errors = 0;
            let lr = 1.0 / (epoch as f64).sqrt();

            for &i in &train_indices {
                let img_hdv = &train_hvs[i];
                let true_label = data.train_labels[i];
                let predicted = engram::predict(img_hdv, &models);
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
            println!(
                "Epoch: {epoch:3} Training Accuracy: {correct:5}/{total} = {acc:.2}% test {test_acc:.2}%"
            );
            if errors == 0 {
                break;
            }
        }
    }

    // --- Final Evaluation ---
    let models: [BinaryHDV<N>; 10] = core::array::from_fn(|i| accumulators[i].finalize());

    let (test_correct, test_total, test_acc) =
        calc_test_accuracy(&test_hvs, &data.test_labels, &models);
    println!("Test Accuracy {test_correct}/{test_total} = {test_acc:.2}%");

    Ok(())
}
