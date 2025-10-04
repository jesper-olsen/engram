use engram::{ItemMemory, encode_image, predict};
use hypervector::Accumulator;
use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use mnist::error::MnistError;
use mnist::{self, Mnist};
use rand::rng;
use rand::seq::SliceRandom;

fn main() -> Result<(), MnistError> {
    const N: usize = 100;
    let imem = ItemMemory::new();
    let data = Mnist::load("MNIST")?;
    println!("Read {} training labels", data.train_labels.len());

    println!("Encoding training images...");
    let train_hvs: Vec<BinaryHDV<N>> = data
        .train_images
        .iter()
        .map(|im| encode_image(im.as_u8_array(), &imem))
        .collect();

    println!("Encoding test images...");
    let test_hvs: Vec<BinaryHDV<N>> = data
        .test_images
        .iter()
        .map(|im| encode_image(im.as_u8_array(), &imem))
        .collect();

    let n_epochs = 10;
    let mut accumulators: [BinaryAccumulator<N>; 10] =
        core::array::from_fn(|_| BinaryAccumulator::<N>::new());

    // --- Initial Bundling (Epoch 1) ---
    println!("Epoch:   1 (Bundling)");
    for (i, hdv) in train_hvs.iter().enumerate() {
        accumulators[data.train_labels[i] as usize].add(hdv, 1.0);
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
                let predicted = predict(img_hdv, &models);

                if predicted != true_label {
                    errors += 1;
                    accumulators[true_label as usize].add(img_hdv, 1.0 * lr);
                    accumulators[predicted as usize].add(img_hdv, -1.0 * lr);
                }
            }
            let total = train_hvs.len();
            let correct = total - errors;
            let acc = 100.0 * correct as f64 / total as f64;
            println!("Epoch: {epoch:3} Training Accuracy: {correct:5}/{total} = {acc:.2}%");
        }
    }

    // --- Final Evaluation ---
    let models: [BinaryHDV<N>; 10] = core::array::from_fn(|i| accumulators[i].finalize());

    let correct: usize = test_hvs
        .iter()
        .zip(&data.test_labels)
        .map(|(hdv, &label)| {
            let predicted = predict(hdv, &models);
            (predicted == label) as usize
        })
        .sum();

    let total = data.test_images.len();
    let acc = 100.0 * correct as f64 / total as f64;
    println!("Test Accuracy {correct}/{total} = {acc:.2}%");
    Ok(())
}
