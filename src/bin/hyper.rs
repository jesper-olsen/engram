use engram::{
    ItemMemory, MultiChannelAccumulator, MultiChannelHDV, encode_image, encode_image3, predict,
};
//use hypervector::Accumulator;
//use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use mnist::error::MnistError;
use mnist::{self, Mnist};
use rand::rng;
use rand::seq::SliceRandom;

fn spread<const N: usize, const M: usize>(models: &[MultiChannelHDV<N, M>]) {
    let mut spread = [0f64; M];
    for channel in 0..M {
        let mut dist = 0;
        for d1 in 0..10 {
            for d2 in (d1+1)..10 {
                dist += models[d1].hdvs[channel].hamming_distance(&models[d2].hdvs[channel]);
            }
        }
        spread[channel] = dist as f64;
        println!("Channel {channel}, Spread {dist}");
    }
    let total: f64 = spread.iter().sum();
    if total > 0.0 {
        for (channel, sp) in spread.iter().enumerate() {
            println!("Channel {channel}, Spread {:.2}", sp / total);
        }
    }
}

fn main() -> Result<(), MnistError> {
    const N: usize = 100;
    let imem = ItemMemory::<N>::new();
    let data = Mnist::load("MNIST")?;
    println!("Read {} training labels", data.train_labels.len());

    const M: usize = 3;

    println!("Encoding training images...");
    //let train_hvs: Vec<BinaryHDV<N>> = data
    let train_hvs: Vec<MultiChannelHDV<N, M>> = data
        .train_images
        .iter()
        .map(|im| encode_image3(im.as_u8_array(), &imem))
        .collect();

    println!("Encoding test images...");
    //let test_hvs: Vec<BinaryHDV<N>> = data
    let test_hvs: Vec<MultiChannelHDV<N, M>> = data
        .test_images
        .iter()
        .map(|im| encode_image3(im.as_u8_array(), &imem))
        .collect();

    let n_epochs = 10;
    let mut accumulators: [MultiChannelAccumulator<N, M>; 10] =
        core::array::from_fn(|_| MultiChannelAccumulator::<N, M>::new());

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

            let models: [MultiChannelHDV<N, M>; 10] =
                core::array::from_fn(|i| accumulators[i].finalize());
            let mut errors = 0;
            let lr = 1.0 / (epoch as f64).sqrt();

            for &i in &train_indices {
                let img_hdv = &train_hvs[i];
                let true_label = data.train_labels[i];
                let predicted = img_hdv.predict(&models, &[29,38,33]);

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
    let models: [MultiChannelHDV<N, M>; 10] = core::array::from_fn(|i| accumulators[i].finalize());

    let correct: usize = test_hvs
        .iter()
        .zip(&data.test_labels)
        .map(|(hdv, &label)| {
            let predicted = hdv.predict(&models, &[1, 1, 1]);
            (predicted == label) as usize
        })
        .sum();

    let total = data.test_images.len();
    let acc = 100.0 * correct as f64 / total as f64;
    println!("Test Accuracy {correct}/{total} = {acc:.2}%");

    spread(&models);

    Ok(())
}
