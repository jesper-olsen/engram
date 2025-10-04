use engram::{ItemMemory, encode_image, predict};
use hypervector::Accumulator;
use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use mnist::error::MnistError;
use mnist::{self, Mnist};

fn main() -> Result<(), MnistError> {
    let imem = ItemMemory::new();
    let data = Mnist::load("MNIST")?;
    println!("Read {} labels", data.train_labels.len());

    const N: usize = 200;
    let n_epochs = 10;
    let mut accumulators: [BinaryAccumulator<N>; 10] =
        core::array::from_fn(|_| BinaryAccumulator::<N>::new());
    println!("Epoch:   1");
    for (i, im) in data.train_images.iter().enumerate() {
        let img_hdv = encode_image(im.as_u8_array(), &imem);
        accumulators[data.train_labels[i] as usize].add(&img_hdv, 1.0);
    }

    if n_epochs > 1 {
        for epoch in 2..=n_epochs {
            let models: [BinaryHDV<N>; 10] = core::array::from_fn(|i| accumulators[i].finalize());
            let mut errors = 0;
            let lr = 2.0/epoch as f64;
            for (i, im) in data.train_images.iter().enumerate() {
                let img_hdv = encode_image(im.as_u8_array(), &imem);
                let predicted = predict(&img_hdv, &models);
                let true_label = data.train_labels[i];
                if predicted != true_label {
                    errors += 1;
                    accumulators[true_label as usize].add(&img_hdv, 1.0*lr);
                    accumulators[predicted as usize].add(&img_hdv, -1.0*lr);
                }
            }
            let total = data.train_images.len();
            let correct = total - errors;
            let acc = 100.0 * correct as f64 / total as f64;
            println!("Epoch: {epoch:3} Training Accuracy: {correct:4}/{total} = {acc:.2}%");
        }
    }

    let models: [BinaryHDV<N>; 10] = core::array::from_fn(|i| accumulators[i].finalize());

    let correct: usize = data
        .test_images
        .iter()
        .zip(&data.test_labels)
        .map(|(im, &label)| {
            let img_hdv = encode_image(im.as_u8_array(), &imem);
            let predicted = predict(&img_hdv, &models);
            (predicted == label) as usize
        })
        .sum();

    let total = data.test_images.len();
    let acc = 100.0 * correct as f64 / total as f64;
    println!("Test Accuracy {correct}/{total} = {acc:.2}%");
    Ok(())
}
