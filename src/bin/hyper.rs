use engram::{ItemMemory, N, encode_image, predict};
use hypervector::Accumulator;
use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use mnist::error::MnistError;
use mnist::{self, Mnist};

fn main() -> Result<(), MnistError> {
    let imem = ItemMemory::new();
    for i in 0..255 {
        let dist = imem.intensities[0].hamming_distance(&imem.intensities[i]);
        println!(
            "ham {i}->{}: {}, 0->{i}: {dist}",
            i + 1,
            imem.intensities[i].hamming_distance(&imem.intensities[i + 1])
        );
    }

    let data = Mnist::load("MNIST")?;
    println!("Read {} labels", data.train_labels.len());

    let mut accumulators: [BinaryAccumulator<N>; 10] =
        core::array::from_fn(|_| BinaryAccumulator::<N>::new());
    for (i, im) in data.train_images.iter().enumerate() {
        let img_hdv = encode_image(im.as_u8_array(), &imem);
        accumulators[data.train_labels[i] as usize].add(&img_hdv, 1.0);
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
    println!("Accuracy {correct}/{total} = {acc:.2}");
    Ok(())
}
