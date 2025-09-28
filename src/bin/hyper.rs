use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use hypervector::Accumulator;
use std::path::PathBuf;
use mnist;
use mnist::error::MnistError;
use engram::{N,predict,encode_image,ItemMemory};


fn train() -> Result<(),MnistError> {
    let dir: PathBuf = PathBuf::from("MNIST/");
    let fname = dir.join("train-labels-idx1-ubyte");
    let labels = mnist::read_labels(&fname)?;
    println!("Read {} labels", labels.len());

    let fname = dir.join("train-images-idx3-ubyte");
    let images = mnist::read_images(&fname).unwrap();
    mnist::plot(&images[0], labels[0]);
    Ok(())
}

fn main() -> Result<(),MnistError> {
    let imem = ItemMemory::new();
    for i in 0..255 {
        let dist = imem.intensities[0].hamming_distance(&imem.intensities[i]);
        println!(
            "ham {i}->{}: {}, 0->{i}: {dist}",
            i + 1,
            imem.intensities[i].hamming_distance(&imem.intensities[i + 1])
        );
    }

    let dir: PathBuf = PathBuf::from("MNIST/");
    let fname = dir.join("train-labels-idx1-ubyte");
    let labels = mnist::read_labels(&fname)?;
    println!("Read {} labels", labels.len());

    let fname = dir.join("train-images-idx3-ubyte");
    let images = mnist::read_images(&fname)?;

    let fname = dir.join("t10k-labels-idx1-ubyte");
    let test_labels = mnist::read_labels(&fname)?;

    let fname = dir.join("t10k-images-idx3-ubyte");
    let test_images = mnist::read_images(&fname)?;

    let mut accumulators: [BinaryAccumulator<N>; 10] =
        core::array::from_fn(|_| BinaryAccumulator::<N>::new());
    for (i, im) in images.iter().enumerate() {
        let img_hdv = encode_image(im.as_u8_array(), &imem);
        //let img_hdv = encode_image_2d(im.as_u8_array(), &imem);
        accumulators[labels[i] as usize].add(&img_hdv, 1.0);
    }
    let models: [BinaryHDV<N>; 10] = core::array::from_fn(|i| accumulators[i].finalize());

    let mut correct = 0;
    for (i, im) in test_images.iter().enumerate() {
        println!("{im}");
        let img_hdv = encode_image(im.as_u8_array(), &imem);
        //let img_hdv = encode_image_2d(im.as_u8_array(), &imem);
        let predicted = predict(&img_hdv, &models);
        if predicted == test_labels[i] {
            correct += 1;
        }
    }
    let total = test_images.len();
    let acc = 100.0 * correct as f64 / total as f64;
    println!("Accuracy {correct}/{total} = {acc:.2}");
    Ok(())
}
