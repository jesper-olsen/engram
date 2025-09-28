use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use hypervector::{Accumulator, HyperVector};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use std::process;
use std::path::PathBuf;
use mnist;
use mnist::error::MnistError;

const N: usize = 157;

struct ItemMemory {
    positions: [BinaryHDV<N>; 784],
    intensities: Vec<BinaryHDV<N>>,
}

impl ItemMemory {
    pub fn new() -> Self {
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let positions = core::array::from_fn(|_| BinaryHDV::<N>::random(&mut rng));

        let intensity_min = BinaryHDV::<N>::random(&mut rng);
        let intensity_max = BinaryHDV::<N>::random(&mut rng);
        let mut intensities = Vec::with_capacity(256);
        intensities.push(intensity_min);

        const DIM: usize = BinaryHDV::<N>::DIM;
        let mut permutations: [usize; DIM] = core::array::from_fn(|i| i);
        permutations.shuffle(&mut rng);

        for i in 1..255 {
            let bit = (i as f64 / 255.0) * DIM as f64;
            let bit = bit as usize;
            let bit = bit.min(DIM);
            let bi = BinaryHDV::<N>::blend(&intensity_min, &intensity_max, &permutations[..bit]);
            //let bi = BinaryHDV::<N>::flip(&intensities[i - 1], nflip, &mut rng);
            intensities.push(bi);
        }
        intensities.push(intensity_max);
        ItemMemory {
            positions,
            intensities,
        }
    }
}

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

fn encode_image(pixels: &[u8], item_memory: &ItemMemory) -> BinaryHDV<N> {
    assert!(pixels.len() == 784);
    let mut accumulator = BinaryAccumulator::new();
    //let threshold = 10;
    //let threshold = 30;
    let threshold = 0;

    for (i, &intensity) in pixels.iter().enumerate() {
        if intensity > threshold {
            let pos_hdv = &item_memory.positions[i];
            let intensity_hdv = &item_memory.intensities[intensity as usize];
            let pixel_hdv = pos_hdv.bind(&intensity_hdv);
            accumulator.add(&pixel_hdv, 1.0);
        }
    }

    accumulator.finalize()
}

fn encode_image2D(pixels: &[u8], item_memory: &ItemMemory) -> BinaryHDV<N> {
    assert!(pixels.len() == 784);
    let mut accumulator = BinaryAccumulator::new();
    let threshold = 10;
    const WIDTH: usize = 28;
    const HEIGHT: usize = 28;

    // Pre-calculate individual pixel HDVs to avoid redundant work
    let pixel_hdvs: Vec<BinaryHDV<N>> = pixels
        .iter()
        .enumerate()
        .map(|(i, &intensity)| {
            if intensity > threshold {
                let pos_hdv = &item_memory.positions[i];
                let intensity_hdv = &item_memory.intensities[intensity as usize];
                pos_hdv.bind(intensity_hdv)
            } else {
                BinaryHDV::zero()
            }
        })
        .collect();

    // Iterate through each pixel to form the top-left corner of our N-grams
    for r in 0..(HEIGHT - 1) {
        // -1 to avoid vertical out-of-bounds
        for c in 0..(WIDTH - 1) {
            // -1 to avoid horizontal out-of-bounds
            let i = r * WIDTH + c;

            // --- Horizontal Trigram ---
            // We can simplify and use 2-grams (bigrams) for efficiency and power
            let h_v0 = &pixel_hdvs[i];
            let h_v1 = &pixel_hdvs[i + 1]; // Pixel to the right

            // Bind the pixel with a permuted version of its right neighbor
            // permute(1) can represent "right"
            let horizontal_bigram = h_v0.bind(&h_v1.permute(1));
            if !horizontal_bigram.is_zero() {
                accumulator.add(&horizontal_bigram, 1.0);
            }

            // --- Vertical Trigram ---
            let v_v0 = &pixel_hdvs[i];
            let v_v1 = &pixel_hdvs[i + WIDTH]; // Pixel below

            // Bind the pixel with a differently permuted version of its bottom neighbor
            // permute(2) can represent "down"
            let vertical_bigram = v_v0.bind(&v_v1.permute(2));
            if !vertical_bigram.is_zero() {
                accumulator.add(&vertical_bigram, 1.0);
            }
        }
    }

    accumulator.finalize()
}

fn predict(img_hdv: &BinaryHDV<N>, models: &[BinaryHDV<N>]) -> u8 {
    let mut min_dist = usize::MAX;
    let mut best_model = 0;
    for j in 0..10 {
        let dist = img_hdv.hamming_distance(&models[j]);
        if dist < min_dist {
            min_dist = dist;
            best_model = j;
        }
    }
    best_model as u8
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
    //    for i in 1..256 {
    //        let dist = imem.intensities[0].hamming_distance(&imem.intensities[i]);
    //        let dist2 = imem.intensities[255].hamming_distance(&imem.intensities[i]);
    //        println!("0 -> {i}: {dist} {dist2}");
    //    }

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
        //let img_hdv = encode_image(im.as_u8_array(), &imem);
        let img_hdv = encode_image2D(im.as_u8_array(), &imem);
        accumulators[labels[i] as usize].add(&img_hdv, 1.0);
    }
    let models: [BinaryHDV<N>; 10] = core::array::from_fn(|i| accumulators[i].finalize());

    let mut correct = 0;
    for (i, im) in test_images.iter().enumerate() {
        println!("{im}");
        //let img_hdv = encode_image(im.as_u8_array(), &imem);
        let img_hdv = encode_image2D(im.as_u8_array(), &imem);
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
