use engram::MnistEncoder;
use engram::kmeans::KMeans;
use hypervector::binary_hdv::BinaryHDV;
use mnist::error::MnistError;
use mnist::{self, Mnist};
//use rand::rng;
//use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

fn compute_codebooks<const N: usize>(
    train_hvs: &[BinaryHDV<N>],
    labels: &[u8],
    k: usize,
) -> Vec<KMeans<N>> {
    let mut cbs = Vec::with_capacity(10);
    for digit in 0..10 {
        // Select all encoded images (hypervectors) whose label == i
        let hvs: Vec<&BinaryHDV<N>> = train_hvs
            .iter()
            .zip(labels.iter())
            .filter(|&(_, &lab)| lab == digit)
            .map(|(h, _)| h)
            .collect();

        //println!("Digit {digit}: {} samples", hvs.len());

        let mut cb = KMeans::new(&hvs, k);
        let max_iter = 100;
        let verbose = false;
        cb.train(&hvs, max_iter, verbose);
        cbs.push(cb);
    }
    cbs
}

fn classify<const N: usize>(test_hvs: &[BinaryHDV<N>], labels: &[u8], models: &[KMeans<N>]) {
    let mut correct = 0;
    for (hv, digit) in test_hvs.iter().zip(labels.iter()) {
        let mut min_dist = u32::MAX;
        let mut best_cb = 0;
        for (i, cb) in models.iter().enumerate() {
            let (_cluster, dist) = cb.nearest(hv);
            if dist < min_dist {
                min_dist = dist;
                best_cb = i;
            }
        }
        correct += (best_cb as u8 == *digit) as usize;
    }
    let total = labels.len();

    if total > 0 {
        println!(
            "Accuracy {correct}/{total} = {:.2}%",
            100.0 * correct as f64 / total as f64
        );
    }
}

fn main() -> Result<(), MnistError> {
    const N: usize = 100;
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    let imem = MnistEncoder::<N>::new(&mut rng)
        .with_feature_pixel_bag()
        .with_feature_edges();
    let data = Mnist::load("MNIST")?;
    println!("Read {} training labels", data.train_labels.len());

    println!("Encoding training images...");
    let train_hvs: Vec<BinaryHDV<N>> = data
        .train_images
        .par_iter()
        .map(|im| imem.encode(im))
        .collect();

    println!("Encoding test images...");
    let test_hvs: Vec<BinaryHDV<N>> = data
        .test_images
        .par_iter()
        .map(|im| imem.encode(im))
        .collect();

    let cbs: Vec<KMeans<N>> = compute_codebooks(&train_hvs, &data.train_labels, 1);

    println!("\n--- Centroid-to-Centroid Distances ---");
    let total_bits = N * std::mem::size_of::<usize>() * 8;
    println!("(Total bits = {total_bits})");

    println!("        0      1      2      3      4      5      6      7      8      9");
    println!("------------------------------------------------------------------------");
    for d1 in 0..10 {
        print!("{d1}: ");
        for d2 in 0..10 {
            let dist = cbs[d1].centroids[0].hamming_distance(&cbs[d2].centroids[0]);
            print!("{dist:6} ");
        }
        println!();
    }

    println!("\n--- Average Test Data Distance to Centroids ---");
    println!("       Centroid ->");
    println!("True    0       1       2       3       4       5       6       7       8       9");
    println!("----------------------------------------------------------------------------------");

    let centroids: Vec<BinaryHDV<N>> = cbs.iter().map(|cb| cb.centroids[0]).collect();
    for true_digit in 0..10u8 {
        // Filter test HVs to get only those for the current true_digit
        let digit_test_hvs: Vec<&BinaryHDV<N>> = test_hvs
            .iter()
            .zip(data.test_labels.iter())
            .filter(|&(_, &label)| label == true_digit)
            .map(|(hdv, _)| hdv)
            .collect();

        let num_samples = digit_test_hvs.len();
        if num_samples == 0 {
            continue;
        }

        print!("{true_digit:2} | ");

        // For this set of HVs, calculate avg distance to EACH centroid
        for centroid in &centroids {
            let total_distance: u32 = digit_test_hvs
                .par_iter()
                .map(|&hdv| hdv.hamming_distance(centroid))
                .sum();

            let avg_distance = total_distance as f64 / num_samples as f64;
            print!("{avg_distance:<7.0} ");
        }
        println!();
    }

    println!("\nClassify - codebook size");
    for n in 1..=20 {
        let cbs: Vec<KMeans<N>> = compute_codebooks(&train_hvs, &data.train_labels, n);
        print!("{n:2}: ");
        classify(&test_hvs, &data.test_labels, &cbs);
    }

    Ok(())
}
