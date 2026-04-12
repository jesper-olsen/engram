use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;
use hypervector::types::binary::BinaryHDV;
use hypervector::types::traits::HyperVector;
use hypervector::hdv; 
use hypervector::trainer::{Classifier, kmeans::KMeans};
use mnist::error::MnistError;
use mnist::{self, Mnist};
use engram::MnistEncoder;

pub struct KMeansClassifier<T: HyperVector>(pub Vec<KMeans<T>>);

impl<T: HyperVector + Send + Sync> Classifier<T> for KMeansClassifier<T> {
    fn predict(&self, h: &T) -> usize {
        self.0
            .iter()
            .enumerate()
            .map(|(i, cb)| (i, cb.nearest(h).1))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }
}

fn compute_codebooks<T: HyperVector + Send + Sync, R: Rng>(
    train_hvs: &[T],
    labels: &[u8],
    k: usize,
    rng: &mut R,
) -> KMeansClassifier<T> {
    let codebooks = (0..10u8)
        .map(|digit| {
            let hvs: Vec<&T> = train_hvs
                .iter()
                .zip(labels.iter())
                .filter(|&(_, &lab)| lab == digit)
                .map(|(h, _)| h)
                .collect();
            let mut cb = KMeans::new(&hvs, k, rng);
            cb.train(&hvs, 100, false);
            cb
        })
        .collect();
    KMeansClassifier(codebooks)
}

fn main() -> Result<(), MnistError> {
    const TOTAL_BITS: usize = 6400;
    hdv!(binary, HDV, TOTAL_BITS);

    let mut rng = StdRng::seed_from_u64(42);
    let imem = MnistEncoder::<HDV>::new(&mut rng)
        .with_feature_pixel_bag()
        .with_feature_edges();
    let data = Mnist::load("MNIST")?;
    println!("Read {} training labels", data.train_labels.len());

    println!("Encoding training images...");
    let train_hvs: Vec<HDV> = data
        .train_images
        .par_iter()
        .map(|im| imem.encode(im))
        .collect();

    println!("Encoding test images...");
    let test_hvs: Vec<HDV> = data
        .test_images
        .par_iter()
        .map(|im| imem.encode(im))
        .collect();

    let classifier = compute_codebooks(&train_hvs, &data.train_labels, 1, &mut rng);
    //let prototypes: [HDV; 10] = std::array::from_fn(|i| {
    //    classifier.0[i].centroids[0].clone()
    //});
    //let classifier = PrototypeModel { prototypes };
    println!("\n--- Centroid-to-Centroid Distances ---");
    println!("(Total bits = {TOTAL_BITS})");
    println!("        0      1      2      3      4      5      6      7      8      9");
    println!("------------------------------------------------------------------------");
    for d1 in 0..10 {
        print!("{d1}: ");
        for d2 in 0..10 {
            let dist = classifier.0[d1].centroids[0].distance(&classifier.0[d2].centroids[0]);
            print!("{dist:6.4} ");
        }
        println!();
    }

    println!("\n--- Average Test Data Distance to Centroids ---");
    println!("       Centroid ->");
    println!("True    0       1       2       3       4       5       6       7       8       9");
    println!("----------------------------------------------------------------------------------");

    for true_digit in 0..10u8 {
        let digit_test_hvs: Vec<&HDV> = test_hvs
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
        for cb in &classifier.0 {
            let total_distance: f32 = digit_test_hvs
                .par_iter()
                .map(|&hdv| hdv.distance(&cb.centroids[0]))
                .sum();
            let avg_distance = total_distance / num_samples as f32;
            print!("{avg_distance:<7.4} ");
        }
        println!();
    }

    println!("\nClassify - codebook size");
    for n in 1..=20 {
        let classifier = compute_codebooks(&train_hvs, &data.train_labels, n, &mut rng);
        print!("{n:2}: ");
        let (correct, _errors, acc) = classifier.accuracy(&test_hvs, &data.test_labels);
        println!("Accuracy {correct}/{}={acc:.2}%", test_hvs.len());
    }

    Ok(())
}
