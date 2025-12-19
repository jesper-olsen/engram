use engram::MnistEncoder;
use hypervector::binary_hdv::BinaryHDV;
use mnist::{self, Image, Mnist, error::MnistError};
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::io::Write;
use std::sync::Arc; // Use Arc to share the encoder

const N: usize = 100;
// 100 prototypes x 10 classes = 1000 vectors.
// This is still tiny (KB) compared to a Neural Net, but gives high resolution.
//const PROTOTYPES_PER_CLASS: usize = 100;
//const BETA: f64 = 30.0;

//const PROTOTYPES_PER_CLASS: usize = 250;
const PROTOTYPES_PER_CLASS: usize = 500;
const BETA: f64 = 60.0;

struct ModernHopfield {
    memories: Vec<(u8, BinaryHDV<N>)>,
    // Share the specific encoder instance used for training
    encoder: Arc<MnistEncoder<N>>,
}

impl ModernHopfield {
    fn new(encoder: Arc<MnistEncoder<N>>) -> Self {
        ModernHopfield {
            memories: Vec::new(),
            encoder,
        }
    }

    fn predict(&self, im: &Image) -> u8 {
        let h = self.encoder.encode(im);
        self.predict_hdv(&h)
    }

    fn predict_hdv(&self, query: &BinaryHDV<N>) -> u8 {
        let mut class_energy = [0.0f64; 10];
        // Pre-calculate dimension constant
        let dim = (N * 64) as f64;

        for (label, memory) in &self.memories {
            let dist = query.hamming_distance(memory);

            // Similarity: 1.0 (Identical) to 0.0 (Inverse)
            // Note: Random vectors will have sim ~0.5
            let sim = 1.0 - (dist as f64 / dim);

            // Softmax / Energy term
            let energy = (BETA * sim).exp();
            class_energy[*label as usize] += energy;
        }

        class_energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u8)
            .unwrap_or(0)
    }

    fn train(&mut self, data: &Mnist, train_hvs: &[BinaryHDV<N>]) {
        println!(
            "Training with Greedy Prototype Selection (Target: {}/class)...",
            PROTOTYPES_PER_CLASS
        );

        let mut indices: Vec<usize> = (0..train_hvs.len()).collect();
        let mut counts = [0usize; 10];

        // 1. Bootstrap
        indices.shuffle(&mut rand::rng());
        for &i in &indices {
            let lbl = data.train_labels[i];
            if counts[lbl as usize] == 0 {
                self.memories.push((lbl, train_hvs[i].clone()));
                counts[lbl as usize] += 1;
            }
            if counts.iter().all(|&c| c > 0) {
                break;
            }
        }

        // 2. Greedy Loop
        let mut loop_count = 0;
        loop {
            loop_count += 1;
            let mut added_this_round = 0;
            let mut errors = 0;

            indices.shuffle(&mut rand::rng());

            for &i in &indices {
                let lbl = data.train_labels[i];
                let is_full = counts[lbl as usize] >= PROTOTYPES_PER_CLASS;

                // Check prediction with current memory bank
                let pred = self.predict_hdv(&train_hvs[i]);

                if pred != lbl {
                    errors += 1;
                    // Only add if we have "storage" slots left for this class
                    if !is_full {
                        self.memories.push((lbl, train_hvs[i].clone()));
                        counts[lbl as usize] += 1;
                        added_this_round += 1;
                    }
                }
            }

            let total_memories = self.memories.len();
            print!(
                "Round {}: Added {} prototypes. Total Mem: {}. Training Errors: {}   \r",
                loop_count, added_this_round, total_memories, errors
            );
            std::io::stdout().flush().unwrap();

            // Exit conditions:
            // 1. We didn't get any errors (Perfect training accuracy)
            // 2. We got errors, but couldn't add any new prototypes (Memory banks full)
            if added_this_round == 0 || errors == 0 {
                break;
            }
        }
        println!("\nFinal Prototype Counts: {:?}", counts);
    }
}

fn main() -> Result<(), MnistError> {
    let data = Mnist::load("MNIST")?;
    let mut rng = StdRng::seed_from_u64(42);

    // 1. Create the Master Encoder
    let encoder = Arc::new(
        MnistEncoder::<N>::new(&mut rng)
            .with_feature_pixel_bag()
            .with_feature_edges(),
    );

    println!("Encoding training data...");
    let train_hvs: Vec<BinaryHDV<N>> = data
        .train_images
        .par_iter()
        .map(|im| encoder.encode(im))
        .collect();

    // 2. Pass the ARC pointer to the model
    let mut model = ModernHopfield::new(encoder.clone());

    // Train
    model.train(&data, &train_hvs);

    // Test
    println!("Testing...");
    let correct = data
        .test_images
        .par_iter()
        .zip(&data.test_labels)
        .map(|(im, &lbl)| if model.predict(im) == lbl { 1 } else { 0 })
        .sum::<usize>();

    let total = data.test_images.len();
    println!(
        "Accuracy: {:.2}% ({}/{})",
        100.0 * correct as f64 / total as f64,
        correct,
        total
    );

    Ok(())
}
