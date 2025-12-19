use engram::{Ensemble, HdvClassifier, ImageClassifier, MnistEncoder, calc_accuracy};
use hypervector::binary_hdv::BinaryHDV;
use mnist::{Image, Mnist, error::MnistError};
use rand::prelude::*;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::io::Write;
use std::sync::Arc;

const N: usize = 100;
const PROTOTYPES_PER_CLASS: usize = 500;
//const BETA: f64 = 60.0;
const BETA: f64 = 40.0;
const NUM_CLASSES: usize = 10;

pub struct ModernHopfield {
    memories: Vec<(u8, BinaryHDV<N>)>,  // label, HDV
    encoder: Arc<MnistEncoder<N>>,
}

impl ModernHopfield {
    pub fn new(encoder: Arc<MnistEncoder<N>>) -> Self {
        ModernHopfield {
            memories: Vec::new(),
            encoder,
        }
    }

    pub fn train(&mut self, data: &Mnist, train_hvs: &[BinaryHDV<N>]) {
        println!(
            "Training with Greedy Prototype Selection (Target: {PROTOTYPES_PER_CLASS}/class)..."
        );

        let mut indices: Vec<usize> = (0..train_hvs.len()).collect();
        let mut counts = [0usize; NUM_CLASSES];

        // Bootstrap
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

        // Greedy Loop
        for loop_count in 0.. {
            let mut added_this_round = 0;
            let mut errors = 0;

            indices.shuffle(&mut rand::rng());

            for &i in &indices {
                let lbl = data.train_labels[i];
                let pred = self.predict_hdv(&train_hvs[i]);

                if pred != lbl {
                    errors += 1;
                    if counts[lbl as usize] < PROTOTYPES_PER_CLASS {
                        self.memories.push((lbl, train_hvs[i]));
                        counts[lbl as usize] += 1;
                        added_this_round += 1;
                    }
                }
            }

            print!(
                "Round {loop_count}: Added {added_this_round} prototypes. Total: {total}. Errors: {errors}   \r",
                total = self.memories.len(),
            );
            std::io::stdout().flush().unwrap();

            if added_this_round == 0 || errors == 0 {
                break;
            }
        }
        println!("\nFinal Prototype Counts: {:?}", counts);
    }
}

impl ImageClassifier for ModernHopfield {
    fn predict(&self, im: &Image) -> u8 {
        let h = self.encoder.encode(im);
        self.predict_hdv(&h)
    }
}

impl HdvClassifier<N> for ModernHopfield {
    fn predict_hdv(&self, query: &BinaryHDV<N>) -> u8 {
        let dim = (N * 64) as f64;

        let mut class_energy = [0.0f64; NUM_CLASSES];
        for (label, memory) in &self.memories {
            let dist = query.hamming_distance(memory);
            let sim = 1.0 - (dist as f64 / dim);
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
}

fn main() -> Result<(), MnistError> {
    let data = Mnist::load("MNIST")?;
    let ensemble_size = 5;

    let mut ensemble: Ensemble<ModernHopfield> = Ensemble::with_capacity(ensemble_size);

    for mn in 1..=ensemble_size {
        let mut rng = StdRng::seed_from_u64(42 + mn as u64);

        let encoder = Arc::new(
            MnistEncoder::<N>::new(&mut rng)
                .with_feature_pixel_bag()
                .with_feature_edges(),
        );

        println!("\nEncoding training data for model {mn}...");
        let train_hvs: Vec<BinaryHDV<N>> = data
            .train_images
            .par_iter()
            .map(|im| encoder.encode(im))
            .collect();

        let mut model = ModernHopfield::new(encoder);
        model.train(&data, &train_hvs);

        // Test individual model
        let (correct, total, acc) = calc_accuracy(&data.test_images, &data.test_labels, &model);
        println!("Model {mn} Accuracy: {correct}/{total} = {acc:.2}%");

        ensemble.push(model);

        // Test ensemble
        if ensemble.len() >= 2 {
            let (correct, total, acc) =
                calc_accuracy(&data.test_images, &data.test_labels, &ensemble);
            println!(
                "Ensemble of {} Accuracy: {correct}/{total} = {acc:.2}%",
                ensemble.len()
            );
        }
    }

    Ok(())
}
