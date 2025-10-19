use engram::MnistEncoder;
use hypervector::{
    Accumulator,
    binary_hdv::{BinaryAccumulator, BinaryHDV},
};
use mnist::{self, Image, Mnist, error::MnistError};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::{rng, seq::SliceRandom};
use rayon::prelude::*;
use std::io::Write;

const NUM_CLASSES: usize = 10;

struct Model<const N: usize> {
    hdvs: [BinaryHDV<N>; NUM_CLASSES],
    encoder: MnistEncoder<N>,
}

struct EnsembleModel<const N: usize> {
    models: Vec<Model<N>>,
}

impl<const N: usize> Classifier<N> for Model<N> {
    fn predict(&self, im: &Image) -> u8 {
        let h = self.encoder.encode(im);
        self.predict_hdv(&h)
    }

    fn predict_hdv(&self, h: &BinaryHDV<N>) -> u8 {
        let mut min_dist = u32::MAX;
        let mut best_model = 0;
        for (j, model) in self.hdvs.iter().enumerate() {
            let dist = model.hamming_distance(h);
            if dist < min_dist {
                min_dist = dist;
                best_model = j;
            }
        }
        best_model as u8
    }
}

impl<const N: usize> Classifier<N> for EnsembleModel<N> {
    // add individual distances
    //fn predict(&self, im: &Image) -> u8 {
    //    let mut min_dist = usize::MAX;
    //    let mut best_class = 0;
    //    for i in 0..NUM_CLASSES {
    //        let dist: usize = self
    //            .models
    //            .iter()
    //            .map(|model| {
    //                let h =model.encoder.encode(im.as_u8_array());
    //                model.hdvs[i].hamming_distance(&h)
    //            })
    //            .sum();
    //        if dist < min_dist {
    //            min_dist = dist;
    //            best_class = i;
    //        }
    //    }
    //    best_class as u8
    //}

    fn predict(&self, im: &Image) -> u8 {
        let mut votes = [0u8; NUM_CLASSES];

        // Get a prediction from each model in the ensemble
        for model in self.models.iter() {
            let prediction = model.predict(im);
            votes[prediction as usize] += 1;
        }

        // Find the digit that received the most votes
        votes
            .iter()
            .enumerate()
            .max_by_key(|&(_, &count)| count)
            .map(|(digit, _)| digit as u8)
            .unwrap_or(0) // Default to 0 in case of an empty ensemble
    }

    fn predict_hdv(&self, _: &BinaryHDV<N>) -> u8 {
        unimplemented!("Not implemented for ensemble model")
    }
}

pub trait Classifier<const N: usize> {
    fn predict(&self, h: &Image) -> u8;
    fn predict_hdv(&self, h: &BinaryHDV<N>) -> u8;
}

fn calc_accuracy<const N: usize, M: Classifier<N> + Sync>(
    test_images: &[Image],
    test_labels: &[u8],
    model: &M,
) -> (usize, usize, f64) {
    let correct: usize = test_images
        .par_iter()
        .zip(test_labels)
        .filter(|&(im, &label)| {
            let predicted = model.predict(im);
            predicted == label
        })
        .count();

    let total = test_images.len();
    let acc = if total > 0 {
        100.0 * correct as f64 / total as f64
    } else {
        0.0
    };
    (correct, total, acc)
}

struct Trainer<'a, const N: usize> {
    accumulators: [BinaryAccumulator<N>; NUM_CLASSES],
    model: Model<N>,
    train_hvs: Vec<BinaryHDV<N>>,
    train_labels: &'a [u8],
    indices: Vec<usize>,
}

impl<'a, const N: usize> Trainer<'a, N> {
    pub fn new(data: &'a Mnist, rng: &mut impl Rng, learned: bool) -> Self {

        let encoder = if learned {
            MnistEncoder::<N>::new(rng)
            .with_feature_learned()
            .train_on(&data.train_images)
        } else {
            MnistEncoder::<N>::new(rng)
            .with_feature_pixel_bag()
            .with_feature_edges()
        };

        println!("Encoding training images (Dim {N}x64={})...", N * 64);
        let train_hvs: Vec<BinaryHDV<N>> = data
            .train_images
            .par_iter()
            .map(|im| encoder.encode(im))
            .collect();

        let mut accumulators: [BinaryAccumulator<N>; NUM_CLASSES] =
            core::array::from_fn(|_| BinaryAccumulator::<N>::new());

        let indices = (0..train_hvs.len()).collect();

        // initial bundling
        for &idx in &indices {
            let hdv = &train_hvs[idx];
            let digit = data.train_labels[idx] as usize;
            accumulators[digit].add(hdv, 1.0);
        }

        let hdvs: [BinaryHDV<N>; NUM_CLASSES] =
            core::array::from_fn(|i| accumulators[i].finalize());
        let model = Model { hdvs, encoder };

        Trainer {
            train_hvs,
            train_labels: &data.train_labels,
            accumulators,
            model,
            indices,
        }
    }

    // one training epoch
    fn step(&mut self, epoch: usize) -> (usize, usize) {
        self.indices.shuffle(&mut rng());

        let mut errors = 0;
        let lr = 1.0 / (epoch as f64).sqrt();

        for &idx in self.indices.iter() {
            let img_hdv = &self.train_hvs[idx];
            let true_label = self.train_labels[idx];
            let predicted = self.model.predict_hdv(img_hdv);
            if predicted != true_label {
                errors += 1;
                self.accumulators[true_label as usize].add(img_hdv, lr);
                self.accumulators[predicted as usize].add(img_hdv, -lr);
            }
        }
        self.model.hdvs = core::array::from_fn(|i| self.accumulators[i].finalize());

        let total = self.train_hvs.len();
        let correct = total - errors;
        (correct, errors)
    }
}

fn main() -> Result<(), MnistError> {
    const N: usize = 100;
    let data = Mnist::load("MNIST")?;
    println!("Read {} training labels", data.train_labels.len());
    let ensemble_size = 11;
    let mut ensemble = EnsembleModel::<N> {
        models: Vec::with_capacity(ensemble_size),
    };
    let seed = 42;
    let mut rng = StdRng::seed_from_u64(seed);
    for mn in 1..=ensemble_size {
        let learned = mn % 2 == 0;
        let mut trainer = Trainer::new(&data, &mut rng, learned);
        let n_epochs = 2000;

        println!("Training model {mn}");
        for epoch in 1..=n_epochs {
            let (correct, errors) = trainer.step(epoch);
            let total = correct + errors;
            let acc = 100.0 * correct as f64 / total as f64;
            print!(
                "Epoch: {epoch:3}/{n_epochs} Training Accuracy: {correct:5}/{total} = {acc:.2}% \r"
            );
            std::io::stdout().flush()?;
            if errors == 0 {
                break;
            }
        }
        let (correct, total, acc) =
            calc_accuracy(&data.test_images, &data.test_labels, &trainer.model);
        println!("\nTest Accuracy: {correct:5}/{total} = {acc:.2}%\n");
        ensemble.models.push(trainer.model);

        let n = ensemble.models.len();
        if n > 2 {
            let (correct, total, acc) =
                calc_accuracy(&data.test_images, &data.test_labels, &ensemble);
            println!("Ensemble of {n} Test Accuracy: {correct:5}/{total} = {acc:.2}%\n");
        }
    }
    Ok(())
}
