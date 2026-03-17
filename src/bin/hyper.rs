use engram::{Ensemble, HdvClassifier, ImageClassifier, MnistEncoder, calc_accuracy};
use hypervector::{Accumulator, HyperVector, binary_hdv::BinaryHDV, modular_hdv::ModularHDV, hdv, nearest};
use mnist::{self, Image, Mnist, error::MnistError};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use std::io::Write;

const NUM_CLASSES: usize = 10;

struct Model<T: HyperVector> {
    hdvs: [T; NUM_CLASSES],
    encoder: MnistEncoder<T>,
}

impl<T: HyperVector> ImageClassifier for Model<T> {
    fn predict(&self, im: &Image) -> u8 {
        let h = self.encoder.encode(im);
        self.predict_hdv(&h)
    }
}

impl<T: HyperVector> HdvClassifier<T> for Model<T> {
    fn predict_hdv(&self, h: &T) -> u8 {
        let (idx, _) = nearest(h, &self.hdvs);
        idx.try_into().unwrap()
    }
}

struct Trainer<'a, T: HyperVector + std::marker::Send + std::marker::Sync, R: Rng> {
    accumulators: [T::Accumulator; NUM_CLASSES],
    model: Model<T>,
    train_hvs: Vec<T>,
    train_labels: &'a [u8],
    indices: Vec<usize>,
    rng: R,
}

impl<'a, T: HyperVector + std::marker::Send + std::marker::Sync, R: Rng> Trainer<'a, T, R> {
    pub fn new(data: &'a Mnist, mut rng: R) -> Self {
        let encoder = MnistEncoder::<T>::new(&mut rng)
            .with_feature_pixel_bag()
            .with_feature_edges();
        //.with_feature_learned()
        //.train_on(&data.train_images, &data.train_labels);

        println!("Encoding training images (Dim {})...", T::DIM);
        let train_hvs: Vec<T> = data
            .train_images
            .par_iter()
            .map(|im| encoder.encode(im))
            .collect();

        let mut accumulators: [T::Accumulator; NUM_CLASSES] =
            core::array::from_fn(|_| T::Accumulator::new());

        let indices = (0..train_hvs.len()).collect();

        // initial bundling
        for &idx in &indices {
            let hdv = &train_hvs[idx];
            let digit = data.train_labels[idx] as usize;
            accumulators[digit].add(hdv, 1.0);
        }

        let hdvs: [T; NUM_CLASSES] = core::array::from_fn(|i| accumulators[i].finalize());
        let model = Model { hdvs, encoder };

        Trainer {
            train_hvs,
            train_labels: &data.train_labels,
            accumulators,
            model,
            indices,
            rng,
        }
    }

    // one training epoch
    //fn step(&mut self, epoch: usize) -> (usize, usize) {
    //    self.indices.shuffle(&mut self.rng);

    //    let mut errors = 0;
    //    let lr = 1.0 / (epoch as f64).sqrt();

    //    for &idx in self.indices.iter() {
    //        let img_hdv = &self.train_hvs[idx];
    //        let true_label = self.train_labels[idx];
    //        let predicted = self.model.predict_hdv(img_hdv);
    //        if predicted != true_label {
    //            errors += 1;
    //            self.accumulators[true_label as usize].add(img_hdv, lr);
    //            self.accumulators[predicted as usize].add(img_hdv, -lr);
    //        }
    //    }
    //    self.model.hdvs = core::array::from_fn(|i| self.accumulators[i].finalize());

    //    let total = self.train_hvs.len();
    //    let correct = total - errors;
    //    (correct, errors)
    //}

    // one training epoch - parallel
    fn step(&mut self, epoch: usize) -> (usize, usize) {
        self.indices.shuffle(&mut self.rng);
        let lr = 1.0 / (epoch as f64).sqrt();

        // Parallel: collect all misclassifications
        let errors: Vec<_> = self
            .indices
            .par_iter()
            .filter_map(|&idx| {
                let img_hdv = &self.train_hvs[idx];
                let true_label = self.train_labels[idx];
                let predicted = self.model.predict_hdv(img_hdv);
                if predicted != true_label {
                    Some((idx, true_label, predicted))
                } else {
                    None
                }
            })
            .collect();

        let error_count = errors.len();

        // Sequential: apply weight updates (accumulators aren't thread-safe)
        for (idx, true_label, predicted) in errors {
            let img_hdv = &self.train_hvs[idx];
            self.accumulators[true_label as usize].add(img_hdv, lr);
            self.accumulators[predicted as usize].add(img_hdv, -lr);
        }

        self.model.hdvs = core::array::from_fn(|i| self.accumulators[i].finalize());

        let total = self.train_hvs.len();
        (total - error_count, error_count)
    }
}

fn main() -> Result<(), MnistError> {
    const TOTAL_BITS: usize = 6400;
    hdv!(binary, HDV, TOTAL_BITS);
    let data = Mnist::load("MNIST")?;
    //let data = Mnist::load("MNISTfashion")?;
    println!("Read {} training labels", data.train_labels.len());
    let ensemble_size = 5;
    let mut ensemble: Ensemble<Model<HDV>> = Ensemble::with_capacity(ensemble_size);
    let seed = 42;
    for mn in 1..=ensemble_size {
        let rng = StdRng::seed_from_u64(seed + mn as u64);
        let mut trainer = Trainer::new(&data, rng);
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
