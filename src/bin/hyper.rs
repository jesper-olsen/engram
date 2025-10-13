use engram::{ItemMemory, encode_image};
use hypervector::{
    Accumulator,
    binary_hdv::{BinaryAccumulator, BinaryHDV},
};
use mnist::{self, Mnist, error::MnistError};
use rand::{rng, seq::SliceRandom};
use rayon::prelude::*;

const NUM_CLASSES: usize = 10;

fn calc_test_accuracy<const N: usize>(
    test_hvs: &[BinaryHDV<N>],
    test_labels: &[u8],
    models: &[BinaryHDV<N>; NUM_CLASSES],
) -> (usize, usize, f64) {
    let correct: usize = test_hvs
        .iter()
        .zip(test_labels)
        .map(|(hdv, &label)| {
            let predicted = engram::predict(hdv, models);
            (predicted == label) as usize
        })
        .sum();

    let total = test_hvs.len();
    let acc = if total > 0 {
        100.0 * correct as f64 / total as f64
    } else {
        0.0
    };
    (correct, total, acc)
}

struct Trainer<'a, const N: usize> {
    accumulators: [BinaryAccumulator<N>; NUM_CLASSES],
    models: [BinaryHDV<N>; NUM_CLASSES],
    train_hvs: &'a [BinaryHDV<N>],
    train_labels: &'a [u8],
    indices: Vec<usize>,
}

impl<'a, const N: usize> Trainer<'a, N> {
    pub fn new(train_hvs: &'a [BinaryHDV<N>], train_labels: &'a [u8]) -> Self {
        let mut accumulators: [BinaryAccumulator<N>; NUM_CLASSES] =
            core::array::from_fn(|_| BinaryAccumulator::<N>::new());
        let indices = (0..train_hvs.len()).collect();

        // initial bundling
        for (i, hdv) in train_hvs.iter().enumerate() {
            let digit = train_labels[i] as usize;
            accumulators[digit].add(hdv, 1.0);
        }

        let models: [BinaryHDV<N>; NUM_CLASSES] =
            core::array::from_fn(|i| accumulators[i].finalize());

        Trainer {
            train_hvs,
            train_labels,
            accumulators,
            models,
            indices,
        }
    }

    // one training epoch
    fn step(&mut self, epoch: usize) -> (usize, usize) {
        self.indices.shuffle(&mut rng());

        let mut errors = 0;
        let lr = 1.0 / (epoch as f64).sqrt();

        for &i in &self.indices {
            let img_hdv = &self.train_hvs[i];
            let true_label = self.train_labels[i];
            let predicted = engram::predict(img_hdv, &self.models);
            if predicted != true_label {
                errors += 1;
                self.accumulators[true_label as usize].add(img_hdv, lr);
                self.accumulators[predicted as usize].add(img_hdv, -lr);
            }
        }
        self.models = core::array::from_fn(|i| self.accumulators[i].finalize());

        let total = self.train_hvs.len();
        let correct = total - errors;
        (correct, errors)
    }
}

fn main() -> Result<(), MnistError> {
    const N: usize = 100;
    let imem = ItemMemory::<N>::new();
    let data = Mnist::load("MNIST")?;
    println!("Read {} training labels", data.train_labels.len());

    println!("Encoding training images...");
    let train_hvs: Vec<BinaryHDV<N>> = data
        .train_images
        .par_iter()
        .map(|im| encode_image(im.as_u8_array(), &imem))
        .collect();

    println!("Encoding test images...");
    let test_hvs: Vec<BinaryHDV<N>> = data
        .test_images
        .par_iter()
        .map(|im| encode_image(im.as_u8_array(), &imem))
        .collect();

    let mut trainer = Trainer::new(&train_hvs, &data.train_labels);
    let n_epochs = 5000;
    for epoch in 1..=n_epochs {
        let (correct, errors) = trainer.step(epoch);
        let total = correct + errors;
        let acc = 100.0 * correct as f64 / total as f64;

        let (_test_correct, _test_total, test_acc) =
            calc_test_accuracy(&test_hvs, &data.test_labels, &trainer.models);
        print!(
            "Epoch: {epoch:3}/{n_epochs} Training Accuracy: {correct:5}/{total} = {acc:.2}% Test: {test_acc:.2}%\r"
        );
        if errors == 0 {
            break;
        }
    }
    println!();

    Ok(())
}
