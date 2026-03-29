use engram::{Ensemble, ImageClassifier, MnistEncoder, calc_accuracy};
use hypervector::{
    HyperVector,
    binary_hdv::BinaryHDV,
    hdv,
    trainer::{PerceptronTrainer, PrototypeModel},
};
use mnist::{self, Image, Mnist, error::MnistError};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

pub struct EncodedModel<T: HyperVector, const N: usize> {
    pub model: PrototypeModel<T, N>,
    pub encoder: MnistEncoder<T>,
}

impl<T: HyperVector, const N: usize> ImageClassifier for EncodedModel<T, N> {
    fn predict(&self, im: &Image) -> u8 {
        let h = self.encoder.encode(im);
        self.model.predict(&h) as u8
    }
}

const NUM_CLASSES: usize = 10;

fn main() -> Result<(), MnistError> {
    const TOTAL_BITS: usize = 6400;
    hdv!(binary, HDV, TOTAL_BITS);
    let data = Mnist::load("MNIST")?;
    //let data = Mnist::load("MNISTfashion")?;
    println!("Read {} training labels", data.train_labels.len());
    let ensemble_size = 5;
    let mut ensemble: Ensemble<EncodedModel<HDV, NUM_CLASSES>> =
        Ensemble::with_capacity(ensemble_size);

    let seed = 42;
    for mn in 1..=ensemble_size {
        println!("Training model {mn}");
        let mut rng = StdRng::seed_from_u64(seed + mn as u64);
        let n_epochs = 2000;

        let encoder = MnistEncoder::<HDV>::new(&mut rng)
            .with_feature_pixel_bag()
            .with_feature_edges();
        //.with_feature_learned()
        //.train_on(&data.train_images, &data.train_labels);

        println!("Encoding images (Dim {})...", HDV::DIM);
        let train_hvs: Vec<HDV> = data
            .train_images
            .par_iter()
            .map(|im| encoder.encode(im))
            .collect();

        let mut trainer = PerceptronTrainer::<HDV, u8, _, 10>::new(
            train_hvs,
            data.train_labels.clone(), // or collect into a Vec<u8>
            rng,
        );

        // drive epochs manually
        for epoch in 1..=n_epochs {
            let r = trainer.step(epoch);
            print!(
                "Epoch {epoch}: {}/{}={:.2}%\r",
                r.correct,
                r.total(),
                r.accuracy() * 100.0
            );
            if r.errors == 0 {
                break;
            }
        }
        println!();
        let model = trainer.into_model();
        let encoded_model = EncodedModel { model, encoder };
        //let (model, history) = trainer.fit(n_epochs);

        let (correct, total, acc) =
            calc_accuracy(&data.test_images, &data.test_labels, &encoded_model);
        println!("Test Accuracy: {correct:5}/{total} = {acc:.2}%\n");

        ensemble.models.push(encoded_model);

        let n = ensemble.models.len();
        if n > 2 {
            let (correct, total, acc) =
                calc_accuracy(&data.test_images, &data.test_labels, &ensemble);
            println!("Ensemble of {n} Test Accuracy: {correct:5}/{total} = {acc:.2}%\n");
        }
    }
    Ok(())
}
