use std::io::Write;

use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::prelude::*;

use hypervector::{
    HyperVector,
    hdv,
    trainer::{MultiPrototypeModel, Classifier, pa::PaTrainer, pa::PaVariant, perceptron::PerceptronTrainer, PrototypeModel},
};
use hypervector::types::binary::BinaryHDV;
use hypervector::trainer::multi_perceptron::PerceptronMultiTrainer;
use hypervector::trainer::lvq::LvqTrainer;
use mnist::{self, Image, Mnist, error::MnistError};
use engram::{Ensemble, ImageClassifier, MnistEncoder, calc_accuracy};

pub struct EncodedModel<T: HyperVector, C: Classifier<T>> {
    pub model: C,
    pub encoder: MnistEncoder<T>,
}

impl<T: HyperVector, C: Classifier<T>> ImageClassifier for EncodedModel<T, C> {
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
    let mut ensemble: Ensemble<EncodedModel<HDV, PrototypeModel<HDV, NUM_CLASSES>>> =
    //let mut ensemble: Ensemble<EncodedModel<HDV, MultiPrototypeModel<HDV>>> =
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

        let mut trainer = PerceptronTrainer::<HDV, u8, _, 10>::new(&train_hvs, &data.train_labels, rng);

        //let mut trainer = PerceptronMultiTrainer::<HDV, _>::new(train_hvs, data.train_labels.clone(), NUM_CLASSES, 3, rng);

        //let mut trainer = LvqTrainer::<HDV, _>::new(train_hvs, data.train_labels.clone(), NUM_CLASSES, 3, rng, 0.25);

        //let mut trainer = PaTrainer::<HDV, u8, _, 10>::new(
        //    train_hvs,
        //    data.train_labels.clone(), // or collect into a Vec<u8>
        //    //PaVariant::Pa,
        //    //PaVariant::PaI {c: 0.1 },
        //    PaVariant::PaII { c: 1.0 },
        //    rng,
        //);

        // drive epochs manually
        for epoch in 1..=n_epochs {
            let r = trainer.step(epoch);
            print!(
                "Epoch {epoch}: {}/{}={:.2}%\r",
                r.correct,
                r.total(),
                r.accuracy() * 100.0
            );
            std::io::stdout().flush().unwrap();

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
