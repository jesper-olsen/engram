use hypervector::HyperVector;
use mnist::Image;

pub trait ImageClassifier {
    fn predict(&self, im: &Image) -> u8;
}

pub trait HdvClassifier<T: HyperVector>: ImageClassifier {
    fn predict_hdv(&self, h: &T) -> u8;
}

/// Shared accuracy calculation
pub fn calc_accuracy<M: ImageClassifier + Sync>(
    images: &[Image],
    labels: &[u8],
    model: &M,
) -> (usize, usize, f64) {
    use rayon::prelude::*;

    assert!(images.len() == labels.len());
    assert!(images.len() > 0);
    let correct: usize = images
        .par_iter()
        .zip(labels)
        .filter(|&(im, &label)| model.predict(im) == label)
        .count();

    let total = images.len();
    let acc = 100.0 * correct as f64 / images.len() as f64;
    (correct, total, acc)
}
