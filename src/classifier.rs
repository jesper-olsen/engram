use hypervector::binary_hdv::BinaryHDV;
use mnist::Image;

pub trait ImageClassifier {
    fn predict(&self, im: &Image) -> u8;
}

pub trait HdvClassifier<const N: usize>: ImageClassifier {
    fn predict_hdv(&self, h: &BinaryHDV<N>) -> u8;
}

/// Shared accuracy calculation
pub fn calc_accuracy<M: ImageClassifier + Sync>(
    test_images: &[Image],
    test_labels: &[u8],
    model: &M,
) -> (usize, usize, f64) {
    use rayon::prelude::*;

    let correct: usize = test_images
        .par_iter()
        .zip(test_labels)
        .filter(|&(im, &label)| model.predict(im) == label)
        .count();

    let total = test_images.len();
    let acc = if total > 0 {
        100.0 * correct as f64 / total as f64
    } else {
        0.0
    };
    (correct, total, acc)
}
