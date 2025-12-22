use crate::classifier::ImageClassifier;
use mnist::Image;

const NUM_CLASSES: usize = 10;

pub struct Ensemble<M> {
    pub models: Vec<M>,
}

impl<M> Ensemble<M> {
    pub fn new() -> Self {
        Ensemble { models: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Ensemble {
            models: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, model: M) {
        self.models.push(model);
    }

    pub fn len(&self) -> usize {
        self.models.len()
    }

    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

impl<M: ImageClassifier> ImageClassifier for Ensemble<M> {
    fn predict(&self, im: &Image) -> u8 {
        let mut votes = [0u32; NUM_CLASSES];

        for model in &self.models {
            let prediction = model.predict(im);
            votes[prediction as usize] += 1;
        }

        votes
            .iter()
            .enumerate()
            .max_by_key(|&(_, &count)| count)
            .map(|(digit, _)| digit as u8)
            .unwrap_or(0)
    }
}

impl<M> Default for Ensemble<M> {
    fn default() -> Self {
        Self::new()
    }
}
