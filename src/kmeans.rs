use hypervector::Accumulator;
use hypervector::binary_hdv::{BinaryAccumulator, BinaryHDV};
use rand::SeedableRng;
use rand::prelude::IndexedRandom;
use rand::rngs::StdRng;
use rayon::prelude::*;
use std::borrow::Borrow;

pub struct KMeans<const N: usize> {
    pub k: usize,
    pub centroids: Vec<BinaryHDV<N>>,
    pub counts: Vec<usize>,
}

impl<const N: usize> KMeans<N> {
    /// Creates a new KMeans model with initial centroids chosen randomly.
    pub fn new<T: Borrow<BinaryHDV<N>> + Sync>(data: &[T], k: usize) -> Self {
        assert!(k > 0 && k <= data.len(), "k must be > 0 and <= data length");

        // Simple random initialization.
        // TODO: implement K-Means++ initialisation.
        let seed = 42;
        let mut rng = StdRng::seed_from_u64(seed);
        let centroids: Vec<BinaryHDV<N>> = data
            .choose_multiple(&mut rng, k)
            .map(|v| *v.borrow())
            .collect();

        Self {
            k,
            centroids,
            counts: vec![0; k], // Initialize counts to zero
        }
    }

    /// Trains the model until convergence or max_iters is reached.
    pub fn train<T: Borrow<BinaryHDV<N>> + Sync>(
        &mut self,
        data: &[T],
        max_iters: u32,
        verbose: bool,
    ) -> usize {
        let mut last_total_dist = usize::MAX;

        if verbose {
            println!("Cluster {} examples", data.len());
        }
        for i in 1..=max_iters {
            let total_dist = self.step(data);
            if verbose {
                println!("Iteration {i}: total distance = {}", total_dist);
            }

            // Convergence check
            if last_total_dist == total_dist {
                if verbose {
                    println!("Converged after {i} iterations.");
                }
                break;
            }
            last_total_dist = total_dist;
        }
        last_total_dist
    }

    fn step<T: Borrow<BinaryHDV<N>> + Sync>(&mut self, data: &[T]) -> usize {
        let mut accumulators: Vec<BinaryAccumulator<N>> =
            (0..self.k).map(|_| BinaryAccumulator::new()).collect();

        let total_dist: u32 = data
            .iter()
            .map(|v| {
                let (idx, dist) = self.nearest(v.borrow());
                accumulators[idx as usize].add(v.borrow(), 1.0);
                dist
            })
            .sum();
        self.centroids = accumulators.iter().map(|a| a.finalize()).collect();
        self.counts = accumulators.iter().map(|acc| acc.count as usize).collect();

        total_dist as usize 
    }

    //fn step<T: Borrow<BinaryHDV<N>> + Sync>(&mut self, data: &[T]) -> usize {
    //    // Parallel assignment: each thread processes a chunk of the data
    //    let results: Vec<(u32, u32)> = data
    //        .par_iter()
    //        .map(|v| self.nearest(v.borrow()))
    //        .collect();

    //    let mut accumulators: Vec<BinaryAccumulator<N>> =
    //        (0..self.k).map(|_| BinaryAccumulator::new()).collect();
    //
    //    let total_dist: u32 = results
    //        .iter()
    //        .zip(data.iter())
    //        .map(|(&(idx, dist), v)| {
    //            accumulators[idx as usize].add(v.borrow(), 1.0);
    //            dist
    //        })
    //        .sum();

    //    // Update centroids and counts
    //    self.centroids = accumulators.iter().map(|a| a.finalize()).collect();
    //    self.counts = accumulators.iter().map(|acc| acc.count as usize).collect();

    //    total_dist as usize
    //}

    // Finds the index and distance of the nearest centroid to a given vector.
    pub fn nearest(&self, hdv: &BinaryHDV<N>) -> (u32, u32) {
        let mut min_dist = u32::MAX;
        let mut best_cluster = 0;

        for (j, centroid) in self.centroids.iter().enumerate() {
            let dist = hdv.hamming_distance(centroid);
            if dist < min_dist {
                min_dist = dist;
                best_cluster = j;
            }
        }
        (best_cluster as u32, min_dist)
    }

    //pub fn nearest(&self, hdv: &BinaryHDV<N>) -> (u32, u32) {
    //    self.centroids
    //        .iter()
    //        .enumerate()
    //        .map(|(j, centroid)| (j as u32, hdv.hamming_distance(centroid)))
    //        .min_by_key(|&(_, dist)| dist)
    //        .expect("Centroids should not be empty")
    //}
}
