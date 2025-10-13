# Engram

An exploration of MNIST classification using high-dimensional binary vectors ("hypervectors") with three different modeling approaches:

1. Perceptron Training: An iterative learning model that achieves 97.03% test accuracy.
2. K-Means Clustering: A Vector Quantization approach that achieves ~94% test accuracy.
3. Hopfield Networks: A classic associative memory model that achieves ~88% test accuracy (95% on unambiguous results).

The Perceptron and K-Means methods are fast and storage-efficient, while the Hopfield network is significantly slower. The primary finding is that the performance of these models is highly dependent on the richness of the feature encoding, with a combination of pixel and edge-based features proving most effective.


## Getting Started

Clone the repository:

```sh
git clone https://github.com/your-username/engram.git
cd engram
```

[Download](https://github.com/jesper-olsen/mnist-rs) the MNIST dataset.

Run the models:
Perceptron: `cargo run --bin hyper --release`
Hopfield: `cargo run --bin hop --release`
K-Means: `cargo run --bin cb --release`


## Perceptron trained hypervectors

This model uses an iterative perceptron-style training rule to refine 10 prototype hypervectors (one for each digit).

```sh
% cargo run --bin hyper --release

Read 60000 training labels
Encoding training images...
Encoding test images...
Epoch:   1 (Bundling)
Epoch: 5000/5000 Training Accuracy: 59144/60000 = 98.57% Test: 95.68%
Test Accuracy 9566/10000 = 95.66%
```

### Feature Encoding

The final hypervector is a bundle of the following features. 

* Pixel Bag: Positional hypervectors bound (XORed) with intensity vectors.
* Edge Features: Edge detection using 3x3 Sobel kernels for horizontal, vertical, and both diagonal directions.

The dimension of the vector is N * 64 bits.

|  N   |  Pixel_Bag | Horizontal | Vertical | Diagonal1 | Diagonal2    | Acc Train (%) | Acc Test (%)  | Epochs    | 
|-----:|:----------:|:----------:|:--------:|:---------:|--------------:|-------------:|---------------|----------:|
|  100 |    +       |  -         |  -       | -         | -             | 97.65        | 92.60         | 5000      | 
|  100 |    -       |  +         |  -       | -         | -             | 91.86        | 85.81         | 5000      | 
|  100 |    -       |  -         |  +       | -         | -             | 87.08        | 85.72         | 5000      | 
|  100 |    -       |  -         |  -       | +         | -             | 85.55        | 79.74         | 5000      | 
|  100 |    -       |  -         |  -       | -         | +             | 89.64        | 87.75         | 5000      | 
|  100 |    +       |  +         |  +       | +         | +             | 98.57        | 95.68         | 5000      |
|  200 |    +       |  +         |  +       | +         | +             | 99.93        | 96.67         | 5000      |
|  400 |    +       |  +         |  +       | +         | +             | 99.98        | 96.87         | 5000      |
|  800 |    +       |  +         |  +       | +         | +             | 99.98        | 96.94         | 5000      |
| 1600 |    +       |  +         |  +       | +         | +             | 99.98        | 97.03         | 5000      |


## Hopfield 
This model uses a Hopfield network as a content-addressable memory. During training, the digit's class label is one-hot encoded and stored directly into the first 10 bits of the image's hypervector. At test time, these 10 bits are zeroed out and the network attempts to reconstruct the correct label through convergence.

```sh
cargo run --bin hop --release
```

For N = 100 (6400 bits)

```text
ambiguous 294/10000 = 2.94%
no result 457/10000 = 4.57%
correct/total 8802/10000 = 88.02%
correct/unambiguous 8802/9249 = 95.17%
errors/unambiguous 447/9249 = 4.83%
```

## KMeans Clustering (Vector Quantisation)

This method uses K-Means to create a "codebook" of K prototype vectors for each digit. Classification is done by finding the single closest prototype to a test image's hypervector across all 10 codebooks and choosing its class.

```text
cargo run --bin cb --release
```

Classification accuracy (%) - Hypervector dimension (N) vs codebook size (K)
```
   K      1     2     4     8    16    20
N
 100: 86.05 88.13 90.44 91.76 93.28 93.52
 200: 86.61 88.70 90.97 92.23 93.12 93.62
 400: 86.71 88.89 90.87 92.56 93.20 93.49
 800: 86.86 88.77 91.24 92.82 93.38 93.84
1600: 86.84 89.07 91.05 92.57 93.13 93.62
```
