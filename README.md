# Engram

Exploring MNIST digit classification with high-dimensional binary vectors (hypervectors). 
All approaches share a common feature encoding that maps 28×28 images to N×64 bit vectors.

**Best Result**: 98.35% test accuracy (Modern Hopfield ensemble)

## Models

| Approach | Single Model | Ensemble (5) | Training |
|----------|-------------|--------------|----------|
| Perceptron | 95-97% | **97.89%** | Iterative error correction |
| Modern Hopfield | 97-98% | **98.35%** | Greedy prototype selection |
| K-Means (VQ) | 95.19% | — | Centroid clustering |
| Classic Hopfield | 88.02% | — | Associative memory |

**Key insight**: Feature encoding quality dominates model choice. A simple perceptron with rich features outperforms complex models with poor features.

## Getting Started

Clone the repository:

```sh
git clone https://github.com/jesper-olsen/engram.git
cd engram
```

[Download](https://github.com/jesper-olsen/mnist-rs) the MNIST dataset.

Run the models:

* Perceptron: `cargo run --bin hyper --release`
* Modern Hopfield: `cargo run --bin mhn --release`
* Classic Hopfield: `cargo run --bin hop --release`
* K-Means: `cargo run --bin cb --release`

## Feature Encoding

All models use the same hypervector encoding:

* **Pixel Bag**: Positional hypervectors bound (XORed) with intensity vectors
* **Edge Features**: Sobel edge detection (horizontal, vertical, both diagonals)

### Impact of Features (N=100, 6400 bits)

| Features | Train Acc | Test Acc |
|----------|-----------|----------|
| Pixel Bag only | 97.65% | 92.60% |
| Single edge direction | 85-91% | 79-88% |
| **All features combined** | **98.57%** | **95.68%** |

### Impact of Dimension (All Features)

| N | Bits | Train Acc | Test Acc |
|--:|-----:|----------:|---------:|
| 100 | 6,400 | 98.57% | 95.68% |
| 200 | 12,800 | 99.93% | 96.67% |
| 400 | 25,600 | 99.98% | 96.87% |
| 800 | 51,200 | 99.98% | 96.94% |
| 1600 | 102,400 | 99.98% | 97.03% |

*Diminishing returns beyond N=200; ensemble diversity is more effective than larger dimensions.*

## Perceptron

Iterative perceptron-style training refines 10 prototype hypervectors (one per digit).

```sh
cargo run --bin hyper --release
```

### Ensemble Results (N=100)

| Model | Accuracy |
|------:|---------:|
| 1 | 97.12% |
| 2 | 96.99% |
| 3 | 96.98% |
| 4 | 97.16% |
| 5 | 97.15% |
| **Ensemble** | **97.89%** |

## Modern Hopfield

Energy-based associative memory with softmax attention over stored prototypes. Greedy prototype selection adds misclassified examples during training.

**Hyperparameters**: N=100 (6400 bits), β=60, up to 500 prototypes/class

```sh
cargo run --bin hopfield --release
```

### Ensemble Results (N=100, β=60)

| Model | Accuracy |
|------:|---------:|
| 1 | 97.77% |
| 2 | 97.81% |
| 3 | 97.81% |
| 4 | 98.13% |
| 5 | 97.84% |
| **Ensemble** | **98.35%** |

## Classic Hopfield

Content-addressable memory where the digit's class label is one-hot encoded into the first 10 bits of the hypervector. At test time, these bits are zeroed and the network reconstructs the label through convergence.

```sh
cargo run --bin hop --release
```

**Results (N=100)**:

| Metric | Value |
|--------|------:|
| Correct | 88.02% |
| Correct (unambiguous) | 95.17% |
| Ambiguous | 2.94% |
| No result | 4.57% |

## K-Means (Vector Quantization)

K-Means creates K prototype vectors per digit class. Classification finds the nearest prototype across all classes.

```sh
cargo run --bin cb --release
```

| Centroids/Class | Accuracy |
|----------------:|---------:|
| 1 | 87.54% |
| 5 | 92.67% |
| 10 | 93.94% |
| 15 | 94.94% |
| 20 | **95.19%** |

<details>
<summary>Centroid Distance Matrices</summary>

**Centroid-to-Centroid Distances** (6400 bits):

```
        0      1      2      3      4      5      6      7      8      9
------------------------------------------------------------------------
0:      0   2272   1489   1491   1820   1376   1494   1931   1540   1816
1:   2272      0   1879   1811   2070   1928   2102   2057   1594   2092
2:   1489   1879      0   1260   1583   1561   1585   1750   1303   1693
3:   1491   1811   1260      0   1647   1057   1735   1730   1141   1585
4:   1820   2070   1583   1647      0   1482   1634   1563   1402   1030
5:   1376   1928   1561   1057   1482      0   1520   1737   1092   1428
6:   1494   2102   1585   1735   1634   1520      0   2071   1644   1748
7:   1931   2057   1750   1730   1563   1737   2071      0   1611   1163
8:   1540   1594   1303   1141   1402   1092   1644   1611      0   1344
9:   1816   2092   1693   1585   1030   1428   1748   1163   1344      0
```

**Average Test Distance to Centroids**:

```
       Centroid ->
True    0       1       2       3       4       5       6       7       8       9
----------------------------------------------------------------------------------
 0 | 1808    2597    2192    2189    2335    2138    2182    2406    2214    2327
 1 | 2649    1889    2408    2394    2491    2447    2531    2486    2284    2492
 2 | 2302    2444    1970    2197    2340    2324    2330    2415    2206    2366
 3 | 2235    2399    2149    1882    2321    2078    2327    2350    2109    2278
 4 | 2430    2523    2334    2354    1973    2292    2349    2317    2259    2122
 5 | 2248    2482    2323    2141    2303    1975    2315    2382    2143    2265
 6 | 2217    2537    2251    2328    2281    2241    1891    2509    2285    2342
 7 | 2448    2512    2355    2338    2282    2355    2515    1909    2285    2110
 8 | 2231    2290    2142    2070    2181    2059    2263    2250    1864    2140
 9 | 2352    2476    2289    2240    2006    2174    2325    2080    2136    1838
```

</details>

## References

* [Hyperdimensional Computing](https://en.wikipedia.org/wiki/Hyperdimensional_computing)
* [Modern Hopfield Networks](https://arxiv.org/abs/2008.02217)
