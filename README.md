# Engram

An exploration of MNIST classification using high-dimensional binary vectors ("hypervectors") with three different modeling approaches:

1. Perceptron Training: An iterative learning model that achieves 97.37% test accuracy.
2. K-Means Clustering: A Vector Quantization approach that achieves ~95% test accuracy.
3. Hopfield Networks: A classic associative memory model that achieves ~88% test accuracy (95% on unambiguous results).

The Perceptron and K-Means methods are fast and storage-efficient, while the Hopfield network is significantly slower. The primary finding is that the performance of these models is highly dependent on the richness of the feature encoding, with a combination of pixel and edge-based features proving most effective.


## Getting Started

Clone the repository:

```sh
git clone https://github.com/jesper-olsen/engram.git
cd engram
```

[Download](https://github.com/jesper-olsen/mnist-rs) the MNIST dataset.

Run the models:

* Perceptron: `cargo run --bin hyper --release`
* Hopfield: `cargo run --bin hop --release`
* K-Means: `cargo run --bin cb --release`


## Perceptron Trained Hypervectors

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

Train ensemble of 5 models (N=100) and combine by voting:

| Model | Accuracy (%) |
|------:|-------------:|
|   1   |  96.37       |
|   2   |  96.18       |  
|   3   |  96.16       |
|   4   |  95.91       |
|   5   |  95.36       |
|  1..5 |  97.37       |

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

```
Read 60000 training labels
Encoding training images...
Encoding test images...

--- Centroid-to-Centroid Distances ---
(Total bits = 6400)
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

--- Average Test Data Distance to Centroids ---
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

 Classify - codebook size
 1: Accuracy 8754/10000 = 87.54%
 2: Accuracy 9032/10000 = 90.32%
 3: Accuracy 9123/10000 = 91.23%
 4: Accuracy 9210/10000 = 92.10%
 5: Accuracy 9267/10000 = 92.67%
 6: Accuracy 9296/10000 = 92.96%
 7: Accuracy 9326/10000 = 93.26%
 8: Accuracy 9376/10000 = 93.76%
 9: Accuracy 9417/10000 = 94.17%
10: Accuracy 9394/10000 = 93.94%
11: Accuracy 9410/10000 = 94.10%
12: Accuracy 9424/10000 = 94.24%
13: Accuracy 9451/10000 = 94.51%
14: Accuracy 9446/10000 = 94.46%
15: Accuracy 9494/10000 = 94.94%
16: Accuracy 9471/10000 = 94.71%
17: Accuracy 9468/10000 = 94.68%
18: Accuracy 9503/10000 = 95.03%
19: Accuracy 9497/10000 = 94.97%
20: Accuracy 9519/10000 = 95.19%
```

