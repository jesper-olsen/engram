## Engram

Modelling MNIST with hypervectors, Hopfield networks and KMeans clustering.

## Perceptron trained hypervectors

```sh
cargo run --bin hyper --release
```

N: dimension = N*64
Features: Pixel_Bag, Horizontal, Vertical, Diagonal (Sobel edge_threshold 250)
          
|  N   |  Pixel_Bag | Horizontal | Vertical | Diagonal1 | Diagonal2    | Acc Train (%) | Acc Test (%)  | Epochs    | 
|-----:|:----------:|:----------:|:--------:|:---------:|--------------:|-------------:|---------------|----------:|
|  100 |    +       |  -         |  -       | -         | -             | 97.65        | 92.60         | 5000      | 
|  100 |    -       |  +         |  -       | -         | -             | 91.86        | 85.81         | 5000      | 
|  100 |    -       |  -         |  +       | -         | -             | 87.08        | 85.72         | 5000      | 
|  100 |    -       |  -         |  -       | +         | -             | 85.55        | 79.74         | 5000      | 
|  100 |    -       |  -         |  -       | -         | +             | 89.64        | 87.75         | 5000      | 
|  100 |    +       |  +         |  +       | +         | +             | 99.70        | 96.24         | 5000      |
|  200 |    +       |  +         |  +       | +         | +             | 99.93        | 96.67         | 5000      |
|  400 |    +       |  +         |  +       | +         | +             | 99.98        | 96.87         | 5000      |
|  800 |    +       |  +         |  +       | +         | +             | 99.98        | 96.94         | 5000      |
| 1600 |    +       |  +         |  +       | +         | +             | 99.98        | 97.03         | 5000      |

## Hopfield 

Bag-of-pixels hypervector stored in a Hopfield network 

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

## KMeans 

```text
cargo run --bin cb --release
```

Classification accuracy (%) - Hypervector dimension vs codebook size
```
   K      1     2     4     8    16    20
N
 100: 86.05 88.13 90.44 91.76 93.28 93.52
 200: 86.61 88.70 90.97 92.23 93.12 93.62
 400: 86.71 88.89 90.87 92.56 93.20 93.49
 800: 86.86 88.77 91.24 92.82 93.38 93.84
1600: 86.84 89.07 91.05 92.57 93.13 93.62
```
