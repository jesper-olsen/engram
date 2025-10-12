## Engram

Modelling MNIST with hypervectors, Hopfield networks and KMeans clustering.

## Perceptron trained hypervectors

Bag-of-pixels hypervector (6400 dimensions) encoded images:

```text
Read 60000 training labels
Encoding training images...
Encoding test images...
Epoch:   1 (Bundling)
Epoch:   2 Training Accuracy: 52654/60000 = 87.76%
Epoch:   3 Training Accuracy: 54205/60000 = 90.34%
Epoch:   4 Training Accuracy: 54687/60000 = 91.14%
Epoch:   5 Training Accuracy: 55058/60000 = 91.76%
Epoch:   6 Training Accuracy: 55366/60000 = 92.28%
Epoch:   7 Training Accuracy: 55556/60000 = 92.59%
Epoch:   8 Training Accuracy: 55708/60000 = 92.85%
Epoch:   9 Training Accuracy: 55855/60000 = 93.09%
Epoch:  10 Training Accuracy: 55984/60000 = 93.31%
Test Accuracy 9380/10000 = 93.80%
```

N: dimension = N*64
Features (2x2): Pixel_Bag, Horizontal, Vertical, Diagonal 
          
|  N   |  Pixel_Bag | Horizontal | Vertical | Diagonal | Acc Train (%) | Acc Test (%) | Epochs                  | 
|-----:|:----------:|:----------:|:--------:|:--------:|--------------:|-------------:|------------------------:|
|  100 |    +       |  -         |  -       | -        |  93.84        | 88.97        |               5000      | 
|  100 |    -       |  +         |  -       | -        |  88.97        | 86.40        |               5000      |
|  100 |    -       |  -         |  +       | -        |  81.96        | 82.05        |               5000      |
|  100 |    -       |  -         |  -       | +        |  97.57        | 92.14        |               5000      |
|  100 |    +       |  +         |  +       | +        |  98.91        | 94.60        |               5000      |
|  200 |    +       |  +         |  +       | +        |  99.94        | 95.53        |               5000      |
|  400 |    +       |  +         |  +       | +        | 100.00        | 95.90        |               5000      |
|  400 |    +       |  +         |  +       | +        |  99.99        | 94.90        | +polarity ->  5000      | 
|  800 |    +       |  +         |  +       | +        | 100.00        | 95.98        |               5000      |
| 1600 |    +       |  +         |  +       | +        | 100.00        | 96.26        |               5000      |

3x3 features 
|  100 |    +       |  +         |  +       | +        |  98.64        | 95.74        |               5000      |
|  200 |    +       |  +         |  +       | +        |  99.92        | 96.46        |               5000      |
|  400 |    +       |  +         |  +       | +        |  99.99        | 96.50        |               5000      |
|  800 |    +       |  +         |  +       | +        |  99.98        | 96.74        |               5000      |
| 1600 |    +       |  +         |  +       | +        |  99.97        | 96.91        |               5000      |

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
