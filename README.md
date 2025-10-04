## Engram

Modelling MNIST with hypervectors and Hopfield networks.

Bag-of-pixels hypervector (6400 dimensions) encoded images:

```text
Read 60000 training labels
Encoding training images...
Encoding test images...
Epoch:   1 (Bundling)
Epoch:   2 Training Accuracy: 48357/60000 = 80.59%
Epoch:   3 Training Accuracy: 50740/60000 = 84.57%
Epoch:   4 Training Accuracy: 51413/60000 = 85.69%
Epoch:   5 Training Accuracy: 51961/60000 = 86.60%
Epoch:   6 Training Accuracy: 52183/60000 = 86.97%
Epoch:   7 Training Accuracy: 52485/60000 = 87.47%
Epoch:   8 Training Accuracy: 52708/60000 = 87.85%
Epoch:   9 Training Accuracy: 52885/60000 = 88.14%
Epoch:  10 Training Accuracy: 53010/60000 = 88.35%
Test Accuracy 8842/10000 = 88.42%
14.35s user 0.23s system 84% cpu 17.198 total
```

Bag-of-pixels hypervector stored in a Hopfield network 

```text
time cargo run --bin hop --release
correct 8800/10000
13172.99s user 164.83s system 94% cpu 3:56:10.80 total
```
