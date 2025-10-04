## Engram

Modelling MNIST with hypervectors and Hopfield networks.

Bag-of-pixels hypervector (6400 dimensions) encoded images:

```text
% cargo run --bin hyper --release
Read 60000 labels
Epoch:   1
Epoch:   2 Training Accuracy: 48329/60000 = 80.55%
Epoch:   3 Training Accuracy: 50517/60000 = 84.19%
Epoch:   4 Training Accuracy: 51743/60000 = 86.24%
Epoch:   5 Training Accuracy: 52277/60000 = 87.13%
Epoch:   6 Training Accuracy: 52505/60000 = 87.51%
Epoch:   7 Training Accuracy: 52671/60000 = 87.78%
Epoch:   8 Training Accuracy: 52794/60000 = 87.99%
Epoch:   9 Training Accuracy: 52930/60000 = 88.22%
Epoch:  10 Training Accuracy: 53005/60000 = 88.34%
Test Accuracy 8862/10000 = 88.62%
```

Bag-of-pixels hypervector stored in a Hopfield network 

```text
time cargo run --bin hop --release
correct 8800/10000
13172.99s user 164.83s system 94% cpu 3:56:10.80 total
```
