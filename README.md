## Engram

Modelling MNIST with hypervectors and Hopfield networks.

Bag-of-pixels hypervector (6400 dimensions) encoded images:

```
% time cargo run --bin hyper --release
Read 60000 labels
Accuracy 8152/10000 = 81.52
13.99s user 0.19s system 85% cpu 16.498 total
```

Bag-of-pixels hypervector stored in a Hopfield network 

```
time cargo run --bin hop --release
correct 8800/10000
13172.99s user 164.83s system 94% cpu 3:56:10.80 total
```
