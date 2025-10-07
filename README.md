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

+ vertical & horizontal
Read 60000 training labels
Encoding training images...
Encoding test images...
Epoch:   1 (Bundling)
Epoch:   2 Training Accuracy: 51311/60000 = 85.52%
Epoch:   3 Training Accuracy: 53321/60000 = 88.87%
Epoch:   4 Training Accuracy: 53956/60000 = 89.93%
Epoch:   5 Training Accuracy: 54331/60000 = 90.55%
Epoch:   6 Training Accuracy: 54633/60000 = 91.06%
Epoch:   7 Training Accuracy: 54843/60000 = 91.41%
Epoch:   8 Training Accuracy: 55015/60000 = 91.69%
Epoch:   9 Training Accuracy: 55152/60000 = 91.92%
Epoch:  10 Training Accuracy: 55307/60000 = 92.18%
Test Accuracy 9281/10000 = 92.81%
```


Bag-of-pixels hypervector stored in a Hopfield network 

```text
ambiguous 289/10000 = 2.89%
no result 502/10000 = 5.02%
correct/total 8644/10000 = 86.44%
correct/unambiguous 8644/9209 = 93.86%
errors/unambiguous 565/9209 = 6.14%
```
