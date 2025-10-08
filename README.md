## Engram

Modelling MNIST with hypervectors and Hopfield networks.

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


Bag-of-pixels hypervector stored in a Hopfield network 

```text
ambiguous 275/10000 = 2.75%
no result 512/10000 = 5.12%
correct/total 8662/10000 = 86.62%
correct/unambiguous 8662/9213 = 94.02%
errors/unambiguous 551/9213 = 5.98%
```
