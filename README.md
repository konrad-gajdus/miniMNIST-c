```
  █████     ███                                                    ███           █████
 ░░███     ░░░                                                    ░░░           ░░███
 ███████   ████  ████████   █████ ████ █████████████   ████████   ████   █████  ███████
░░░███░   ░░███ ░░███░░███ ░░███ ░███ ░░███░░███░░███ ░░███░░███ ░░███  ███░░  ░░░███░
  ░███     ░███  ░███ ░███  ░███ ░███  ░███ ░███ ░███  ░███ ░███  ░███ ░░█████   ░███
  ░███ ███ ░███  ░███ ░███  ░███ ░███  ░███ ░███ ░███  ░███ ░███  ░███  ░░░░███  ░███ ███
  ░░█████  █████ ████ █████ ░░███████  █████░███ █████ ████ █████ █████ ██████   ░░█████
   ░░░░░  ░░░░░ ░░░░ ░░░░░   ░░░░░███ ░░░░░ ░░░ ░░░░░ ░░░░ ░░░░░ ░░░░░ ░░░░░░     ░░░░░
                             ███ ░███
                            ░░██████
                             ░░░░░░
```

**tinymnist** is a minimal neural network in pure C that learns to classify handwritten MNIST digits. ~250 lines, no dependencies beyond the C standard library.

**784 → 256 (ReLU) → 10 (softmax)** · cross-entropy loss · SGD with momentum

## Quick start

Put `train-images.idx3-ubyte` and `train-labels.idx1-ubyte` in `data/`, then:

```bash
gcc -O3 -march=native -ffast-math -o main main.c -lm
./main
```

## Results

~98% test accuracy in under 10 seconds of training:

```
Epoch 1, Accuracy: 94.99%, Avg Loss: 0.2969, Time: 0.46 seconds
Epoch 5, Accuracy: 97.53%, Avg Loss: 0.0522, Time: 0.48 seconds
Epoch 10, Accuracy: 97.89%, Avg Loss: 0.0164, Time: 0.47 seconds
Epoch 20, Accuracy: 98.08%, Avg Loss: 0.0028, Time: 0.51 seconds
```

80% of the training set is used for training, the remaining 20% for testing.

## Configuration

Tweak the `#define`s at the top of `main.c`: `HIDDEN_SIZE`, `LEARNING_RATE`, `MOMENTUM`, `EPOCHS`, `TRAIN_SPLIT`.

## License

[MIT](LICENSE)
