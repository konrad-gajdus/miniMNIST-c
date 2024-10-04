# miniMNIST-c

This project implements a **minimal** neural network in C for classifying handwritten digits from the [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download). The entire implementation is  **~200 lines of code** and uses only the standard C library.

## Features

- Two-layer neural network (input → hidden → output)
- ReLU activation function for the hidden layer
- Softmax activation function for the output layer
- Cross-entropy loss function
- Stochastic Gradient Descent (SGD) optimizer

## Performance

```bash
Epoch 1, Accuracy: 95.61%, Avg Loss: 0.2717, Time: 2.61 seconds
Epoch 2, Accuracy: 96.80%, Avg Loss: 0.1167, Time: 2.62 seconds
Epoch 3, Accuracy: 97.21%, Avg Loss: 0.0766, Time: 2.66 seconds
Epoch 4, Accuracy: 97.38%, Avg Loss: 0.0550, Time: 2.64 seconds
Epoch 5, Accuracy: 97.49%, Avg Loss: 0.0397, Time: 2.64 seconds
Epoch 6, Accuracy: 97.47%, Avg Loss: 0.0285, Time: 2.65 seconds
Epoch 7, Accuracy: 97.47%, Avg Loss: 0.0205, Time: 2.66 seconds
Epoch 8, Accuracy: 97.72%, Avg Loss: 0.0151, Time: 2.66 seconds
Epoch 9, Accuracy: 97.88%, Avg Loss: 0.0112, Time: 2.67 seconds
Epoch 10, Accuracy: 97.82%, Avg Loss: 0.0084, Time: 2.67 seconds
Epoch 11, Accuracy: 97.88%, Avg Loss: 0.0063, Time: 2.68 seconds
Epoch 12, Accuracy: 97.92%, Avg Loss: 0.0049, Time: 2.68 seconds
Epoch 13, Accuracy: 97.92%, Avg Loss: 0.0039, Time: 2.69 seconds
Epoch 14, Accuracy: 98.02%, Avg Loss: 0.0032, Time: 2.69 seconds
Epoch 15, Accuracy: 98.06%, Avg Loss: 0.0027, Time: 2.70 seconds
Epoch 16, Accuracy: 98.09%, Avg Loss: 0.0024, Time: 2.70 seconds
Epoch 17, Accuracy: 98.11%, Avg Loss: 0.0021, Time: 2.69 seconds
Epoch 18, Accuracy: 98.12%, Avg Loss: 0.0019, Time: 2.70 seconds
Epoch 19, Accuracy: 98.16%, Avg Loss: 0.0017, Time: 2.70 seconds
Epoch 20, Accuracy: 98.17%, Avg Loss: 0.0015, Time: 2.71 seconds
```

## Prerequisites

- GCC compiler
- MNIST dataset files:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`

## Compilation

```bash
gcc -O3 -march=native -ffast-math -o nn nn.c -lm
```

## Usage

1. Place the MNIST dataset files in the `data/` directory.
2. Compile the program.
3. Run the executable:

   ```bash
   ./nn
   ```

The program will train the neural network on the MNIST dataset and output the accuracy and average loss for each epoch.

## Configuration

You can adjust the following parameters in `nn.c`:

- `HIDDEN_SIZE`: Number of neurons in the hidden layer
- `LEARNING_RATE`: Learning rate for SGD
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Mini-batch size for training
- `TRAIN_SPLIT`: Proportion of data used for training (the rest is used for testing)

## License

This project is open-source and available under the [MIT License](LICENSE).
