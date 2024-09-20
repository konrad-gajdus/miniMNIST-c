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
Epoch 1, Accuracy: 96.12%, Avg Loss: 0.2188
Epoch 2, Accuracy: 96.98%, Avg Loss: 0.0875
Epoch 3, Accuracy: 97.41%, Avg Loss: 0.0561
Epoch 4, Accuracy: 97.63%, Avg Loss: 0.0383
Epoch 5, Accuracy: 97.63%, Avg Loss: 0.0270
Epoch 6, Accuracy: 97.69%, Avg Loss: 0.0193
Epoch 7, Accuracy: 97.98%, Avg Loss: 0.0143
Epoch 8, Accuracy: 98.03%, Avg Loss: 0.0117
Epoch 9, Accuracy: 98.03%, Avg Loss: 0.0103
Epoch 10, Accuracy: 98.06%, Avg Loss: 0.0094
Epoch 11, Accuracy: 98.06%, Avg Loss: 0.0087
Epoch 12, Accuracy: 98.16%, Avg Loss: 0.0081
Epoch 13, Accuracy: 98.16%, Avg Loss: 0.0078
Epoch 14, Accuracy: 98.18%, Avg Loss: 0.0075
Epoch 15, Accuracy: 98.19%, Avg Loss: 0.0074
Epoch 16, Accuracy: 98.20%, Avg Loss: 0.0072
Epoch 17, Accuracy: 98.24%, Avg Loss: 0.0070
Epoch 18, Accuracy: 98.23%, Avg Loss: 0.0069
Epoch 19, Accuracy: 98.23%, Avg Loss: 0.0069
Epoch 20, Accuracy: 98.22%, Avg Loss: 0.0068
```

## Prerequisites

- GCC compiler
- MNIST dataset files:
  - `train-images.idx3-ubyte`
  - `train-labels.idx1-ubyte`

## Compilation

```bash
gcc -o nn nn.c -lm
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
