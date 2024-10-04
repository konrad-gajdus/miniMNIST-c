# miniMNIST-c

This project implements a mini neural network in C for classifying handwritten digits from the MNIST dataset.

## Features

- Two-layer neural network (input -> hidden -> output)
- ReLU activation for hidden layer
- Softmax activation for output layer
- Cross-entropy loss function
- Stochastic Gradient Descent (SGD) with momentum
- L2 regularization

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
- `LEARNING_RATE`: Initial learning rate
- `MOMENTUM`: Momentum coefficient for SGD
- `L2_LAMBDA`: L2 regularization coefficient
- `EPOCHS`: Number of training epochs
- `BATCH_SIZE`: Mini-batch size for training

## License

This project is open-source and available under the [MIT License](LICENSE).
