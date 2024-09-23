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

## Contributions From Mark (mrmcsoftware)

Every C source file in the **`contrib`** directory are modified versions of **`nn.c`** provided by Mark Craig (mrmcsoftware).

>**Note:** Either run these from the directory where the data directory resides or copy data directory to the contrib directory (or symbolic link to it).

- **`nn_alt.c`** is a version that should compile on any C compiler (new or old, Linux or Windows or Mac?).  If you're using Windows but not using Microsoft Visual Studio, you would need to change **`_MSC_VER`** to whatever is appropriate for your compiler.  Also, depending on your compiler, you may or may not need the #defines for the math functions.
- **`nn_alt_save.c`** is the same as **`nn_alt.c`** but adds saving the final neural net to a file and changing the number of epochs to 40.
- **`nn_alt_recog.c`** loads in the saved neural net (rather than retraining) and tries to recognize either the MNIST images or user supplied images (in ASCII form).  The supplied image should be (by default) an ASCII file 28 lines of 28 characters each with a `.` representing black, and any other character representing white.  Type **`nn_alt_recog -h`** to get usage information.  Type **`nn_alt_recog -a`** to get an ASCII image printout that can be used as a test input image.  Sample test images are provided.
- **`test*`** are test ASCII images that can be used like **`nn_alt_recog -ctest`**

>**Note:**  **`nn_alt*.c`** files have been reformatted due to Mark's preference for tab indentation rather than spaces and preference for non-Egyptian braces/brackets.  Sorry if this causes any inconvenience.

### Mark's Recognition Results

The saved neural net used was the result of 40 epochs of training.  Since the training is random and the split between training images and testing images is therefore random, your results may vary.

| Test File | Guessed Value | Correct Value | Result |
|-----------|:-------------:|:-------------:|:------:|
| test      |  4            |  9            | Fail   |
| testn     |  9            |  9            | Pass   |
| testn2    |  7            |  9            | Fail   |
| test2     |  7            |  7            | Pass   |
| test3     |  4            |  4            | Pass   |
| test4     |  6            |  6            | Pass   |
| test4n    |  6            |  6            | Pass   |
| test4n2   |  6            |  6            | Pass   |

Based on **`test4*`**, it seems to accept some variation but not too much (try moving `test4n2` over a little more and you'll see).

Perhaps using a greater number of epochs or changing **`TRAIN_SPLIT`** would help.

## License

This project is open-source and available under the [MIT License](LICENSE).
