#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001f
#define EPOCHS 20
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8

#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"

typedef struct {
    float *weights, *biases;
    int input_size, output_size;
} Layer;

typedef struct {
    Layer hidden, output;
} Network;

typedef struct {
    unsigned char *images, *labels;
    int nImages;
} InputData;

void softmax(float *input, int size) {
    float max = input[0], sum = 0;
    for (int i = 1; i < size; i++)
        if (input[i] > max) max = input[i];
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++)
        input[i] /= sum;
}

void init_layer(Layer *layer, int in_size, int out_size) {
    int n = in_size * out_size;
    float scale = sqrtf(2.0f / in_size);

    layer->input_size = in_size;
    layer->output_size = out_size;
    layer->weights = malloc(n * sizeof(float));
    layer->biases = calloc(out_size, sizeof(float));

    for (int i = 0; i < n; i++)
        layer->weights[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
}

void forward(Layer *layer, float *input, float *output) {
    for (int i = 0; i < layer->output_size; i++) {
        output[i] = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++)
            output[i] += input[j] * layer->weights[j * layer->output_size + i];
    }
}

void backward(Layer *layer, float *input, float *output_grad, float *input_grad, float lr) {
    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            int idx = j * layer->output_size + i;
            float grad = output_grad[i] * input[j];
            layer->weights[idx] -= lr * grad;
            if (input_grad)
                input_grad[j] += output_grad[i] * layer->weights[idx];
        }
        layer->biases[i] -= lr * output_grad[i];
    }
}

void train(Network *net, float *input, int label, float lr) {
    float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];
    float output_grad[OUTPUT_SIZE] = {0}, hidden_grad[HIDDEN_SIZE] = {0};

    forward(&net->hidden, input, hidden_output);
    for (int i = 0; i < HIDDEN_SIZE; i++)
        hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0;  // ReLU

    forward(&net->output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++)
        output_grad[i] = final_output[i] - (i == label);

    backward(&net->output, hidden_output, output_grad, hidden_grad, lr);

    for (int i = 0; i < HIDDEN_SIZE; i++)
        hidden_grad[i] *= hidden_output[i] > 0 ? 1 : 0;  // ReLU derivative

    backward(&net->hidden, input, hidden_grad, NULL, lr);
}

int predict(Network *net, float *input) {
    float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];

    forward(&net->hidden, input, hidden_output);
    for (int i = 0; i < HIDDEN_SIZE; i++)
        hidden_output[i] = hidden_output[i] > 0 ? hidden_output[i] : 0;  // ReLU

    forward(&net->output, hidden_output, final_output);
    softmax(final_output, OUTPUT_SIZE);

    int max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++)
        if (final_output[i] > final_output[max_index])
            max_index = i;

    return max_index;
}

void read_mnist_images(const char *filename, unsigned char **images, int *nImages) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int temp, rows, cols;
    fread(&temp, sizeof(int), 1, file);
    fread(nImages, sizeof(int), 1, file);
    *nImages = __builtin_bswap32(*nImages);

    fread(&rows, sizeof(int), 1, file);
    fread(&cols, sizeof(int), 1, file);

    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    *images = malloc((*nImages) * IMAGE_SIZE * IMAGE_SIZE);
    fread(*images, sizeof(unsigned char), (*nImages) * IMAGE_SIZE * IMAGE_SIZE, file);
    fclose(file);
}

void read_mnist_labels(const char *filename, unsigned char **labels, int *nLabels) {
    FILE *file = fopen(filename, "rb");
    if (!file) exit(1);

    int temp;
    fread(&temp, sizeof(int), 1, file);
    fread(nLabels, sizeof(int), 1, file);
    *nLabels = __builtin_bswap32(*nLabels);

    *labels = malloc(*nLabels);
    fread(*labels, sizeof(unsigned char), *nLabels, file);
    fclose(file);
}

void shuffle_data(unsigned char *images, unsigned char *labels, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        for (int k = 0; k < INPUT_SIZE; k++) {
            unsigned char temp = images[i * INPUT_SIZE + k];
            images[i * INPUT_SIZE + k] = images[j * INPUT_SIZE + k];
            images[j * INPUT_SIZE + k] = temp;
        }
        unsigned char temp = labels[i];
        labels[i] = labels[j];
        labels[j] = temp;
    }
}

int main() {
    Network net;
    InputData data = {0};
    float learning_rate = LEARNING_RATE, img[INPUT_SIZE];

    srand(time(NULL));

    init_layer(&net.hidden, INPUT_SIZE, HIDDEN_SIZE);
    init_layer(&net.output, HIDDEN_SIZE, OUTPUT_SIZE);

    read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages);
    read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nImages);

    shuffle_data(data.images, data.labels, data.nImages);

    int train_size = (int)(data.nImages * TRAIN_SPLIT);
    int test_size = data.nImages - train_size;

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0;
        for (int i = 0; i < train_size; i += BATCH_SIZE) {
            for (int j = 0; j < BATCH_SIZE && i + j < train_size; j++) {
                int idx = i + j;
                for (int k = 0; k < INPUT_SIZE; k++)
                    img[k] = data.images[idx * INPUT_SIZE + k] / 255.0f;

                train(&net, img, data.labels[idx], learning_rate);

                float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];
                forward(&net.hidden, img, hidden_output);
                for (int k = 0; k < HIDDEN_SIZE; k++)
                    hidden_output[k] = hidden_output[k] > 0 ? hidden_output[k] : 0;  // ReLU
                forward(&net.output, hidden_output, final_output);
                softmax(final_output, OUTPUT_SIZE);

                total_loss += -logf(final_output[data.labels[idx]] + 1e-10f);
            }
        }
        int correct = 0;
        for (int i = train_size; i < data.nImages; i++) {
            for (int k = 0; k < INPUT_SIZE; k++)
                img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;
            if (predict(&net, img) == data.labels[i])
                correct++;
        }
        printf("Epoch %d, Accuracy: %.2f%%, Avg Loss: %.4f\n", epoch + 1, (float)correct / test_size * 100, total_loss / train_size);
    }

    free(net.hidden.weights);
    free(net.hidden.biases);
    free(net.output.weights);
    free(net.output.biases);
    free(data.images);
    free(data.labels);

    return 0;
}
