#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.0005f
#define MOMENTUM 0.9f
#define EPOCHS 40
#define BATCH_SIZE 64
#define IMAGE_SIZE 28
#define TRAIN_SPLIT 0.8
#define PRINT_INTERVAL 1000

#ifdef _MSC_VER
#define expf exp
#define logf log
#define sqrtf sqrt

int __builtin_bswap32(int vin)
{
unsigned char *vcast=(unsigned char *)&vin;
unsigned char vrev[4];
int vout;

vrev[3]=vcast[0]; vrev[2]=vcast[1]; vrev[1]=vcast[2]; vrev[0]=vcast[3];
vout=*(int *)vrev;
return(vout);
}

#define TRAIN_IMG_PATH "data\\train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data\\train-labels.idx1-ubyte"
#else
#define TRAIN_IMG_PATH "data/train-images.idx3-ubyte"
#define TRAIN_LBL_PATH "data/train-labels.idx1-ubyte"
#endif

typedef struct
	{
	float *weights, *biases, *weight_momentum, *bias_momentum;
	int input_size, output_size;
	} Layer;

typedef struct
	{
	Layer hidden, output;
	} Network;

typedef struct
	{
	unsigned char *images, *labels;
	int nImages;
	} InputData;

void softmax(float *input, int size)
{
int i;
float max = input[0], sum = 0;

for (i = 1; i < size; i++) { if (input[i] > max) max = input[i]; }
for (i = 0; i < size; i++)
	{
	input[i] = expf(input[i] - max);
	sum += input[i];
	}
for (i = 0; i < size; i++) { input[i] /= sum; }
}

void forward(Layer *layer, float *input, float *output)
{
int i, j;

for (i = 0; i < layer->output_size; i++)
	output[i] = layer->biases[i];

for (j = 0; j < layer->input_size; j++)
	{
	float in_j = input[j];
	float *weight_row = &layer->weights[j * layer->output_size];
	for (i = 0; i < layer->output_size; i++)
		{
		output[i] += in_j * weight_row[i];
		}
	}

for (i = 0; i < layer->output_size; i++)
	output[i] = output[i] > 0 ? output[i] : 0;
}

int predict(Network *net, float *input)
{
int i;
int max_index = 0;
float hidden_output[HIDDEN_SIZE], final_output[OUTPUT_SIZE];

forward(&net->hidden, input, hidden_output);
forward(&net->output, hidden_output, final_output);
softmax(final_output, OUTPUT_SIZE);

for (i = 1; i < OUTPUT_SIZE; i++)
	if (final_output[i] > final_output[max_index])
		max_index = i;

return(max_index);
}

void read_mnist_images(const char *filename, unsigned char **images, int *nImages)
{
int temp, rows, cols;
FILE *file = fopen(filename, "rb");
if (!file) exit(1);

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

void read_mnist_labels(const char *filename, unsigned char **labels, int *nLabels)
{
int temp;
FILE *file = fopen(filename, "rb");
if (!file) exit(1);

fread(&temp, sizeof(int), 1, file);
fread(nLabels, sizeof(int), 1, file);
*nLabels = __builtin_bswap32(*nLabels);

*labels = malloc(*nLabels);
fread(*labels, sizeof(unsigned char), *nLabels, file);
fclose(file);
}

void shuffle_data(unsigned char *images, unsigned char *labels, int n)
{
int i, k;
unsigned char temp;

for (i = n - 1; i > 0; i--)
	{
	int j = rand() % (i + 1);
	for (k = 0; k < INPUT_SIZE; k++)
		{
		temp = images[i * INPUT_SIZE + k];
		images[i * INPUT_SIZE + k] = images[j * INPUT_SIZE + k];
		images[j * INPUT_SIZE + k] = temp;
		}
	temp = labels[i];
	labels[i] = labels[j];
	labels[j] = temp;
	}
}

void print_image(InputData *d, int i, int alt)
{
char acols[16] = " ,;-+=*xoOX@W#$%";
unsigned int c, x, y;
char *p = d->images + i * IMAGE_SIZE * IMAGE_SIZE;
if (alt) { acols[0] = '.'; }
if (*(d->labels + i) != '?') { printf("%d: \n", *(d->labels + i)); }
for (y = 0; y < IMAGE_SIZE; y++)
	{
	for (x = 0; x < IMAGE_SIZE; x++)
		{
		c = (unsigned char)(*(p + y * IMAGE_SIZE + x));
		printf("%c", acols[c / 16]);
		}
	printf("\n");
	}
}

void write_layer(Layer *layer, FILE *fp)
{
int n = layer->input_size * layer->output_size;

fwrite(&layer->input_size, sizeof(int), 1, fp);
fwrite(&layer->output_size, sizeof(int), 1, fp);
fwrite(layer->weights, sizeof(float), n, fp);
fwrite(layer->biases, sizeof(float), layer->output_size, fp);
fwrite(layer->weight_momentum, sizeof(float), n, fp);
fwrite(layer->bias_momentum, sizeof(float), layer->output_size, fp);
}

void read_layer(Layer *layer, FILE *fp)
{
int n;

fread(&layer->input_size, sizeof(int), 1, fp);
fread(&layer->output_size, sizeof(int), 1, fp);
n = layer->input_size * layer->output_size;
layer->weights = malloc(n * sizeof(float));
layer->biases = calloc(layer->output_size, sizeof(float));
layer->weight_momentum = malloc(n * sizeof(float));
layer->bias_momentum = calloc(layer->output_size, sizeof(float));
fread(layer->weights, sizeof(float), n, fp);
fread(layer->biases, sizeof(float), layer->output_size, fp);
fread(layer->weight_momentum,sizeof(float), n, fp);
fread(layer->bias_momentum, sizeof(float), layer->output_size, fp);
}

void load_test_image(char *filename, InputData *data)
{
char *p;
int x, y, c;
FILE *fp = fopen(filename, "r");
if (!fp) { printf("Can't open '%s'\n", filename); exit(1); }
data->images = malloc(IMAGE_SIZE * IMAGE_SIZE);
data->labels = malloc(1);
data->nImages = 1;
data->labels[0] = '?';
p = data->images;
for (y = 0; y < IMAGE_SIZE; y++)
	{
	for (x = 0; x < IMAGE_SIZE; x++)
		{
		c = fgetc(fp);
		if (c == '.') { *p = 0; }
		else { *p = 255; }
		p++;
		}
	fgetc(fp);
	}
fclose(fp);
}

int main(int argc, char **argv)
{
int epoch, i, j, k;
Network net;
InputData data = {0};
float learning_rate = LEARNING_RATE, img[INPUT_SIZE];
clock_t start, end;
double cpu_time_used;
FILE *fp;
int guess;
float total_loss = 0;
int correct = 0, n = 1, alt = 0;
int load_mnist_images = 1;

for (i = 1; i < argc; i++)
	{
	if (argv[i][0] == '-')
		{
		switch (argv[i][1])
			{
			case 'c':
				load_test_image(argv[i] + 2, &data);
				load_mnist_images = 0;
				n = 1;
				break;
			case 'n': if (load_mnist_images) { sscanf(argv[i] + 2, "%d", &n); }
				break;
			case 'a': alt = 1; break;
			case 'h':
				printf("Usage: nn_alt_recog [-c{filename}] [-n{number}] [-a] [-h]\n");
				printf("       -c - Load test image from ascii file\n");
				printf("       -n - If using MNIST data, number of tests\n");
				printf("       -a - Replace space with '.' in ASCII print\n");
				printf("       -h - Help\n");
				exit(1);
			}
		}
	}

if (load_mnist_images)
	{
	srand(time(NULL));

	read_mnist_images(TRAIN_IMG_PATH, &data.images, &data.nImages);
	read_mnist_labels(TRAIN_LBL_PATH, &data.labels, &data.nImages);

	shuffle_data(data.images, data.labels, data.nImages);
	if (n > data.nImages) { n = data.nImages; }
	}

fp = fopen("nnet.raw", "rb");
if (!fp) { printf("Can't open 'nnet.raw'.  May need to run nn_alt_save\n"); exit(1); }
read_layer(&net.hidden, fp);
read_layer(&net.output, fp);
fclose(fp);

for (i = 0; i < n; i++)
	{
	print_image(&data, i, alt);
	for (k = 0; k < INPUT_SIZE; k++)
		img[k] = data.images[i * INPUT_SIZE + k] / 255.0f;
	if ((guess = predict(&net, img)) == data.labels[i]) correct++;
	printf("Guess: %d\n", guess);
	if (load_mnist_images) printf("Correct Count: %d\n", correct);
	}
if (load_mnist_images) printf("Accuracy: %.2f%%\n", (float)correct / n * 100);

free(net.hidden.weights);
free(net.hidden.biases);
free(net.output.weights);
free(net.output.biases);
free(data.images);
free(data.labels);

return(0);
}
