#include "zolfoneural.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//Activations
//LINEAR
static double linear(double x);
static double linearDerivative(double x);
//SIGMOID
static double sigmoid(double x);
static double sigmoidDerivative(double x);
//TANH
static double tanhDerivative(double x);
//RELU
static double relu(double x);
static double reluDerivative(double x);
//SWISH
static double swish(double x);
static double swishDerivative(double x);
//MISH
static double mish(double x);
static double mishDerivative(double x);
//SOFTMAX
static double softmax(double x);
static double softmaxDerivative(double x);
//SOFTPLUS
static double softplus(double x);
static double softplusDerivative(double x);
//BINARYSTEP
static double binaryStep(double x);
static double binaryStepDerivative(double x);

static void setActivationFunction(Layer* layer, Activation activation);

static double randomWeight();
static void initWeights(NeuralNetwork* neuralNetwork);
static double generateUniformBias();
static double generateGaussianBias(double avg, double deviation);
static Layer newLayer(int numNeurons, Activation activation);

void initZolfoNeural()
{
	srand(time(NULL));
}

static double linear(double x)
{
	return x;
}

static double linearDerivative(double x)
{
	return 1;
}

static double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

static double sigmoidDerivative(double x)
{
	return x * (1.0 - x);
}

static double tanhDerivative(double x)
{
	double tanhX = tanh(x);
	return 1.0 - (tanhX * tanhX);
}

static double relu(double x)
{
    return (x > 0.0) ? x : 0.0;
}

static double reluDerivative(double x)
{
    return (x > 0.0) ? 1.0 : 0.0;
}

static double swish(double x)
{
	return x * sigmoid(x);
}

static double swishDerivative(double x)
{
	double sigmoidX = sigmoid(x);
	return sigmoidX + x * sigmoidX * (1.0 - sigmoidX);
}

static double mish(double x)
{
	return x * tanh(log(1.0 + exp(x)));
}

static double mishDerivative(double x)
{
	double exp2X = exp(2.0 * x);
	double denominator = 2.0 * exp(x) + exp2X + 2.0;
	return exp2X * (4.0 * x + 4.0) / (denominator * denominator);
}

static double softmax(double x)
{
	double expVal = exp(x);
    return expVal / (expVal + 1.0);
}

static double softmaxDerivative(double x)
{
	return x * (1.0 - x);
}

static double softplus(double x)
{
	return log(1.0 + exp(x));
}

static double softplusDerivative(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

static double binaryStep(double x)
{
	return (x >= 0) ? 1.0 : 0.0;
}

static double binaryStepDerivative(double x)
{
	return 0.0;
}

static void setActivationFunction(Layer* layer, Activation activation)
{
	layer->activationType = activation;
	switch(activation)
	{
		case LINEAR:
			layer->activation = &linear;
			layer->activationDerivative = &linearDerivative;
			break;
		case SIGMOID:
			layer->activation = &sigmoid;
			layer->activationDerivative = &sigmoidDerivative;
			break;
		case TANH:
			layer->activation = &tanh;
			layer->activationDerivative = &tanhDerivative;
			break;
		case RELU:
			layer->activation = &relu;
			layer->activationDerivative = &reluDerivative;
			break;
		case SWISH:
			layer->activation = &swish;
			layer->activationDerivative = &swishDerivative;
			break;
		case MISH:
			layer->activation = &mish;
			layer->activationDerivative = &mishDerivative;
			break;
		case SOFTMAX:
			layer->activation = &softmax;
			layer->activationDerivative = &softmaxDerivative;
			break;
		case SOFTPLUS:
			layer->activation = &softplus;
			layer->activationDerivative = &softplusDerivative;
			break;
		case BINARYSTEP:
			layer->activation = &binaryStep;
			layer->activationDerivative = &binaryStepDerivative;
			break;
	}
}

static double randomWeight()
{
	return (((double) rand()) / RAND_MAX * 2.0 - 1.0);
}

static void initWeights(NeuralNetwork* neuralNetwork)
{
	for(int layer = 1; layer < neuralNetwork->numLayers; layer++)
	{
		Layer* prevLayer = &neuralNetwork->layers[layer - 1];
		Layer* currLayer = &neuralNetwork->layers[layer];
		
		for(int neuron = 0; neuron < currLayer->numNeurons; neuron++)
		{
			Neuron* currNeuron = &currLayer->neurons[neuron];
			
			currNeuron->output = 0.0;
			currNeuron->delta = 0.0;
			
			currNeuron->numWeights = prevLayer->numNeurons;
			currNeuron->weights = (double*) malloc(currNeuron->numWeights * sizeof(double));
			
			for(int weight = 0; weight < currNeuron->numWeights; weight++)
			{
				currNeuron->weights[weight] = randomWeight();
			}
		}
	}
}

static double generateUniformBias()
{
	return (((double) rand()) / RAND_MAX * 2.0 - 1.0);
}

static double generateGaussianBias(double avg, double deviation)
{
	double rand1 = rand() / (double) RAND_MAX;
	double rand2 = rand() / (double) RAND_MAX;
	double res = sqrt(-2.0 * log(rand1)) * cos(2.0 * M_PI * rand2);
	
	return avg + deviation * res;
}

void initBiases(NeuralNetwork* neuralNetwork, BiasInitParams* biasInitParams)
{
	for(int layer = 1; layer < neuralNetwork->numLayers; layer++)
	{
		Layer* currLayer = &neuralNetwork->layers[layer];
		
		for(int neuron = 0; neuron < currLayer->numNeurons; neuron++)
		{
			Neuron* currNeuron = &currLayer->neurons[neuron];
			
			currNeuron->biasInitMethod = biasInitParams->biasInitMethod;
			
			switch(currNeuron->biasInitMethod)
			{
				case ZERO:
					currNeuron->bias = 0.0;
					break;
				case CONSTANT:
					currNeuron->bias = biasInitParams->constantVal;
					break;
				case UNIFORM:
					currNeuron->bias = generateUniformBias();
					break;
				case GAUSSIAN:
					currNeuron->bias = generateGaussianBias(biasInitParams->gaussianAvg, biasInitParams->gaussianDeviation);
					break;
			}
		}
	}
}

static Layer newLayer(int numNeurons, Activation activation)
{
	Layer layer;
	
	layer.numNeurons = numNeurons;
	layer.neurons = (Neuron*) malloc(numNeurons * sizeof(Neuron));
	setActivationFunction(&layer, activation);
	
	return layer;
}

NeuralNetwork newNeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int hiddenLayers[], Activation activations[])
{
	NeuralNetwork neuralNetwork;
	
	neuralNetwork.numLayers = numHiddenLayers + 2;
	neuralNetwork.layers = (Layer*) malloc(neuralNetwork.numLayers * sizeof(Layer));
	
	//Input layer
	neuralNetwork.layers[0] = newLayer(numInputs, activations[0]);
	
	//Hidden layers
	for(int layer = 1; layer <= numHiddenLayers; layer++)
	{
		neuralNetwork.layers[layer] = newLayer(hiddenLayers[layer - 1], activations[layer]);
	}
	
	//Output layer
	neuralNetwork.layers[neuralNetwork.numLayers - 1] = newLayer(numOutputs, activations[neuralNetwork.numLayers - 1]);
	
	initWeights(&neuralNetwork);
	
	BiasInitParams biasInitParams;
	biasInitParams.biasInitMethod = CONSTANT;
	biasInitParams.constantVal = 0.0;
	initBiases(&neuralNetwork, &biasInitParams);
	
	return neuralNetwork;
}

NeuralNetwork loadNeuralNetwork(char* filename)
{
	FILE* file = fopen(filename, "r");
	
	NeuralNetwork neuralNetwork;
	int numInputs, numOutputs, numHiddenLayers;
	int* hiddenLayers;
	Activation* activations;
	
	fscanf(file, "%d", &numInputs);
	fscanf(file, "%d", &numOutputs);
	fscanf(file, "%d", &numHiddenLayers);
	
	hiddenLayers = (int*) malloc(numHiddenLayers * sizeof(int));
	activations = (Activation*) malloc((numHiddenLayers + 1) * sizeof(Activation));
	
	for(int i = 0; i < numHiddenLayers; i++)
	{
		fscanf(file, "%d", &hiddenLayers[i]);
	}
	
	for(int i = 0; i < numHiddenLayers + 2; i++)
	{
		fscanf(file, "%d", &activations[i]);
	}
	
	neuralNetwork = newNeuralNetwork(numInputs, numOutputs, numHiddenLayers, hiddenLayers, activations);
	
	for(int layer = 1; layer < neuralNetwork.numLayers; layer++)
	{
		Layer* currLayer = &neuralNetwork.layers[layer];
		
		for(int neuron = 0; neuron < currLayer->numNeurons; neuron++)
		{
			Neuron* currNeuron = &currLayer->neurons[neuron];
			
			currNeuron->output = 0.0;
			currNeuron->delta = 0.0;
			
			for(int weight = 0; weight < currNeuron->numWeights; weight++)
			{
				fscanf(file, "%lf", &currNeuron->weights[weight]);
			}
			
			fscanf(file, "%lf", &currNeuron->bias);
		}
	}
	
	fclose(file);
	
	return neuralNetwork;
}

void saveNeuralNetwork(char* filename, NeuralNetwork* neuralNetwork)
{
	FILE* file = fopen(filename, "w");
	
	int numInputs = neuralNetwork->layers[0].numNeurons;
	int numOutputs = neuralNetwork->layers[neuralNetwork->numLayers - 1].numNeurons;
	int numHiddenLayers = neuralNetwork->numLayers - 2;
	
	fprintf(file, "%d %d %d ", numInputs, numOutputs, numHiddenLayers);
	
	for(int i = 1; i <= numHiddenLayers; i++)
	{
		fprintf(file, "%d ", neuralNetwork->layers[i].numNeurons);
	}
	
	for(int i = 0; i < neuralNetwork->numLayers; i++)
	{
		fprintf(file, "%d ", neuralNetwork->layers[i].activationType);
	}
	
	for(int layer = 1; layer < neuralNetwork->numLayers; layer++)
	{
		Layer* currLayer = &neuralNetwork->layers[layer];
		
		for(int neuron = 0; neuron < currLayer->numNeurons; neuron++)
		{
			Neuron* currNeuron = &currLayer->neurons[neuron];
			
			for(int weight = 0; weight < currNeuron->numWeights; weight++)
			{
				fprintf(file, "%.10lf ", currNeuron->weights[weight]);
			}
			
			fprintf(file, "%.10lf ", currNeuron->bias);
		}
	}
	
	fclose(file);
}

void freeNeuralNetwork(NeuralNetwork* neuralNetwork)
{
	for(int layer = 1; layer < neuralNetwork->numLayers; layer++)
	{
		Layer* currLayer = &neuralNetwork->layers[layer];
		
		for(int neuron = 0; neuron < currLayer->numNeurons; neuron++)
		{
			Neuron* currNeuron = &currLayer->neurons[neuron];
			
			free(currNeuron->weights);
		}
		free(currLayer->neurons);
	}
	free(neuralNetwork->layers);
}

void feedForward(NeuralNetwork* neuralNetwork, double* inputs)
{
	Layer* inputLayer = &neuralNetwork->layers[0];
	for(int neuron = 0; neuron < inputLayer->numNeurons; neuron++)
	{
		inputLayer->neurons[neuron].output = inputs[neuron];
	}
	
	for(int layer = 1; layer < neuralNetwork->numLayers; layer++)
	{
		Layer* prevLayer = &neuralNetwork->layers[layer - 1];
		Layer* currLayer = &neuralNetwork->layers[layer];
		
		for(int neuron = 0; neuron < currLayer->numNeurons; neuron++)
		{
			Neuron* currNeuron = &currLayer->neurons[neuron];
			
			double weightedSum = 0.0;
			for(int weight = 0; weight < currNeuron->numWeights; weight++)
			{
				weightedSum += prevLayer->neurons[weight].output * currNeuron->weights[weight];
			}
			
			weightedSum += currNeuron->bias;
			
			currNeuron->output = (*(currLayer->activation))(weightedSum);
		}
	}
}

void backwardPropagation(NeuralNetwork* neuralNetwork, double* inputs, double* targets, double learningRate)
{
	feedForward(neuralNetwork, inputs);
	
	//Delta update
	Layer* outputLayer = &neuralNetwork->layers[neuralNetwork->numLayers - 1];
	for(int neuron = 0; neuron < outputLayer->numNeurons; neuron++)
	{
		Neuron* currNeuron = &outputLayer->neurons[neuron];
		
		double error = targets[neuron] - currNeuron->output;
		currNeuron->delta = error * (*(outputLayer->activationDerivative))(currNeuron->output);
	}
	
	for(int layer = neuralNetwork->numLayers - 2; layer > 0; layer--)
	{
		Layer* currLayer = &neuralNetwork->layers[layer];
		Layer* nextLayer = &neuralNetwork->layers[layer + 1];
		
		for(int neuron = 0; neuron < currLayer->numNeurons; neuron++)
		{
			Neuron* currNeuron = &currLayer->neurons[neuron];
			
			double delta = 0.0;
			for(int neuronNext = 0; neuronNext < nextLayer->numNeurons; neuronNext++)
			{
				delta += nextLayer->neurons[neuronNext].weights[neuron] * nextLayer->neurons[neuronNext].delta;
			}
			delta *= (*(currLayer->activationDerivative))(currNeuron->output);
			currNeuron->delta = delta;
		}
	}
	
	//Weights and biases update
	Layer* firstLayer = &neuralNetwork->layers[1];
	for(int neuron = 0; neuron < firstLayer->numNeurons; neuron++)
	{
		Neuron* currNeuron = &firstLayer->neurons[neuron];
		
		for(int weight = 0; weight < currNeuron->numWeights; weight++)
		{
			currNeuron->weights[weight] += learningRate * (inputs[weight] * currNeuron->delta);
		}
		
		if(currNeuron->biasInitMethod != CONSTANT)
		{
			currNeuron->bias += learningRate * currNeuron->delta;
		}
	}
	
	for(int layer = 2; layer < neuralNetwork->numLayers; layer++)
	{
		Layer* prevLayer = &neuralNetwork->layers[layer - 1];
		Layer* currLayer = &neuralNetwork->layers[layer];
		
		for(int neuron = 0; neuron < currLayer->numNeurons; neuron++)
		{
			Neuron* currNeuron = &currLayer->neurons[neuron];
			
			for(int weight = 0; weight < currNeuron->numWeights; weight++)
			{
				currNeuron->weights[weight] += learningRate * (prevLayer->neurons[weight].output * currNeuron->delta);
			}
			
			if(currNeuron->biasInitMethod != CONSTANT)
			{
				currNeuron->bias += learningRate * currNeuron->delta;
			}
		}
	}
}

void train(NeuralNetwork* neuralNetwork, int numInputs, double inputs[][neuralNetwork->layers[0].numNeurons], double targets[][neuralNetwork->layers[neuralNetwork->numLayers - 1].numNeurons], double learningRate, int epochs)
{
	for(int epoch = 0; epoch < epochs; epoch++)
	{
		for(int input = 0; input < numInputs; input++)
		{
			backwardPropagation(neuralNetwork, inputs[input], targets[input], learningRate);
		}
	}
}

double* predict(NeuralNetwork* neuralNetwork, double* inputs)
{
	feedForward(neuralNetwork, inputs);
	
	Layer* outputLayer = &neuralNetwork->layers[neuralNetwork->numLayers - 1];
	double* output = (double*) malloc(outputLayer->numNeurons * sizeof(double));
	
	for(int i = 0; i < outputLayer->numNeurons; i++)
	{
		output[i] = outputLayer->neurons[i].output;
	}
	
	return output;
}

void loadDataset(char* filename, int numEntries, int numInputs, int numOutputs, double inputs[numEntries][numInputs], double targets[numEntries][numOutputs])
{
	FILE* file = fopen(filename, "r");
	
	for(int entry = 0; entry < numEntries; entry++)
	{
		for(int input = 0; input < numInputs; input++)
		{
			fscanf(file, "%lf", &inputs[entry][input]);
		}
		
		for(int output = 0; output < numOutputs; output++)
		{
			fscanf(file, "%lf", &targets[entry][output]);
		}
	}
	
	fclose(file);
}

void zScoreNormalization(int numEntries, int numInputs, double inputs[numEntries][numInputs])
{
	for(int input = 0; input < numInputs; input++)
	{
		double* vals = (double*) malloc(numEntries * sizeof(double));
		for(int entry = 0; entry < numEntries; entry++)
		{
			vals[entry] = inputs[entry][input];
		}
		
		double sum = 0.0, avg;
		for(int val = 0; val < numEntries; val++)
		{
			sum += vals[val];
		}
		avg = sum / numEntries;
		
		double squaredSum = 0.0, stdDeviation;
		for(int val = 0; val < numEntries; val++)
		{
			squaredSum += pow(vals[val] - avg, 2);
		}
		stdDeviation = sqrt(squaredSum / numEntries);
		
		for(int entry = 0; entry < numEntries; entry++)
		{
			inputs[entry][input] = (inputs[entry][input] - avg) / stdDeviation;
		}
		
		free(vals);
	}
}

void minMaxScaling(int numEntries, int numInputs, double inputs[numEntries][numInputs])
{
	double minVals[numInputs], maxVals[numInputs];
	
	for(int input = 0; input < numInputs; input++)
	{
		minVals[input] = inputs[0][input];
		maxVals[input] = inputs[0][input];
	}
	
	for(int entry = 0; entry < numEntries; entry++)
	{
		for(int input = 0; input < numInputs; input++)
		{
			if(inputs[entry][input] < minVals[input])
			{
				minVals[input] = inputs[entry][input];
			}
			if(inputs[entry][input] > maxVals[input])
			{
				maxVals[input] = inputs[entry][input];
			}
		}
	}
	
	for(int entry = 0; entry < numEntries; entry++)
	{
		for(int input = 0; input < numInputs; input++)
		{
			inputs[entry][input] = (inputs[entry][input] - minVals[input]) / (maxVals[input] - minVals[input]);
		}
	}
}

void maxAbsScaling(int numEntries, int numInputs, double inputs[numEntries][numInputs])
{
	double maxAbsVals[numInputs];
	for(int input = 0; input < numInputs; input++)
	{
		for(int entry = 0; entry < numEntries; entry++)
		{
			if(abs(inputs[entry][input]) > maxAbsVals[input])
			{
				maxAbsVals[input] = abs(inputs[entry][input]);
			}
		}
	}
	
	for(int entry = 0; entry < numEntries; entry++)
	{
		for(int input = 0; input < numInputs; input++)
		{
			inputs[entry][input] /= maxAbsVals[input];
		}
	}
}

void logScaling(int numEntries, int numInputs, double inputs[numEntries][numInputs])
{
	for(int entry = 0; entry < numEntries; entry++)
	{
		for(int input = 0; input < numInputs; input++)
		{
			inputs[entry][input] = log(1 + inputs[entry][input]);
		}
	}
}

void powerTransformation(int numEntries, int numInputs, double inputs[numEntries][numInputs], double exponent)
{
	for(int entry = 0; entry < numEntries; entry++)
	{
		for(int input = 0; input < numInputs; input++)
		{
			inputs[entry][input] = pow(inputs[entry][input], exponent);
		}
	}
}

void printNeuralNetwork(NeuralNetwork* neuralNetwork)
{
	puts("NeuralNetwork:");
	for(int layer = 1; layer < neuralNetwork->numLayers; layer++)
	{
		Layer* currLayer = &neuralNetwork->layers[layer];
		
		printf("\tLayer: %d\n", layer);
		printf("\t\tActivation: %d\n", currLayer->activationType);
		
		for(int neuron = 0; neuron < currLayer->numNeurons; neuron++)
		{
			Neuron* currNeuron = &currLayer->neurons[neuron];
			
			printf("\t\tNeuron: %d\n", neuron);
			printf("\t\t\tBias: %lf\n", currNeuron->bias);
			printf("\t\t\tOutput: %lf\n", currNeuron->output);
			printf("\t\t\tDelta: %lf\n", currNeuron->delta);
			
			for(int weight = 0; weight < currNeuron->numWeights; weight++)
			{
				printf("\t\t\tWeight: %d,\tValue: %.10lf\n", weight, currNeuron->weights[weight]);
			}
		}
	}
	puts("");
}
