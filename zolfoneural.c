#include "zolfoneural.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

//Activations
//LINEAR
static double linear(double x);
static double linear_derivative(double x);
//SIGMOID
static double sigmoid(double x);
static double sigmoid_derivative(double x);
//RELU
static double relu(double x);
static double relu_derivative(double x);
//SOFTMAX
static double softmax(double x);
static double softmax_derivative(double x);

static void setActivationFunction(Layer* layer, enum Activation activation);

static double randomWeight();
static void initializeWeights(NeuralNetwork* neuralNetwork);
static Layer newLayer(int numNeurons, enum Activation activation);

static double linear(double x)
{
	return x;
}

static double linear_derivative(double x)
{
	return 1;
}

static double sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}

static double sigmoid_derivative(double x)
{
	return x * (1 - x);
}

static double relu(double x)
{
    return (x > 0.0) ? x : 0.0;
}

static double relu_derivative(double x)
{
    return (x > 0.0) ? 1.0 : 0.0;
}

static double softmax(double x)
{
	double exp_val = exp(x);
    return exp_val / (exp_val + 1.0);
}

static double softmax_derivative(double x)
{
	return x * (1.0 - x);
}

static void setActivationFunction(Layer* layer, enum Activation activation)
{
	layer->activationType = activation;
	switch(activation)
	{
		case LINEAR:
			layer->activation = &linear;
			layer->activation_derivative = &linear_derivative;
			break;
		case SIGMOID:
			layer->activation = &sigmoid;
			layer->activation_derivative = &sigmoid_derivative;
			break;
		case RELU:
			layer->activation = &relu;
			layer->activation_derivative = &relu_derivative;
			break;
		case SOFTMAX:
			layer->activation = &softmax;
			layer->activation_derivative = &softmax_derivative;
			break;
	}
}

static double randomWeight()
{
	return (((double) rand()) / RAND_MAX * 2.0 - 1.0);
}

static void initializeWeights(NeuralNetwork* neuralNetwork)
{
	srand(time(NULL));
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

static Layer newLayer(int numNeurons, enum Activation activation)
{
	Layer layer;
	
	layer.numNeurons = numNeurons;
	layer.neurons = (Neuron*) malloc(numNeurons * sizeof(Neuron));
	setActivationFunction(&layer, activation);
	
	return layer;
}

NeuralNetwork newNeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int hiddenLayers[], enum Activation activations[])
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
	
	initializeWeights(&neuralNetwork);
	
	return neuralNetwork;
}

NeuralNetwork loadNeuralNetwork(char* filename)
{
	FILE* file = fopen(filename, "r");
	
	NeuralNetwork neuralNetwork;
	int numInputs, numOutputs, numHiddenLayers;
	int* hiddenLayers;
	enum Activation* activations;
	
	fscanf(file, "%d", &numInputs);
	fscanf(file, "%d", &numOutputs);
	fscanf(file, "%d", &numHiddenLayers);
	
	hiddenLayers = (int*) malloc(numHiddenLayers * sizeof(int));
	activations = (enum Activation*) malloc((numHiddenLayers + 1) * sizeof(enum Activation));
	
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
			
			currNeuron->output = 0.0;
			currNeuron->delta = 0.0;
			
			for(int weight = 0; weight < currNeuron->numWeights; weight++)
			{
				fprintf(file, "%.10lf ", currNeuron->weights[weight]);
			}
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
		currNeuron->delta = error * (*(outputLayer->activation_derivative))(currNeuron->output);
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
			delta *= (*(currLayer->activation_derivative))(currNeuron->output);
			currNeuron->delta = delta;
		}
	}
	
	//Weights update
	Layer* firstLayer = &neuralNetwork->layers[1];
	for(int neuron = 0; neuron < firstLayer->numNeurons; neuron++)
	{
		Neuron* currNeuron = &firstLayer->neurons[neuron];
		
		for(int weight = 0; weight < currNeuron->numWeights; weight++)
		{
			currNeuron->weights[weight] += learningRate * (inputs[weight] * currNeuron->delta);
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
