#ifndef ZOLFONEURAL_H
#define ZOLFONEURAL_H

typedef enum {
	LINEAR,
	SIGMOID,
	RELU,
	SOFTMAX
} Activation;

typedef enum {
	ZERO,
	CONSTANT,
	UNIFORM,
	GAUSSIAN
} BiasInitMethod;

typedef struct {
	BiasInitMethod biasInitMethod;
	double constantVal;
    double gaussianAvg;
    double gaussianDeviation;
} BiasInitParams;

typedef struct {
	int numWeights;
    double* weights;
    BiasInitMethod biasInitMethod;
    double bias;
    double output;
    double delta;
} Neuron;

typedef struct {
    int numNeurons;
    Neuron* neurons;
    Activation activationType;
    double (*activation)(double);
    double (*activation_derivative)(double);
} Layer;

typedef struct {
    int numLayers;
    Layer* layers;
} NeuralNetwork;

//Initializes library
void initZolfoNeural();

//Create a neural network and initializes with random weights
NeuralNetwork newNeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int hiddenLayers[], Activation activations[]);

//Load a neural network from a file
NeuralNetwork loadNeuralNetwork(char* filename);

//Save a neural network to a file
void saveNeuralNetwork(char* filename, NeuralNetwork* neuralNetwork);

//Free the memory allocated to neuralNetwork
void freeNeuralNetwork(NeuralNetwork* neuralNetwork);

void initBiases(NeuralNetwork* neuralNetwork, BiasInitParams* biasInitParams);

void feedForward(NeuralNetwork* neuralNetwork, double* inputs);

void backwardPropagation(NeuralNetwork* neuralNetwork, double* inputs, double* targets, double learning_rate);

void train(NeuralNetwork* neuralNetwork, int numInputs, double inputs[][neuralNetwork->layers[0].numNeurons], double targets[][neuralNetwork->layers[neuralNetwork->numLayers - 1].numNeurons], double learningRate, int epochs);

double* predict(NeuralNetwork* neuralNetwork, double* inputs);

void loadDataset(char* filename, int numEntries, int numInputs, int numOutputs, double inputs[numEntries][numInputs], double targets[numEntries][numOutputs]);

//Prints neural network (NOT recommended for large networks)
void printNeuralNetwork(NeuralNetwork* neuralNetwork);

#endif
