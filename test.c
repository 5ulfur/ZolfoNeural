#include "zolfoneural.h"
#include <stdio.h>
#include <math.h>

int main()
{
	initZolfoNeural();
	
	//Neural network structure
	int numInputs = 4;
	int numOutputs = 3;
	int numHiddenLayers = 1;
	int hiddenLayers[] = {4};
	Activation activations[] = {LINEAR, SIGMOID, SIGMOID};
	
	//Hyperparameters
	int epochs = 10000;
	double learningRate = 0.1;
	
	//Dataset
	int numEntries = 150;
	double inputs[numEntries][numInputs];
	double targets[numEntries][numOutputs];
	loadDataset("dataset/dataset_iris.txt", numEntries, numInputs, numOutputs, inputs, targets);
	
	NeuralNetwork neuralNetwork = newNeuralNetwork(numInputs, numOutputs, numHiddenLayers, hiddenLayers, activations);
	train(&neuralNetwork, sizeof(inputs) / sizeof(inputs[0]), inputs, targets, learningRate, epochs);
	
	double testInput[] = {5.9, 3.0, 4.7, 1.4}; //Expected output: "Iris versicolor"
	double* output = predict(&neuralNetwork, testInput);
	if(round(output[0]))
	{
		puts("Iris setosa");
	}
	else if(round(output[1]))
	{
		puts("Iris versicolor");
	}
	else if(round(output[2]))
	{
		puts("Iris virginica");
	}
	
	saveNeuralNetwork("iris_example.txt", &neuralNetwork);
	
	freeNeuralNetwork(&neuralNetwork);
	return 0;
}
