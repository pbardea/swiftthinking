//
//  main.swift
//  NeuralNetwork
//
//  Created by Paul Bardea on 2015-12-26.
//  Copyright © 2015 pbardea stdios. All rights reserved.
//

import Foundation

func intToDoubleMatrix(input: [[Int]]) -> Matrix<Double> {
    let vectors = input.map {
        Vector($0.map { Double($0) })
    }
    return Matrix(vectors)
}

func test1() {
    
    let trainingSetInputs = Matrix([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ].map { Vector($0.map{Double($0)}) })
    let trainingSetOutputs = intToDoubleMatrix(input: [[0, 1, 1, 0]]).transpose
    
    let inputSize = trainingSetInputs[0].count
    
    let layer1width = 5
    
    let outputSize = 1
    
    let layer1 = NeuronLayer(numNeurons: layer1width, numInputsPerNeuron: inputSize)
    let layer2 = NeuronLayer(numNeurons: outputSize, numInputsPerNeuron: layer1width)
    
    let neuralNetwork = NeuralNetwork(layers: [layer1, layer2])
    
    neuralNetwork.train(trainingSetInputs, trainingSetOutputs: trainingSetOutputs, numberOfTrainingIterations: 10000)
    
    let outputs = neuralNetwork.think(intToDoubleMatrix(input: [[1, 1, 0]]))
    
    print("Predicted output for input [[1, 1, 0]]")
    print(outputs.last) // Outputs about 0.574
}

test1()
