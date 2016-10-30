//
//  NeuralNetTests.swift
//  NeuralNetTests
//
//  Created by Paul Bardea on 2016-10-30.
//  Copyright © 2016 pbardea. All rights reserved.
//

import XCTest
@testable import NeuralNet

class NeuralNetTests: XCTestCase {
    
    func getTrainedNet() -> NeuralNetwork {
        
        let trainingSetInputs = Matrix([
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
            ].map { Vector($0.map{Double($0)}) })
        let trainingSetOutputs = Matrix<Double>.doubleMatrixFrom(input: [[0, 1, 1, 0]]).transpose
        
        let inputSize = 3
        
        let layer1width = 5
        
        let outputSize = 1
        
        let layer1 = NeuronLayer(numNeurons: layer1width, numInputsPerNeuron: inputSize)
        let layer2 = NeuronLayer(numNeurons: outputSize, numInputsPerNeuron: layer1width)
        
        let neuralNetwork = NeuralNetwork(layers: [layer1, layer2])
        
        neuralNetwork.train(trainingSetInputs, trainingSetOutputs: trainingSetOutputs, numberOfTrainingIterations:4000)

        return neuralNetwork
    }
    
    func test1() {
        
        let userInput = [1,1,0]
        
        let neuralNetwork = getTrainedNet()
        
        let outputs = neuralNetwork.think(Matrix.doubleMatrixFrom(input: [userInput]))
        
        guard let output = outputs.last else { return }
        print("Predicted output for input \(userInput) is \(output[0][0])")
        print(output[0][0])
    }
    
}
