//
//  NNTests.swift
//  NNTests
//
//  Created by Paul Bardea on 2015-12-26.
//  Copyright © 2015 pbardea stdios. All rights reserved.
//

import XCTest

class LinAlgTests: XCTestCase {
    
    func testDotVector() {
        let a = [1.0, 2.0, 3.0]
        let b = [3.0, 4.0, 5.0]
        let result = dotVector(a, withB: b)
        XCTAssertTrue(result == 26)
    }
    
    func testDotMatrix() {
        let a = [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]
        let b = [[10.0, 11.0, 12.0], [4.0, 6.0, 7.0]]
        let result = dotMatrix(a, withB: b)
        XCTAssertTrue(result == [[18.0, 23.0, 26.0], [60.0, 74.0, 83.0], [102.0, 125.0, 140.0]])
    }
    
}

class NNTests: XCTestCase {
    
    func testGeneralNetwork() {
        let training_set_inputs = IntToDoubleMatrix([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]])
        let training_set_outputs = transpose(IntToDoubleMatrix([[0, 1, 1, 1, 1, 0, 0]]))
        
        let inputSize = training_set_inputs[0].count;
        
        let layer1width = 5;
        let layer2width = 6;
        
        let outputSize = 1;
        
        let layer1 = NeuronLayer(numNeurons: layer1width, numInputsPerNeuron: inputSize)
        let layer2 = NeuronLayer(numNeurons: layer2width, numInputsPerNeuron: layer1width)
        let layer3 = NeuronLayer(numNeurons: outputSize, numInputsPerNeuron: layer2width)
        
        let neural_network = NeuralNetwork(layers: [layer1, layer2, layer3])
        
        
        neural_network.train(training_set_inputs, trainingSetOutputs: training_set_outputs, numberOfTrainingIterations: 6)
        
        let outputs = neural_network.think([[1,1,0]])
        print("Predicted output for input [[1,1,0]]")
        print(outputs.last)
    }
    
    func testTwoInput() {
        let inputSize = 2;
        
        let layer1width = 5;
        let layer2width = 6;
        
        let outputSize = 1;
        
        let layer1 = NeuronLayer(numNeurons: layer1width, numInputsPerNeuron: inputSize)
        let layer2 = NeuronLayer(numNeurons: layer2width, numInputsPerNeuron: layer1width)
        let layer3 = NeuronLayer(numNeurons: outputSize, numInputsPerNeuron: layer2width)
        
        let neural_network = ThreeLayerNeuralNetwork(layer1: layer1, layer2: layer2, layer3: layer3)
        
        let training_set_inputs = IntToDoubleMatrix([[0, 0], [0, 1], [1, 0], [1, 1]])
        let training_set_outputs = transpose(IntToDoubleMatrix([[0, 1, 1, 0]]))
        
        neural_network.train(training_set_inputs, trainingSetOutputs: training_set_outputs, numberOfTrainingIterations: 600)
        let (lev1, lev2, output) = neural_network.think([[1,1]])
        print("Predicted output for input [[1,1]]")
        print(lev1)
        print(lev2)
        print(output)
    }
    
}


