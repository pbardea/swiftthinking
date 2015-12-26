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
        assert(result == 26)
    }
    
    func testDotMatrix() {
        let a = [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]
        let b = [[10.0, 11.0, 12.0], [4.0, 6.0, 7.0]]
        let result = dotMatrix(a, withB: b)
        assert(result == [[18.0, 23.0, 26.0], [60.0, 74.0, 83.0], [102.0, 125.0, 140.0]])
    }
    
}

class NNTests: XCTestCase {
    
    func twoLayerTest() {
        let layer1 = NeuronLayer(numNeurons: 4, numInputsPerNeuron: 3)
        let layer2 = NeuronLayer(numNeurons: 1, numInputsPerNeuron: 4)
        
        let neural_network = NeuralNetwork(layer1: layer1, layer2: layer2)
        print("Random starting weights")
        neural_network.printWeight()
        
        let training_set_inputs = IntToDoubleMatrix([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
        let training_set_outputs = transpose(IntToDoubleMatrix([[0, 1, 1, 1, 1, 0]]))
        
        neural_network.train(training_set_inputs, trainingSetOutputs: training_set_outputs, numberOfTrainingIterations: 3)
        let (_, output) = neural_network.think([[1,1,0]])
        print("Predicted output for input [[1,1,0]]")
        
        print(output)
    }
    
}
