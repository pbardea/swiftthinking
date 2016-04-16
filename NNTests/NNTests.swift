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
        let result = a.dot(b)
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


    // These 2 tests experiment the effects of overfitting

    func testGeneralNetwork1() { // This test overfits the data and does not perform very well.
        let trainingSetInputs = IntToDoubleMatrix([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]])
        let trainingSetOutputs = transpose(intToDoubleMatrix([[0, 1, 1, 1, 1, 0, 0]]))

        let inputSize = trainingSetInputs[0].count

        let layer1width = 5
        let layer2width = 6

        let outputSize = 1

        let layer1 = NeuronLayer(numNeurons: layer1width, numInputsPerNeuron: inputSize)
        let layer2 = NeuronLayer(numNeurons: layer2width, numInputsPerNeuron: layer1width)
        let layer3 = NeuronLayer(numNeurons: outputSize, numInputsPerNeuron: layer2width)

        let neuralNetwork = NeuralNetwork(layers: [layer1, layer2, layer3])


        neuralNetwork.train(trainingSetInputs, trainingSetOutputs: trainingSetOutputs, numberOfTrainingIterations: 6)

        let outputs = neuralNetwork.think([[1, 1, 0]])
        print("Predicted output for input [[1, 1, 0]]")
        print(outputs.last) // Outputs about 0.574
    }

    func testGeneralNetwork2() { // This test performs better than test 1 (has less data, so doesn't over-fit)
        let trainingSetInputs = IntToDoubleMatrix([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]])
        let trainingSetOutputs = transpose(IntToDoubleMatrix([[0, 1, 1, 0, 0]]))

        let inputSize = trainingSetInputs[0].count

        let layer1width = 5
        let layer2width = 6

        let outputSize = 1

        let layer1 = NeuronLayer(numNeurons: layer1width, numInputsPerNeuron: inputSize)
        let layer2 = NeuronLayer(numNeurons: layer2width, numInputsPerNeuron: layer1width)
        let layer3 = NeuronLayer(numNeurons: outputSize, numInputsPerNeuron: layer2width)

        let neuralNetwork = NeuralNetwork(layers: [layer1, layer2, layer3])


        neuralNetwork.train(trainingSetInputs, trainingSetOutputs: trainingSetOutputs, numberOfTrainingIterations: 6)

        let outputs = neuralNetwork.think([[1, 1, 0]])
        print("Predicted output for input [[1, 1, 0]]")
        print(outputs.last) // outputs around 0.36 - 0.4 (closer to 0)
    }

    func testTwoInput() {
        let inputSize = 2

        let layer1width = 5
        let layer2width = 6

        let outputSize = 1

        let layer1 = NeuronLayer(numNeurons: layer1width, numInputsPerNeuron: inputSize)
        let layer2 = NeuronLayer(numNeurons: layer2width, numInputsPerNeuron: layer1width)
        let layer3 = NeuronLayer(numNeurons: outputSize, numInputsPerNeuron: layer2width)

        let neuralNetwork = ThreeLayerNeuralNetwork(layer1: layer1, layer2: layer2, layer3: layer3)

        let trainingSetInputs = intToDoubleMatrix([[0, 0], [0, 1], [1, 0], [1, 1]])
        let trainingSetOutputs = transpose(IntToDoubleMatrix([[0, 1, 1, 0]]))

        neuralNetwork.train(trainingSetInputs, trainingSetOutputs: trainingSetOutputs, numberOfTrainingIterations: 600)
        let (lev1, lev2, output) = neuralNetwork.think([[1, 1]])
        print("Predicted output for input [[1,1]]")
        print(lev1)
        print(lev2)
        print(output)
    }

}
