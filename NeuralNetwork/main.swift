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
    /* Inputs:
     0,0,1 => 0
     0,1,1 => 1
     1,0,1 => 1
     0,1,0 => 1
     1,0,0 => 1
     1,1,1 => 0
     1,1,0 => 0
     */
    
    let trainingSetInputs = Matrix([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]].map {
        Vector($0.map{Double($0)})
    })
    let trainingSetOutputs = intToDoubleMatrix(input: [[0, 1, 1, 1, 1, 0, 0]]).transpose
    
    
    let inputSize = trainingSetInputs[0].count
    
    let layer1width = 5
    let layer2width = 6
    
    let outputSize = 1
    
    let layer1 = NeuronLayer(numNeurons: layer1width, numInputsPerNeuron: inputSize)
    let layer2 = NeuronLayer(numNeurons: layer2width, numInputsPerNeuron: layer1width)
    let layer3 = NeuronLayer(numNeurons: outputSize, numInputsPerNeuron: layer2width)
    
    let neuralNetwork = NeuralNetwork(layers: [layer1, layer2, layer3])
    
    
    neuralNetwork.train(trainingSetInputs, trainingSetOutputs: trainingSetOutputs, numberOfTrainingIterations: 6)
    
    let outputs = neuralNetwork.think(intToDoubleMatrix(input: [[1, 1, 0]]))
    
    print("Predicted output for input [[1, 1, 0]]")
    print(outputs.last) // Outputs about 0.574
}

func test2() {
    /* Inputs:
     $0 xor $1
     0,0,1,0 => 0
     0,1,0,1 => 1
     1,0,0,0 => 1
     1,1,1,1 => 0
     1,1,0,0 => 0
     1,0,1,1 => 1
     */
//    let inputs = [
//        [1,1,1,0,1,0,1],
//        [0,1,1,1,0,0,1],
//        [1,0,1,0,1,0,1],
//        [0,0,0,0,1,1,1],
//        [1,1,1,1,1,1,1],
//        [0,1,0,1,0,1,0],
//        [0,1,0,1,1,1,1],
//        [0,0,0,1,1,1,1],
//        [0,0,0,1,0,1,0],
//        [1,0,0,1,0,1,0]
//    ]
    let inputs: [[Int]] = (0..<256).map { num in
        let maxDigits = 8
        return num.bits(numOfDigits: maxDigits)
    }
    
    let trainingSetInputs = intToDoubleMatrix(input: inputs)
    let trainingSetOutputs = intToDoubleMatrix(input: inputs.map { [$0[0] ^ $0[1]] } ).transpose
    
    let inputSize = trainingSetInputs[0].count
    
    let layer1width = 10
    let layer2width = 17
    let layer3width = 40
    let layer4width = 50
    let layer5width = 30

    
    let outputSize = 1
    
    let layer1 = NeuronLayer(numNeurons: layer1width, numInputsPerNeuron: inputSize)
    let layer2 = NeuronLayer(numNeurons: layer2width, numInputsPerNeuron: layer1width)
    let layer3 = NeuronLayer(numNeurons: layer3width, numInputsPerNeuron: layer2width)
    let layer4 = NeuronLayer(numNeurons: layer4width, numInputsPerNeuron: layer3width)
    let layer5 = NeuronLayer(numNeurons: layer5width, numInputsPerNeuron: layer4width)
    let layer6 = NeuronLayer(numNeurons: outputSize, numInputsPerNeuron: layer5width)
    
    let neuralNetwork = NeuralNetwork(layers: [layer1, layer2, layer3, layer4, layer5, layer6])
    
    print("Start training")
    neuralNetwork.train(trainingSetInputs, trainingSetOutputs: trainingSetOutputs, numberOfTrainingIterations: 20)
    print("End training")
    
    let input = [1,0,0,1,1,1,0,0]
    let outputs = neuralNetwork.think(intToDoubleMatrix(input: [input]))
    print("Predicted output for input [\(input)]")
    let ans = outputs.last?.getRow(row: 0)[0]
    print(ans)
}

test2()

