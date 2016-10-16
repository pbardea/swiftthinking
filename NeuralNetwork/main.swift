//
//  main.swift
//  NeuralNetwork
//
//  Created by Paul Bardea on 2015-12-26.
//  Copyright © 2015 pbardea stdios. All rights reserved.
//

import Foundation
@testable import NeuralNetwork


func getRandomNumMatrixWithHeight(_ height: Int, byWidth width: Int) -> Matrix<Double> {
    let data = (0..<height).map { _ in
        (0..<width).map { _ in
            2 * drand48() - 1
        }
    }
    let vectors = data.map(Vector.init)
    return Matrix(vectors)
}

class NeuronLayer {
    var synapticWeights: Matrix<Double>
    init(numNeurons: Int, numInputsPerNeuron: Int) {
        synapticWeights = getRandomNumMatrixWithHeight(numInputsPerNeuron, byWidth: numNeurons)
    }
}

class NeuralNetwork {
    let layers: [NeuronLayer]

    init(layers: [NeuronLayer]) {
        self.layers = layers
    }

    func sigmoid(_ x: Double) -> Double {
        return 1 / (1 + exp(-1*x))
    }

    func sigmoidDerivative(_ x: Double) -> Double {
        return x * (1 - x)
    }

    func synapticWeightAtIndex(_ index: Int) -> Matrix<Double>? {
        let synapticWeights = self.layers.map { $0.synapticWeights } // TODO: DRY up this code

        return synapticWeights.indices ~= index ? synapticWeights[index] : nil
    }

    func think(_ inputs: Matrix<Double>) -> [Matrix<Double>] {
        // Array of the synaptic weights
        let synapticWeights = self.layers.map { $0.synapticWeights }

        func recur(_ outputFromPreviousLayer: Matrix<Double>, withAccumulator accumulator: [Matrix<Double>], andSynapticDepth synapticDepth: Int) -> [Matrix<Double>] {
            if let lastOut = accumulator.last, let synapticLayer = synapticWeightAtIndex(synapticDepth) {
                // Apply the weights of each synapsis to the last layer of input
                let z = lastOut * synapticLayer //dotMatrix(lastOut, withB: synapticLayer)
                // Apply the sigmoid function (each neuron)
                let output = z.apply(function: sigmoid) //apply(sigmoid, toMatrix: z)

                // Recursively apply to the rest of the layers
                return [output] + recur(output, withAccumulator: accumulator + [output], andSynapticDepth: synapticDepth + 1)
            } else {
                return []
            }
        }


        return recur(inputs, withAccumulator: [inputs], andSynapticDepth: 0)
    }

    // Trains using back-propogation
    func train(_ trainingSetInputs: Matrix<Double>, trainingSetOutputs: Matrix<Double>, numberOfTrainingIterations: Int) -> Void {
        // Purpose is to minimize the cost function
        for _ in 0...numberOfTrainingIterations {
            let outputs = self.think(trainingSetInputs)

            func recur(_ layerDepth: Int, withAccumulator accumulator: [Matrix<Double>]) -> [Matrix<Double>] {
                if layerDepth >= 0 { // starts at self.layers.count - 1
                    var layerNError: Matrix<Double>
                    if layerDepth == self.layers.count - 1 {
                        layerNError = ([trainingSetOutputs] + accumulator).last! - outputs[layerDepth]
                    } else {
                        layerNError = ([trainingSetOutputs] + accumulator).last! * self.layers[layerDepth + 1].synapticWeights.transpose
                    }
                    let layerNDelta: Matrix = layerNError * outputs[layerDepth].apply(function: sigmoidDerivative)

                    return recur(layerDepth-1, withAccumulator: accumulator + [layerNDelta])
                }
                return accumulator
            }

            let layerDeltas = recur(self.layers.count - 1, withAccumulator: [])

            let layerAdjustments = (0..<layerDeltas.count).map { i in
                ([trainingSetInputs]+outputs)[i].transpose * layerDeltas[self.layers.count - 1 - i]
            }
            for (index, layer) in self.layers.enumerated() {
                layer.synapticWeights = layer.synapticWeights + layerAdjustments[index]
            }
        }
    }

}

class ThreeLayerNeuralNetwork {
    let layer1: NeuronLayer
    let layer2: NeuronLayer
    let layer3: NeuronLayer

    init(layer1: NeuronLayer, layer2: NeuronLayer, layer3: NeuronLayer) {
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
    }

    func sigmoid(_ x: Double) -> Double {
        return 1 / (1 + exp(-1*x))
    }

    func sigmoidDerivative(_ x: Double) -> Double {
        return x * (1 - x)
    }

    func think(_ inputs: Matrix<Double>) -> (Matrix<Double>, Matrix<Double>, Matrix<Double>) {
        let outputFromLayer1 = (inputs * self.layer1.synapticWeights).apply(function: sigmoid)
        let outputFromLayer2 = (outputFromLayer1 * self.layer2.synapticWeights).apply(function: sigmoid)
        let outputFromLayer3 = (outputFromLayer2 * self.layer3.synapticWeights).apply(function: sigmoid)

        return (outputFromLayer1, outputFromLayer2, outputFromLayer3)
    }

    func train(_ trainingSetInputs: Matrix<Double>, trainingSetOutputs: Matrix<Double>, numberOfTrainingIterations: Int) -> Void {
        for _ in 0...numberOfTrainingIterations {
            let (outputFromLayer1, outputFromLayer2, outputFromLayer3) = self.think(trainingSetInputs)

            let layer3error: Matrix = trainingSetOutputs - outputFromLayer3
            let layer3delta: Matrix = outputFromLayer3.apply(function: sigmoidDerivative) * layer3error

            let layer2error: Matrix = layer3delta * self.layer3.synapticWeights.transpose
            let layer2delta: Matrix = layer2error * (outputFromLayer2.apply(function: sigmoidDerivative))

            let layer1error: Matrix = layer2delta * self.layer2.synapticWeights.transpose //dotMatrix(layer2delta, withB: transpose(self.layer2.synapticWeights))
            let layer1delta: Matrix = layer1error * outputFromLayer1.apply(function: sigmoidDerivative)

            let layer1adjustmnets: Matrix = trainingSetInputs.transpose * layer1delta
            let layer2adjustments: Matrix = outputFromLayer1.transpose * layer2delta
            let layer3adjustments: Matrix = outputFromLayer2.transpose * layer3delta

            self.layer1.synapticWeights = self.layer1.synapticWeights + layer1adjustmnets
            self.layer2.synapticWeights = self.layer2.synapticWeights + layer2adjustments
            self.layer3.synapticWeights = self.layer3.synapticWeights + layer3adjustments
        }
    }

    func printuWeight() -> Void {
        print("\tLayer 1")
        print(layer1.synapticWeights)
        print("\tLayer 2")
        print(layer2.synapticWeights)
    }

}

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
    
    let trainingSetInputs = intToDoubleMatrix(input: [[0, 0, 1,0], [0, 1, 0,1], [1, 0, 0,0], [1, 1, 1,1], [1, 1, 0,0], [1,0,1,1]])
    let trainingSetOutputs = intToDoubleMatrix(input: [[0, 1, 1, 0, 0,1]]).transpose
    
    let inputSize = trainingSetInputs[0].count
    
    let layer1width = 6
    let layer2width = 7
    
    let outputSize = 1
    
    let layer1 = NeuronLayer(numNeurons: layer1width, numInputsPerNeuron: inputSize)
    let layer2 = NeuronLayer(numNeurons: layer2width, numInputsPerNeuron: layer1width)
    let layer3 = NeuronLayer(numNeurons: outputSize, numInputsPerNeuron: layer2width)
    
    let neuralNetwork = NeuralNetwork(layers: [layer1, layer2, layer3])
    
    
    neuralNetwork.train(trainingSetInputs, trainingSetOutputs: trainingSetOutputs, numberOfTrainingIterations: 6)
    
    let input = [1,1,0,1]
    let outputs = neuralNetwork.think(intToDoubleMatrix(input: [input]))
    print("Predicted output for input [\(input)]")
    let ans = outputs.last?.getRow(row: 0)[0]
    print(ans)
    print(Int(ans!)) // outputs around 0.36 - 0.4 (closer to 0)
}

test1()
test2()
