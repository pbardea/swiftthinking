//
//  ThreeLayerNeuralNet.swift
//  NeuralNetwork
//
//  Created by pbardea on 10/28/16.
//  Copyright © 2016 pbardea stdios. All rights reserved.
//

import Foundation

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
