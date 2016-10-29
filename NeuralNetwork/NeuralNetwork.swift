//
//  NeuralNetwork.swift
//  NeuralNetwork
//
//  Created by pbardea on 10/28/16.
//  Copyright © 2016 pbardea stdios. All rights reserved.
//

import Foundation

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
        for iteration in 0...numberOfTrainingIterations {
            print(iteration)
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
