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
            guard let lastOut = accumulator.last, let synapticLayer = synapticWeightAtIndex(synapticDepth) else { return [] }
            
            // Apply the weights of each synapsis to the last layer of input
            let z = lastOut * synapticLayer //dotMatrix(lastOut, withB: synapticLayer)
            // Apply the sigmoid function (each neuron)
            let output = z.apply(function: sigmoid) //apply(sigmoid, toMatrix: z)

            // Recursively apply to the rest of the layers
            return [output] + recur(output, withAccumulator: accumulator + [output], andSynapticDepth: synapticDepth + 1)
        }


        return recur(inputs, withAccumulator: [inputs], andSynapticDepth: 0)
    }

    // Trains using back-propogation
    func train(_ trainingSetInputs: Matrix<Double>, trainingSetOutputs: Matrix<Double>, numberOfTrainingIterations: Int, verbose: Bool = false) -> Void {
        let outputs = self.think(trainingSetInputs)
        
        func backpropogate(_ layerDepth: Int = self.layers.count - 1, withCorrectedLayers layers: [Matrix<Double>]) -> [Matrix<Double>] {
            guard layerDepth >= 0 else { return layers }
            
            var layerError: Matrix<Double>
            if layerDepth == self.layers.count - 1 {
                layerError = ([trainingSetOutputs] + layers).last! - outputs[layerDepth]
            } else {
                layerError = ([trainingSetOutputs] + layers).last! * self.layers[layerDepth + 1].synapticWeights.transpose
            }
            let layerDelta: Matrix = layerError * outputs[layerDepth].apply(function: sigmoidDerivative)

            return backpropogate(layerDepth-1, withCorrectedLayers: layers + [layerDelta])
        }

        // Purpose is to minimize the cost function
        for _ in 0...numberOfTrainingIterations {
            let layerDeltas = backpropogate(withCorrectedLayers: [])

            let layerAdjustments = (0..<layerDeltas.count).map { i in
                ([trainingSetInputs]+outputs)[i].transpose * layerDeltas[self.layers.count - 1 - i]
            }
            self.layers.enumerated().forEach { (index, layer) in 
                layer.synapticWeights = layer.synapticWeights + layerAdjustments[index]
            }
        }
    }
}
