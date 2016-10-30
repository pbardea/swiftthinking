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
    let alpha = 1.0

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
        let synapticWeights = self.layers.map { $0.synapticWeights }

        return synapticWeights.indices ~= index ? synapticWeights[index] : nil
    }

    func think(_ inputs: Matrix<Double>) -> [Matrix<Double>] {
        func propogate(_ outputFromPreviousLayer: Matrix<Double>, withLayers layers: [Matrix<Double>], atSynapticDepth synapticDepth: Int = 0) -> [Matrix<Double>] {
            guard
                let lastOut = layers.last,
                let synapticLayer = synapticWeightAtIndex(synapticDepth)
            else {
                return []
            }
            
            // Apply the weights of each synapsis to the last layer of input
            let nextLayer = (lastOut * synapticLayer).apply(function: sigmoid)

            // Recursively apply to the rest of the layers
            return [nextLayer] + propogate(nextLayer, withLayers: layers + [nextLayer], atSynapticDepth: synapticDepth + 1)
        }

        return propogate(inputs, withLayers: [inputs])
    }

    // Trains using back-propogation
    func train(_ trainingSetInputs: Matrix<Double>, trainingSetOutputs: Matrix<Double>, numberOfTrainingIterations: Int, verbose: Bool = false) -> Void {

        // Purpose is to minimize the cost function
        (0...numberOfTrainingIterations).forEach { iteration in
            let outputs = self.think(trainingSetInputs)
            
            /**
             With default arugments:
                Returns the layer deltas for all layers
             
             With specified arguments:
                Returns the layer deltas including and after the specified `layerDepth`
            */
            func backpropogate(_ layerDepth: Int, withLayerDeltas deltaLayers: [Matrix<Double>] = []) -> [Matrix<Double>] {
                guard layerDepth >= 0 else { return deltaLayers }
                
                var layerError: Matrix<Double>
                if let lastDelta = deltaLayers.last {
                    layerError = lastDelta * self.layers[layerDepth + 1].synapticWeights.transpose
                } else { // Special case for when generating the first delta (for the last layer)
                    layerError = trainingSetOutputs - outputs[layerDepth]
                }
                let layerDelta: Matrix = layerError * outputs[layerDepth].apply(function: sigmoidDerivative)

                return backpropogate(layerDepth-1, withLayerDeltas: deltaLayers + [layerDelta])
            }
        
            // Backpropogate from the last layer
            let layerDeltas = backpropogate(self.layers.count - 1)

            let layerAdjustments = (0..<layerDeltas.count).map { i in
                ([trainingSetInputs]+outputs)[i].transpose * layerDeltas[self.layers.count - 1 - i]
            }
            self.layers.enumerated().forEach { (index, layer) in 
                layer.synapticWeights = layer.synapticWeights + layerAdjustments[index].apply() {
                    $0 * self.alpha
                }
            }
        }
    }
}
