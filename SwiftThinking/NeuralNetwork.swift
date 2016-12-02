//
//  NeuralNetwork.swift
//  NeuralNetwork
//
//  Created by pbardea on 10/28/16.
//  Copyright © 2016 pbardea stdios. All rights reserved.
//

import Foundation

class NeuralNetwork {
    var layers: [NeuronLayer] = []
    let alpha = 1.0
    
    init(layers: [NeuronLayer]) {
        layers.forEach { layer in
            self.addLayer(layer: layer)
        }
    }
    
    private func addLayer(layer: NeuronLayer) {
        if let previousLayer = layers.last {
            previousLayer.nextLayer = layer
            layer.previousLayer = previousLayer
        }
        self.layers.append(layer)
    }
    
    func synapticWeightAtIndex(_ index: Int) -> Matrix<Double>? {
        let synapticWeights = self.layers.map { $0.synapticWeights }

        return synapticWeights.indices ~= index ? synapticWeights[index] : nil
    }

    func think(_ inputs: Matrix<Double>) -> [Matrix<Double>] {
        guard let firstLayer = layers.first else { return [] }
        return firstLayer.propogate(previousOutputs: inputs)
    }

    // Trains using back-propogation
    func train(_ trainingSetInputs: Matrix<Double>, trainingSetOutputs: Matrix<Double>, numberOfTrainingIterations: Int, verbose: Bool = false) -> Void {

        // Purpose is to minimize the cost function
        (0...numberOfTrainingIterations).forEach { iteration in
            let outputs = self.think(trainingSetInputs)
            
            // Backpropogate from the last layer
            guard let lastLayer = self.layers.last else { return }
                
            let layerDeltas = lastLayer.backpropogate(layerDepth: self.layers.count - 1, outputs: outputs, trainingSetOutputs: trainingSetOutputs, withLayerDeltas: [])

            let layerAdjustments = (0..<layerDeltas.count).map { i in
                ([trainingSetInputs]+outputs)[i].transpose * layerDeltas[self.layers.count - 1 - i]
            }
            var index = 0
            for layer in layers {
                layer.synapticWeights = layer.synapticWeights + layerAdjustments[index].apply { $0 * self.alpha }
                index += 1
            }
        }
    }
}
