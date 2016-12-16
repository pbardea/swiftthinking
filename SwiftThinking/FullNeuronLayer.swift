//
//  FullNeuronLayer.swift
//  SwiftThinking
//
//  Created by Paul Bardea on 2016-12-02.
//  Copyright © 2016 pbardea. All rights reserved.
//

import Foundation

class FullNeuronLayer: NeuronLayer {

    var synapticWeights: Matrix<Double>
    var activationFunction: ActivationFunction
    
    init(numNeurons: Int, numInputsPerNeuron: Int, activationFunction: ActivationFunction = .sigmoid) {
        synapticWeights = Matrix<Double>.getRandomNumMatrixWithHeight(numInputsPerNeuron, byWidth: numNeurons)
        self.activationFunction = activationFunction
    }
    
    var previousLayer: NeuronLayer? = nil
    var nextLayer: NeuronLayer? = nil
    
    func propogate(previousOutputs: Matrix<Double>) -> [Matrix<Double>] {
        let newLayer = (previousOutputs * synapticWeights).apply(function: activationFunction.function)
        
        return [newLayer] + (nextLayer?.propogate(previousOutputs: newLayer) ?? [])
    }
    
    /**
     With default arugments:
        Returns the layer deltas for all layers
     
     With specified arguments:
        Returns the layer deltas including and after the specified `layerDepth`
    */
    func backpropogate(layerDepth: Int, outputs: [Matrix<Double>], trainingSetOutputs: Matrix<Double>, withLayerDeltas deltaLayers: [Matrix<Double>] = []) -> [Matrix<Double>] {
        var layerError: Matrix<Double>
        if let nextLayer = self.nextLayer, let lastDelta = deltaLayers.last {
            layerError = lastDelta * nextLayer.synapticWeights.transpose
        } else {
            layerError = trainingSetOutputs - outputs[layerDepth]
        }
        let a: Matrix = outputs[layerDepth].apply(function: self.activationFunction.derivative)
        let layerDelta: Matrix = layerError * a
        
        let newDeltaLayers = deltaLayers + [layerDelta]

        return self.previousLayer?.backpropogate(layerDepth: layerDepth - 1, outputs: outputs, trainingSetOutputs: trainingSetOutputs, withLayerDeltas: newDeltaLayers) ?? newDeltaLayers
    }
    
}
