//
//  NeuronLayer.swift
//  NeuralNetwork
//
//  Created by pbardea on 10/28/16.
//  Copyright © 2016 pbardea stdios. All rights reserved.
//

import Foundation

enum ActivationFunction {
    case sigmoid
    case relu
    case tanh
    case leakyRelu
    case eLU
    
    func function(_ x: Double) -> Double {
        switch self {
        case .sigmoid:
            return 1 / (1 + exp(-1 * x))
        case .relu:
            return max(0.0, x)
        case .tanh:
            fatalError("Not implemented")
        case .leakyRelu:
            fatalError("Not implemented")
        case .eLU:
            fatalError("Not implemented")
        }
    }
    
    func derivative(_ x: Double) -> Double {
        switch self {
        case .sigmoid:
            return x * (1 - x)
        case .relu:
            return x > 0 ? 1 : 0
        case .tanh:
            let e = exp(2 * x)
            return (e - 1) / ( e + 1 )
        case .leakyRelu:
            fatalError("Not implemented")
        case .eLU:
            fatalError("Not implemented")
        }
    }
}

protocol NeuronLayer: class {
    // Properties of the neuron layer
    var synapticWeights: Matrix<Double> { get set }
    var activationFunction: ActivationFunction { get }
    
    // Relationship to other layers
    var nextLayer: NeuronLayer? { get set }
    var previousLayer: NeuronLayer? { get set }
    
    // Functions that the layer should be able to do
    func propogate(previousOutputs: Matrix<Double>) -> [Matrix<Double>]
    func backpropogate(layerDepth: Int, outputs: [Matrix<Double>], trainingSetOutputs: Matrix<Double>, withLayerDeltas deltaLayers: [Matrix<Double>]) -> [Matrix<Double>]
}

class FullNeuronLayer: NeuronLayer {

    var synapticWeights: Matrix<Double>
    var activationFunction: ActivationFunction
    
    init(numNeurons: Int, numInputsPerNeuron: Int, activationFunction: ActivationFunction = .sigmoid) {
        synapticWeights = Matrix.getRandomNumMatrixWithHeight(numInputsPerNeuron, byWidth: numNeurons)
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
