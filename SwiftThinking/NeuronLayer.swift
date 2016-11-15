//
//  NeuronLayer.swift
//  NeuralNetwork
//
//  Created by pbardea on 10/28/16.
//  Copyright © 2016 pbardea stdios. All rights reserved.
//

import Foundation


protocol NeuronLayer {
    var synapticWeights: Matrix<Double> { get set }
    
    func propogate(previousOutputs: Matrix<Double>) -> [Matrix<Double>]
}

class FullNeuronLayer: NeuronLayer {
    var synapticWeights: Matrix<Double>
    init(numNeurons: Int, numInputsPerNeuron: Int) {
        synapticWeights = Matrix.getRandomNumMatrixWithHeight(numInputsPerNeuron, byWidth: numNeurons)
    }
    
    let activationFunction: ActivationFunction = .sigmoid
    var nextLayer: NeuronLayer? = nil
    var previousLayer: NeuronLayer? = nil
    
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
    
    func propogate(previousOutputs: Matrix<Double>) -> [Matrix<Double>] {
        let newLayer = (previousOutputs * synapticWeights).apply(function: activationFunction.function)
        
        return [newLayer] + (nextLayer?.propogate(previousOutputs: newLayer) ?? [])
    }
    
    func think(_ inputs: Matrix<Double>) -> [Matrix<Double>] {
        return propogate(previousOutputs: inputs)
    }
    
//    func train() {
//        let trainingSetOutputs = Matrix<Double>()
//        let outputs = [Matrix<Double>]()
//        
//        func backpropogate(withLayerDeltas layerDeltas: [Matrix<Double>] = []) -> [Matrix<Double>] {
//            guard let previousLayer = previousLayer else { return layerDeltas }
//            
//            var layerError: Matrix<Double>
//            if let lastDelta = layerDeltas.last {
//                layerError = lastDelta * self.layers[layerDepth + 1].synapticWeights.transpose
//            } else { // Special case for when generating the first delta (for the last layer)
//                layerError = trainingSetOutputs - outputs[layerDepth]
//            }
//            let layerDelta: Matrix = layerError * outputs[layerDepth].apply(function: activationFunction.derivative)
//
//            return previousLayer.backpropogate(withLayerDeltas: layerDeltas + [layerDelta])
//        }
//    }
    
}

