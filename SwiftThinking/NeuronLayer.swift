//
//  NeuronLayer.swift
//  NeuralNetwork
//
//  Created by pbardea on 10/28/16.
//  Copyright © 2016 pbardea stdios. All rights reserved.
//

import Foundation

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
