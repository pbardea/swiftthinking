//
//  NeuronLayer.swift
//  NeuralNetwork
//
//  Created by pbardea on 10/28/16.
//  Copyright © 2016 pbardea stdios. All rights reserved.
//

import Foundation

class NeuronLayer {
    var synapticWeights: Matrix<Double>
    init(numNeurons: Int, numInputsPerNeuron: Int) {
        synapticWeights = Matrix.getRandomNumMatrixWithHeight(numInputsPerNeuron, byWidth: numNeurons)
    }
}
