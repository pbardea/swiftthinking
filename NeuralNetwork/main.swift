//
//  main.swift
//  NeuralNetwork
//
//  Created by Paul Bardea on 2015-12-26.
//  Copyright © 2015 pbardea stdios. All rights reserved.
//

import Foundation


func getRandomNumMatrixWithHeight(_ height: Int, byWidth width: Int) -> Matrix {
    return (0..<height).map { _ in
        (0..<width).map { _ in
            2 * drand48() - 1
        }
    }
}

class NeuronLayer {
    var synapticWeights: Matrix
    init(numNeurons: Int, numInputsPerNeuron: Int) {
        synapticWeights = getRandomNumMatrixWithHeight(numInputsPerNeuron, byWidth: numNeurons)
    }
}

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

    func synapticWeightAtIndex(_ index: Int) -> Matrix? {
        let synapticWeights = self.layers.map { $0.synapticWeights } // TODO: DRY up this code

        return synapticWeights.indices ~= index ? synapticWeights[index] : nil
    }

    func think(_ inputs: Matrix) -> [Matrix] {
        // Array of the synaptic weights
        let synapticWeights = self.layers.map { $0.synapticWeights }

        func recur(_ outputFromPreviousLayer: Matrix, withAccumulator accumulator: [Matrix], andSynapticDepth synapticDepth: Int) -> [Matrix] {
            if let lastOut = accumulator.last, let synapticLayer = synapticWeightAtIndex(synapticDepth) {
                // Apply the weights of each synapsis to the last layer of input
                let z = dotMatrix(lastOut, withB: synapticLayer)
                // Apply the sigmoid function (each neuron)
                let output = apply(sigmoid, toMatrix: z)

                // Recursively apply to the rest of the layers
                return [output] + recur(output, withAccumulator: accumulator + [output], andSynapticDepth: synapticDepth + 1)
            } else {
                return []
            }
        }


        return recur(inputs, withAccumulator: [inputs], andSynapticDepth: 0)
    }

    // Trains using back-propogation
    func train(_ trainingSetInputs: Matrix, trainingSetOutputs: Matrix, numberOfTrainingIterations: Int) -> Void {
        // Purpose is to minimize the cost function
        for _ in 0...numberOfTrainingIterations {
            let outputs = self.think(trainingSetInputs)

            func recur(_ layerDepth: Int, withAccumulator accumulator: [Matrix]) -> [Matrix] {
                if layerDepth >= 0 { // starts at self.layers.count - 1
                    var layerNError: Matrix
                    if layerDepth == self.layers.count - 1 {
                        layerNError = matrixSub(([trainingSetOutputs] + accumulator).last!, withB: outputs[layerDepth])
                    } else {
                        layerNError = dotMatrix(([trainingSetOutputs] + accumulator).last!, withB: transpose(self.layers[layerDepth + 1].synapticWeights))
                    }
                    let layerNDelta: Matrix = dotMatrix(layerNError, withB: apply(sigmoidDerivative, toMatrix: outputs[layerDepth]))

                    return recur(layerDepth-1, withAccumulator: accumulator + [layerNDelta])
                }
                return accumulator
            }

            let layerDeltas = recur(self.layers.count - 1, withAccumulator: [])

            let layerAdjustments = (0..<layerDeltas.count).map { i in
                dotMatrix(transpose(([trainingSetInputs]+outputs)[i]), withB: layerDeltas[self.layers.count - 1 - i])
            }
            for (index, layer) in self.layers.enumerated() {
                layer.synapticWeights = matrixAdd(layer.synapticWeights, withB: layerAdjustments[index])
            }
        }
    }

}

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

    func think(_ inputs: Matrix) -> (Matrix, Matrix, Matrix) {
        let outputFromLayer1 = apply(sigmoid, toMatrix: dotMatrix(inputs, withB: self.layer1.synapticWeights))
        let outputFromLayer2 = apply(sigmoid, toMatrix: dotMatrix(outputFromLayer1, withB: self.layer2.synapticWeights))
        let outputFromLayer3 = apply(sigmoid, toMatrix: dotMatrix(outputFromLayer2, withB: self.layer3.synapticWeights))

        return (outputFromLayer1, outputFromLayer2, outputFromLayer3)
    }

    func train(_ trainingSetInputs: Matrix, trainingSetOutputs: Matrix, numberOfTrainingIterations: Int) -> Void {
        for _ in 0...numberOfTrainingIterations {
            let (outputFromLayer1, outputFromLayer2, outputFromLayer3) = self.think(trainingSetInputs)

            let layer3error: Matrix = matrixSub(trainingSetOutputs, withB: outputFromLayer3)
            let layer3delta: Matrix = dotMatrix(layer3error, withB: apply(sigmoidDerivative, toMatrix: outputFromLayer3))

            let layer2error: Matrix = dotMatrix(layer3delta, withB: transpose(self.layer3.synapticWeights))
            let layer2delta: Matrix = dotMatrix(layer2error, withB: apply(sigmoidDerivative, toMatrix: outputFromLayer2))

            let layer1error: Matrix = dotMatrix(layer2delta, withB: transpose(self.layer2.synapticWeights))
            let layer1delta: Matrix = dotMatrix(layer1error, withB: apply(sigmoidDerivative, toMatrix: outputFromLayer1))

            let layer1adjustmnets: Matrix = dotMatrix(transpose(trainingSetInputs), withB: layer1delta)
            let layer2adjustments: Matrix = dotMatrix(transpose(outputFromLayer1), withB: layer2delta)
            let layer3adjustments: Matrix = dotMatrix(transpose(outputFromLayer2), withB: layer3delta)

            self.layer1.synapticWeights = matrixAdd(self.layer1.synapticWeights, withB: layer1adjustmnets)
            self.layer2.synapticWeights = matrixAdd(self.layer2.synapticWeights, withB: layer2adjustments)
            self.layer3.synapticWeights = matrixAdd(self.layer3.synapticWeights, withB: layer3adjustments)
        }
    }

    func printWeight() -> Void {
        print("\tLayer 1")
        print(layer1.synapticWeights)
        print("\tLayer 2")
        print(layer2.synapticWeights)
    }

}
