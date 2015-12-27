
//
//  main.swift
//  NeuralNetwork
//
//  Created by Paul Bardea on 2015-12-26.
//  Copyright © 2015 pbardea stdios. All rights reserved.
//

import Foundation

typealias Matrix = [[Double]]

func getRandomNumMatrixWithHeight(height: Int, byWidth width: Int) -> Matrix {
    return (0..<height).map { _ in
        (0..<width).map { _ in
            2 * drand48() - 1
        }
    }
}

func dotVector(a: [Double], withB b: [Double]) -> Double {
    var n: Double = 0
    let lim = min(a.count,b.count);
    for i in (0..<lim) { n += a[i] * b[i] }
    return n;
}

func getColumn(i: Int, ofMatrix a: Matrix) -> [Double] {
    return a.map {$0[i]}
}

func getRow(i: Int, ofMatrix a: Matrix) -> [Double] {
    return a[i]
}

func dotMatrix(a: Matrix, withB b: Matrix) -> [[Double]] {
    assert(b.count > 0)
    let result = (0..<a.count).map { i in
        (0..<b[0].count).map { j in
            dotVector(getRow(i, ofMatrix: a), withB: getColumn(j, ofMatrix: b))
        }
    }
    return result
}


func transpose(a: Matrix) -> Matrix {
    assert(a.count > 0)
    let width = a[0].count
    
    return (0..<width).map {i in getColumn(i, ofMatrix: a)}
}

func IntToDoubleMatrix(a: [[Int]]) -> Matrix {
    return a.map { $0.map {x in Double(x)} }
}

func matrixOp(op: (Double, Double)->Double, onA a: Matrix, withB b: Matrix) -> Matrix {
    assert(a.count == b.count)
    if (a.count > 0 && b.count > 0) { assert(a[0].count == b[0].count) }
    return (0..<a.count).map { i in
        (0..<a[i].count).map { j in
            op(a[i][j], b[i][j])
        }
    }
}

func matrixSub(a: Matrix, withB b: Matrix) -> Matrix {
    let result = matrixOp(-, onA: a, withB: b)
    return result
}

func matrixAdd(a: Matrix, withB b: Matrix) -> Matrix {
    let result = matrixOp(+, onA: a, withB: b)
    return result
}

func apply(function: (Double)->Double, toMatrix a: Matrix) -> Matrix {
    return a.map { $0.map (function) }
}


class NeuronLayer {
    var synaptic_weights: Matrix
    init(numNeurons: Int, numInputsPerNeuron: Int) {
        synaptic_weights = getRandomNumMatrixWithHeight(numInputsPerNeuron, byWidth: numNeurons)
    }
}

class NeuralNetwork {
    let layer1: NeuronLayer
    let layer2: NeuronLayer
    let layer3: NeuronLayer
    
    init(layer1: NeuronLayer, layer2: NeuronLayer, layer3: NeuronLayer) {
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
    }
    
    func sigmoid(x: Double) -> Double {
        return 1 / (1 + exp(-1*x))
    }
    
    func sigmoidDerivative(x: Double) -> Double {
        return x * (1 - x)
    }
    
    func think(inputs: Matrix) -> (Matrix, Matrix, Matrix) {
        let outputFromLayer1 = apply(sigmoid, toMatrix: dotMatrix(inputs, withB: self.layer1.synaptic_weights))
        let outputFromLayer2 = apply(sigmoid, toMatrix: dotMatrix(outputFromLayer1, withB: self.layer2.synaptic_weights))
        let outputFromLayer3 = apply(sigmoid, toMatrix: dotMatrix(outputFromLayer2, withB: self.layer3.synaptic_weights))
        
        return (outputFromLayer1, outputFromLayer2, outputFromLayer3)
    }
    
    func train(trainingSetInputs: Matrix, trainingSetOutputs: Matrix, numberOfTrainingIterations: Int) -> Void {
        for _ in 0...numberOfTrainingIterations {
            let outputFromLayer1, outputFromLayer2, outputFromLayer3: Matrix
            (outputFromLayer1, outputFromLayer2, outputFromLayer3) = self.think(trainingSetInputs)
            
            let layer3error: Matrix = matrixSub(trainingSetOutputs, withB: outputFromLayer3)
            let layer3delta: Matrix = dotMatrix(layer3error, withB: apply(sigmoidDerivative, toMatrix: outputFromLayer3))
            
            let layer2error: Matrix = dotMatrix(layer3delta, withB: transpose(self.layer3.synaptic_weights))
            let layer2delta: Matrix = dotMatrix(layer2error, withB: apply(sigmoidDerivative, toMatrix: outputFromLayer2))
            
            let layer1error: Matrix = dotMatrix(layer2delta, withB: transpose(self.layer2.synaptic_weights))
            let layer1delta: Matrix = dotMatrix(layer1error, withB: apply(sigmoidDerivative, toMatrix: outputFromLayer1))
            
            let layer1adjustmnets: Matrix = dotMatrix(transpose(trainingSetInputs), withB: layer1delta)
            let layer2adjustments: Matrix = dotMatrix(transpose(outputFromLayer1), withB: layer2delta)
            let layer3adjustments: Matrix = dotMatrix(transpose(outputFromLayer2), withB: layer3delta)
            
            self.layer1.synaptic_weights = matrixAdd(self.layer1.synaptic_weights, withB: layer1adjustmnets)
            self.layer2.synaptic_weights = matrixAdd(self.layer2.synaptic_weights, withB: layer2adjustments)
            self.layer3.synaptic_weights = matrixAdd(self.layer3.synaptic_weights, withB: layer3adjustments)
        }
    }
    
    
    func printWeight() -> Void {
        print("\tLayer 1")
        print(layer1.synaptic_weights)
        print("\tLayer 2")
        print(layer2.synaptic_weights)
    }
}
