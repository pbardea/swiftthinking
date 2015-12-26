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

func linAlgTest() -> Void {
    func testDotVector() -> Bool {
        let a = [1.0, 2.0, 3.0]
        let b = [3.0, 4.0, 5.0]
        let result = dotVector(a, withB: b)
        return result == 26
    }
    
    func testDotMatrix() -> Bool {
        let a = [[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]]
        let b = [[10.0, 11.0, 12.0], [4.0, 6.0, 7.0]]
        let result = dotMatrix(a, withB: b)
        return result == [[18.0, 23.0, 26.0], [60.0, 74.0, 83.0], [102.0, 125.0, 140.0]]
    }
    
    assert(testDotMatrix())
    assert(testDotVector())
}

class NeuralNetwork {
    let layer1: NeuronLayer
    let layer2: NeuronLayer
    
    init(layer1: NeuronLayer, layer2: NeuronLayer) {
        self.layer1 = layer1
        self.layer2 = layer2
    }
    
    func sigmoid(x: Double) -> Double {
        return 1 / (1 - exp(-1*x))
    }
    
    func sigmoidDerivative(x: Double) -> Double {
        return x * (1-x)
    }
    
    func think(inputs: Matrix) -> (Matrix, Matrix) {
        let outputFromLayer1 = apply(sigmoid, toMatrix: dotMatrix(inputs, withB: self.layer1.synaptic_weights))
        let outputFromLayer2 = apply(sigmoid, toMatrix: dotMatrix(outputFromLayer1, withB: self.layer2.synaptic_weights))
        return (outputFromLayer1, outputFromLayer2)
    }
    
    func train(trainingSetInputs: Matrix, trainingSetOutputs: Matrix, numberOfTrainingIterations: Int) -> Void {
        for _ in 0...numberOfTrainingIterations {
            let outputFromLayer1, outputFromLayer2: Matrix
            (outputFromLayer1, outputFromLayer2) = self.think(trainingSetInputs)
            
            let layer2error: Matrix = matrixSub(trainingSetOutputs, withB: outputFromLayer2)
            let layer2delta: Matrix = dotMatrix(layer2error, withB: apply(sigmoidDerivative, toMatrix: outputFromLayer2))
            
            let layer1error: Matrix = dotMatrix(layer2delta, withB: transpose(self.layer2.synaptic_weights))
            let layer1delta: Matrix = dotMatrix(layer1error, withB: apply(sigmoidDerivative, toMatrix: outputFromLayer1))
            
            let layer1adjustmnets: Matrix = dotMatrix(transpose(trainingSetInputs), withB: layer1delta)
            let layer2adjustments: Matrix = dotMatrix(transpose(outputFromLayer1), withB: layer2delta)
            
            self.layer1.synaptic_weights = matrixAdd(self.layer1.synaptic_weights, withB: layer1adjustmnets)
            self.layer2.synaptic_weights = matrixAdd(self.layer2.synaptic_weights, withB: layer2adjustments)
        }
    }
    
    
    func printWeight() -> Void {
        print("\tLayer 1")
        print(layer1.synaptic_weights)
        print("\tLayer 2")
        print(layer2.synaptic_weights)
    }
}

func main() {
    
}


main()