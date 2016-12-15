//
//  ConvolutionLayer.swift
//  SwiftThinking
//
//  Created by Paul Bardea on 2016-12-05.
//  Copyright © 2016 pbardea. All rights reserved.
//

//import Accelerate
import Foundation

//// Convolution of a signal [x], with a kernel [k]. The signal must be at least as long as the kernel.
//public func conv(_ x: Matrix<Double>, _ k: Matrix<Double>) -> Matrix<Double> {
//    precondition(x.count >= k.count, "Input vector [x] must have at least as many elements as the kernel,  [k]")
//    
//    let resultSize = x.count + k.count - 1
//    var result = [Double](repeating: 0, count: resultSize)
//    let kEnd = UnsafePointer<Double>(k).advanced(by: k.count - 1)
//    let xPad = repeatElement(Double(0.0), count: k.count-1)
//    let xPadded = xPad + x + xPad
//    vDSP_convD(xPadded, 1, kEnd, -1, &result, 1, vDSP_Length(resultSize), vDSP_Length(k.count))
//    
//    return Matrix(result)
//}

class ConvolutionLayer: NeuronLayer {
    
    // Neuron layer conformance
    var synapticWeights: Matrix<Double>
    var activationFunction: ActivationFunction
    
    var previousLayer: NeuronLayer? = nil
    var nextLayer: NeuronLayer? = nil
    
    let stride: Int
    let padding: Padding
    let depth: Int
    let borderMode: BorderMode
    
    enum Padding {
        case none
        case full
        case half
    }
    
    enum BorderMode {
        case zero
    }
    
    init(numNeurons: Int, numInputsPerNeuron: Int, depth: Int, stride: Int, padding: Padding, activationFunction: ActivationFunction = .sigmoid) {
        self.stride = stride
        self.padding = padding
        self.depth = depth
        synapticWeights = [Matrix<Double>.getRandomNumMatrixWithHeight(numInputsPerNeuron, byWidth: numNeurons)]
        self.activationFunction = activationFunction
    }
    
    func propogate(previousLayer: [Matrix<Double>]) -> [Matrix<Double>] {
        // TODO:
        return [Matrix<Double>()]
    }
    
    func backpropogate(layerDepth: Int, outputs: [Matrix<Double>], trainingSetOutputs: Matrix<Double>, withLayerDeltas deltaLayers: [Matrix<Double>] = []) -> [Matrix<Double>] {
        // TODO:
        return [Matrix<Double>()]
    }
    
}
