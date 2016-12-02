//
//  ActivationFunction.swift
//  SwiftThinking
//
//  Created by Paul Bardea on 2016-12-02.
//  Copyright © 2016 pbardea. All rights reserved.
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
