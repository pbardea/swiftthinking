//
//  LinearAlgebra.swift
//  NeuralNetwork
//
//  Created by Paul Bardea on 2015-12-27.
//  Copyright © 2015 pbardea stdios. All rights reserved.
//

import Foundation

typealias Matrix = [[Double]]
typealias Vector = [Double]

func dot(a: Vector, withB b: Vector) -> Double {
    var n: Double = 0
    let lim = min(a.count, b.count)
    for i in (0..<lim) {
        n += a[i] * b[i]
    }
    return n
}

func transpose(a: Matrix) -> Matrix {
    assert(a.count > 0)
    let width = a[0].count

    return (0..<width).map {i in getColumn(i, ofMatrix: a)}
}

func getColumn(i: Int, ofMatrix a: Matrix) -> Vector {
    return a.map {$0[i]}
}

func getRow(i: Int, ofMatrix a: Matrix) -> Vector {
    return a[i]
}



func dotMatrix(a: Matrix, withB b: Matrix) -> [[Double]] {
    assert(b.count > 0)
    return (0..<a.count).map { i in
        (0..<b[0].count).map { j in
            dot(getRow(i, ofMatrix: a), withB: getColumn(j, ofMatrix: b))
        }
    }
}

func intToDoulbeMatrix(a: [[Int]]) -> Matrix {
    return a.map { $0.map {x in Double(x)} }
}

func matrixOp(op: (Double, Double) -> Double, onA a: Matrix, withB b: Matrix) -> Matrix {
    assert(a.count == b.count)
    if a.count > 0 && b.count > 0 {
        assert(a[0].count == b[0].count)
    }
    return (0..<a.count).map { i in
        (0..<a[i].count).map { j in
            op(a[i][j], b[i][j])
        }
    }
}

func matrixSub(a: Matrix, withB b: Matrix) -> Matrix {
    return matrixOp(-, onA: a, withB: b)
}

func matrixAdd(a: Matrix, withB b: Matrix) -> Matrix {
    return matrixOp(+, onA: a, withB: b)
}

func apply(function: (Double) -> Double, toMatrix a: Matrix) -> Matrix {
    return a.map { $0.map (function) }
}
