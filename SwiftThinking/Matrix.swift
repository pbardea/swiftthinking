//
//  Matrix.swift
//  Swector
//
//  Created by Paul Bardea on 2016-05-10.
//  Copyright © 2016 pbardea. All rights reserved.
//

import Foundation

public struct Matrix<T: Numeric> {
    public typealias Element = T

    fileprivate var data: [Vector<Element>]
    var size: (Int, Int) {
        return (self.data.count, self.data.first?.size ?? 0)
    }

    public subscript(index: Int) -> Vector<Element> {
        assert(index < self.data.count && index >= 0)
        return data[index]
    }
    
    public init(_ data: [Vector<Element>] = []) {
        self.data = data
    }
    
    public init(_ data: [[Element]] = []) {
        self.data = data.map { d in Vector(d) }
    }
}

public extension Matrix {
    static func doubleMatrixFrom(input: [[Int]]) -> Matrix<Element> {
        let vectors = input.map {
            Vector($0.map { Element($0) })
        }
        return Matrix(vectors)
    }
}

extension Matrix: Sequence {
    public typealias Generator = AnyIterator<Vector<T>>

    public func generate() -> Generator {
        var index = 0
        return AnyIterator {
            if index < self.data.count {
                let d = self.data[index]
                index += 1
                return d
            }
            return nil
        }
    }

}


public extension Matrix {

    func getRow(row: Int) -> Vector<Element> {
        return self.data[row]
    }

    func getColumn(column: Int) -> Vector<Element> {
        return Vector(self.data.map { $0[column] })
    }

}

extension Matrix: Collection {
    /// Returns the position immediately after the given index.
    ///
    /// - Parameter i: A valid index of the collection. `i` must be less than
    ///   `endIndex`.
    /// - Returns: The index value immediately after `i`.
    public func index(after i: Int) -> Int {
        guard i < endIndex else { fatalError() }
        return i + 1
    }

    public typealias Index = Int
    public typealias _Element = Vector<T>

    public subscript(index: Int) -> Element? {
        let (_, width) = self.size

        if width > 0 {
            let x = index % width
            let y = index / width

            return self.data[y][x]
        }
        return nil
    }

    public var startIndex: Int { return 0 }
    public var endIndex: Int {
        let (h, w) = self.size
        return h * w - 1
    }
}

private extension Matrix {

    func mult(m: Matrix) -> Matrix {
        assert(self.data.count > 0 && m.data.count > 0)
        let newData = (0..<self.data.count).map { i in
            (0..<m.data[0].size).map { j in
                self.getRow(row: i) ** m.getColumn(column: j)
            }
        }
        return Matrix(newData)
    }

    func add(m: Matrix) -> Matrix {
        return Matrix(zip(self.data, m.data).map { $0 + $1 })
    }

    func subtract(m: Matrix) -> Matrix {
        return Matrix(zip(self.data, m.data).map { $0 - $1 })
    }

    func isSameSizeAs(m: Matrix) -> Bool {
        return self.size == m.size
    }

}

public extension Matrix {
    func apply(function: @escaping (Element) -> Element) -> Matrix<Element> {
        let newData = data.map { $0.apply(function: function) }
        return Matrix(newData)
    }
    
    var transpose: Matrix<Element> {
        let width = data[0].size
        let newData = (0..<width).map {i in self.getColumn(column: i)}
        return Matrix(newData)
    }
}

// Operators
public func + <T: Numeric>(lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
    return lhs.add(m: rhs)
}

public func - <T: Numeric>(lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
    return lhs.subtract(m: rhs)
}

public func * <T: Numeric>(lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
    return lhs.mult(m: rhs)
}

// Debugging
extension Matrix: CustomStringConvertible {
    public var description: String {
        return self.data.reduce("") {
            "\($0)\($1)"
        }
    }
}
