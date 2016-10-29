//
//  Matrix.swift
//  Swector
//
//  Created by Paul Bardea on 2016-05-10.
//  Copyright © 2016 pbardea. All rights reserved.
//

import Foundation

struct Matrix<T: Numeric> {
    typealias Element = T

    fileprivate var data: [Vector<Element>]
    var size: (Int, Int) {
        return (self.data.count, self.data.first?.size ?? 0)
    }

    subscript(index: Int) -> Vector<Element> {
        assert(index < self.data.count && index >= 0)
        return data[index]
    }

    init(_ data: [Vector<Element>] = []) {
        self.data = data
    }

    init(_ data: [[Element]] = []) {
        self.data = data.map { d in Vector(d) }
    }
}

extension Matrix: Sequence {
    typealias Generator = AnyIterator<Vector<T>>

    func generate() -> Generator {
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

extension Matrix {

    func getRow(row: Int) -> Vector<Element> {
        return self.data[row]
    }

    func getColumn(column: Int) -> Vector<Element> {
        return Vector(self.data.map { v in
            v[column]
        })
    }

}

extension Matrix: Indexable, Collection {
    /// Returns the position immediately after the given index.
    ///
    /// - Parameter i: A valid index of the collection. `i` must be less than
    ///   `endIndex`.
    /// - Returns: The index value immediately after `i`.
    public func index(after i: Int) -> Int {
        guard i < endIndex else { fatalError() }
        return i + 1
    }

    typealias Index = Int
    typealias _Element = Vector<T>

    subscript(index: Int) -> Element? {
        let (_, width) = self.size

        if width > 0 {
            let x = index % width
            let y = index / width

            return self.data[y][x]
        }
        return nil
    }

    var startIndex: Int { return 0 }
    var endIndex: Int {
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

extension Matrix {
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

extension Matrix {
    static func getRandomNumMatrixWithHeight(_ height: Int, byWidth width: Int) -> Matrix<Element> {
        let data = (0..<height).map { _ in
            (0..<width).map { _ in
                Element(2 * drand48() - 1)
            }
        }
        let vectors = data.map(Vector.init)
        return Matrix(vectors)
    }
}

func + <T: Numeric>(lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
    return lhs.add(m: rhs)
}

func - <T: Numeric>(lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
    return lhs.subtract(m: rhs)
}

func * <T: Numeric>(lhs: Matrix<T>, rhs: Matrix<T>) -> Matrix<T> {
    return lhs.mult(m: rhs)
}
