//
//  Extensions.swift
//  SwiftThinking
//
//  Created by Paul Bardea on 2016-12-15.
//  Copyright © 2016 pbardea. All rights reserved.
//

import Foundation

extension Matrix {
    static func getRandomNumMatrixWithHeight(_ height: Int, byWidth width: Int) -> Matrix<Double> {
        let data = (0..<height).map { _ in
            (0..<width).map { _ in
                Double(2 * drand48() - 1)
            }
        }
        let vectors = data.map(Vector.init)
        return Matrix<Double>(vectors)
    }
}
