//
//  Helpers.swift
//  NeuralNetwork
//
//  Created by pbardea on 10/28/16.
//  Copyright © 2016 pbardea stdios. All rights reserved.
//

import Foundation

extension Int {
    
    func bits(numOfDigits: Int) -> [Int] {
        let binaryString = String(self, radix: 2)
        let numberOfZeros = numOfDigits - binaryString.characters.count
        let zeros: [Int] = [Int](repeating: 0, count: numberOfZeros)
        return zeros + binaryString.characters.map { Int(String($0)) ?? 0 }
    }
    
}
