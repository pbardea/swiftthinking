//
//  Numeric.swift
//  Swector
//
//  Created by Paul Bardea on 2016-05-10.
//  Copyright © 2016 pbardea. All rights reserved.
//

import Foundation

public protocol Numeric {
    static func + (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Self) -> Self
    static func / (lhs: Self, rhs: Self) -> Self
    static func % (lhs: Self, rhs: Self) -> Self
    init(_ v: Int)
    init(_ v: Double)
    init(_ v: Float)
    init(_ v: Int8)
    init(_ v: Int16)
    init(_ v: Int32)
    init(_ v: Int64)
    init(_ v: UInt)
    init(_ v: UInt8)
    init(_ v: UInt32)
    init(_ v: UInt64)

}

extension Double: Numeric {}
extension Float: Numeric {}
extension Int: Numeric {}
extension Int8: Numeric {}
extension Int16: Numeric {}
extension Int32: Numeric {}
extension Int64: Numeric {}
extension UInt: Numeric {}
extension UInt8: Numeric {}
extension UInt16: Numeric {}
extension UInt32: Numeric {}
extension UInt64: Numeric {}
