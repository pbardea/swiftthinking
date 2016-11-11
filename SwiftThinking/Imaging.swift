//
//  Imaging.swift
//  NeuralNet
//
//  Created by Paul Bardea on 2016-11-05.
//  Copyright © 2016 pbardea. All rights reserved.
//

// import Foundation
// import UIKit

public struct Pixel {
    public var value: UInt32 = 0
    
    //red
    public var R: UInt8 {
        get { return UInt8(value & 0xFF); }
        set { value = UInt32(newValue) | (value & 0xFFFFFF00) }
    }
    
    //green
    public var G: UInt8 {
        get { return UInt8((value >> 8) & 0xFF) }
        set { value = (UInt32(newValue) << 8) | (value & 0xFFFF00FF) }
    }
    
    //blue
    public var B: UInt8 {
        get { return UInt8((value >> 16) & 0xFF) }
        set { value = (UInt32(newValue) << 16) | (value & 0xFF00FFFF) }
    }
    
    //alpha
    public var A: UInt8 {
        get { return UInt8((value >> 24) & 0xFF) }
        set { value = (UInt32(newValue) << 24) | (value & 0x00FFFFFF) }
    }
    
    public init(red: UInt8, green: UInt8, blue: UInt8) {
        self.R = red
        self.G = green
        self.B = blue
        self.A = 255
    }
    
}

public struct RGBAImage {
    public var pixels: UnsafeMutableBufferPointer<Pixel>
    public var width: Int
    public var height: Int
    
    public init?(image: UIImage) {
        guard let cgImage = image.cgImage else {
            return nil
        }
        
        width = Int(image.size.width)
        height = Int(image.size.height)
        
        let bytesPerPixels = 4
        let bytesPerRow = width * bytesPerPixels
        let imageData = UnsafeMutablePointer<Pixel>.allocate(capacity: width * height)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        
        var bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Big.rawValue
        bitmapInfo = bitmapInfo | CGImageAlphaInfo.premultipliedLast.rawValue & CGBitmapInfo.alphaInfoMask.rawValue
        
        guard let imageContext = CGContext(data: imageData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo) else {
            return nil
        }
        
        //CGContextDrawImage(imageContext, CGRect(origin: .zero, size: image.size), cgImage)
        imageContext.draw(cgImage, in: CGRect(x: 0.0,y: 0.0,width: image.size.width,height: image.size.height))
        
        pixels = UnsafeMutableBufferPointer<Pixel>(start: imageData, count: width * height)
    }
    
    
    /// Convert the RBGAImage to a UIImage
    public func toUIImage() -> UIImage? {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var bitmapInfo: UInt32 = CGBitmapInfo.byteOrder32Big.rawValue
        let bytesPerRow = width * 4
        
        bitmapInfo |= CGImageAlphaInfo.premultipliedLast.rawValue & CGBitmapInfo.alphaInfoMask.rawValue
        
        let imageContext = CGContext(data: pixels.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: bytesPerRow, space: colorSpace, bitmapInfo: bitmapInfo, releaseCallback: nil, releaseInfo: nil)
        guard let cgImage = imageContext?.makeImage() else { return nil }
        
        let image = UIImage(cgImage: cgImage)
        return image
    }
    
    /// Return the pixel at (`x`,`y`)
    public func pixel(x : Int, _ y : Int) -> Pixel? {
        guard x >= 0 && x < width && y >= 0 && y < height else {
            return nil
        }
        
        let address = y * width + x
        return pixels[address]
    }
    
    public func map<T>( function: (Pixel) -> T) -> [T] {
        var newVals = [T]()
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                let outPixel = function(pixels[index])
                newVals.append(outPixel)
                // newVals[index] = outPixel
            }
        }
        return newVals
    }
    
    public func process( functor : ((Pixel) -> Pixel) ) {
        for y in 0..<height {
            for x in 0..<width {
                let index = y * width + x
                let outPixel = functor(pixels[index])
                pixels[index] = outPixel
            }
        }
    }
}

