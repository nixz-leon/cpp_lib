#pragma once
#include "GPUAccelerator.hpp"
#include "vec.hpp"
#include "matrix.hpp"
#include <type_traits>

/**
 * GPUVecAdapter - A utility class that adapts vec and matrix operations to use GPU acceleration when appropriate.
 * This class provides a bridge between your existing linear algebra classes and the GPU accelerator.
 */
class GPUVecAdapter {
public:
    /**
     * Matrix multiplication with automatic GPU/CPU selection based on matrix size.
     */
    template<typename T>
    static matrix<T> multiply(const matrix<T>& a, const matrix<T>& b) {
        // Check if dimensions match
        if (a.col != b.row) {
            std::cerr << "Matrix dimensions don't match for multiplication" << std::endl;
            return matrix<T>();
        }
        
        // Create result matrix
        matrix<T> result(a.row, b.col);
        
        // Decide whether to use GPU or CPU based on matrix size
        bool use_gpu = false;
        
        // Only use GPU for float/double operations
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            // Check if matrix size makes GPU acceleration worthwhile
            int operation_size = a.row * a.col * b.col;
            use_gpu = GPUAccelerator::shouldUseGPU(operation_size);
        }
        
        if (use_gpu) {
            bool success = GPUAccelerator::multiplyMatricesGPU(
                a.data, a.row, a.col,
                b.data, b.row, b.col,
                result.data
            );
            
            // Fall back to CPU if GPU fails
            if (!success) {
                GPUAccelerator::multiplyMatricesThreaded(
                    a.data, a.row, a.col,
                    b.data, b.row, b.col,
                    result.data
                );
            }
        } else {
            // Use multithreaded CPU version
            GPUAccelerator::multiplyMatricesThreaded(
                a.data, a.row, a.col,
                b.data, b.row, b.col,
                result.data
            );
        }
        
        return result;
    }
    
    /**
     * Matrix-Vector multiplication with GPU acceleration when appropriate.
     */
    template<typename T>
    static vec<T> multiply(const matrix<T>& a, const vec<T>& b) {
        // Check if dimensions match
        if (a.col != b.size) {
            std::cerr << "Matrix dimensions don't match vector size for multiplication" << std::endl;
            return vec<T>();
        }
        
        // Create result vector
        vec<T> result(a.row);
        
        // Only try GPU for float operations (since we only implemented float GPU vector ops)
        if constexpr (std::is_same_v<T, float>) {
            // Check if size makes GPU acceleration worthwhile
            if (GPUAccelerator::shouldUseGPU(a.row * a.col)) {
                bool success = GPUAccelerator::multiplyMatrixVectorGPU(
                    a.data, a.row, a.col,
                    b.data, result.data
                );
                
                if (success) {
                    return result;
                }
                // Fall back to CPU if GPU fails
            }
        }
        
        // Use CPU implementation
        for (int i = 0; i < a.row; i++) {
            T sum = 0;
            for (int j = 0; j < a.col; j++) {
                sum += a(i, j) * b(j);
            }
            result(i) = sum;
        }
        
        return result;
    }
    
    /**
     * Vector dot product with GPU acceleration for float vectors.
     */
    template<typename T>
    static T dotProduct(const vec<T>& a, const vec<T>& b) {
        // Check if dimensions match
        if (a.size != b.size) {
            std::cerr << "Vector dimensions don't match for dot product" << std::endl;
            return T(0);
        }
        
        // Only try GPU for float operations (since we only implemented float GPU vector ops)
        if constexpr (std::is_same_v<T, float>) {
            // Check if size makes GPU acceleration worthwhile (vectors need to be fairly large)
            if (GPUAccelerator::shouldUseGPU(a.size) && a.size >= 10000) {
                float result = 0.0f;
                bool success = GPUAccelerator::dotProductGPU(
                    a.data, b.data, &result, a.size
                );
                
                if (success) {
                    return static_cast<T>(result);
                }
                // Fall back to CPU if GPU fails
            }
        }
        
        // For smaller vectors or non-float types, use multithreaded CPU version
        return GPUAccelerator::dotProductThreaded(a.data, b.data, a.size);
    }
    
    /**
     * Vector addition with GPU acceleration for float vectors.
     */
    template<typename T>
    static vec<T> add(const vec<T>& a, const vec<T>& b) {
        // Check if dimensions match
        if (a.size != b.size) {
            std::cerr << "Vector dimensions don't match for addition" << std::endl;
            return vec<T>();
        }
        
        // Create result vector
        vec<T> result(a.size);
        
        // Only try GPU for float operations (since we only implemented float GPU vector ops)
        if constexpr (std::is_same_v<T, float>) {
            // Check if size makes GPU acceleration worthwhile (vectors need to be large)
            if (GPUAccelerator::shouldUseGPU(a.size) && a.size >= 100000) {
                bool success = GPUAccelerator::addVectorsGPU(
                    a.data, b.data, result.data, a.size
                );
                
                if (success) {
                    return result;
                }
                // Fall back to CPU if GPU fails
            }
        }
        
        // Use CPU implementation with SIMD hints
        #pragma omp simd
        for (int i = 0; i < a.size; i++) {
            result.data[i] = a.data[i] + b.data[i];
        }
        
        return result;
    }
    
    /**
     * Vector element-wise multiplication with GPU acceleration for float vectors.
     */
    template<typename T>
    static vec<T> multiplyElements(const vec<T>& a, const vec<T>& b) {
        // Check if dimensions match
        if (a.size != b.size) {
            std::cerr << "Vector dimensions don't match for element-wise multiplication" << std::endl;
            return vec<T>();
        }
        
        // Create result vector
        vec<T> result(a.size);
        
        // Only try GPU for float operations (since we only implemented float GPU vector ops)
        if constexpr (std::is_same_v<T, float>) {
            // Check if size makes GPU acceleration worthwhile (vectors need to be large)
            if (GPUAccelerator::shouldUseGPU(a.size) && a.size >= 100000) {
                bool success = GPUAccelerator::multiplyVectorsGPU(
                    a.data, b.data, result.data, a.size
                );
                
                if (success) {
                    return result;
                }
                // Fall back to CPU if GPU fails
            }
        }
        
        // Use CPU implementation with SIMD hints
        #pragma omp simd
        for (int i = 0; i < a.size; i++) {
            result.data[i] = a.data[i] * b.data[i];
        }
        
        return result;
    }
    
    /**
     * Clean up GPU resources when done using the adapter.
     */
    static void cleanup() {
        GPUAccelerator::cleanup();
    }
}; 