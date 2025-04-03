#pragma once

/**
 * @file EnableGPUAcceleration.hpp
 * @brief Header file to enable GPU acceleration for vector and matrix operations
 * 
 * This file provides functionality to enable GPU acceleration for linear algebra operations.
 * Include this file in your project to activate GPU acceleration for compatible operations.
 */

// Define macro to enable GPU acceleration
#define GPU_ACCELERATION_ENABLED

// Include GPU adapter implementation
#include "GPUVecAdapter.hpp"

// Global variable to track initialization state
static bool gpu_initialized = false;

/**
 * @brief Initialize GPU acceleration
 * 
 * This function attempts to detect and initialize GPU hardware capabilities
 * for linear algebra operations. It returns true if GPU acceleration is available.
 * 
 * @return bool True if GPU is available and initialized, false otherwise
 */
inline bool initializeGPUAcceleration() {
    if (gpu_initialized) {
        return true;
    }
    
    try {
        bool success = GPUVecAdapter::detectHardwareCapabilities();
        gpu_initialized = success;
        
        if (success) {
            std::cout << "GPU acceleration initialized successfully." << std::endl;
        } else {
            std::cout << "GPU acceleration not available. Using CPU fallbacks." << std::endl;
        }
        
        return success;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize GPU acceleration: " << e.what() << std::endl;
        std::cerr << "Operations will use CPU fallbacks." << std::endl;
        return false;
    }
}

/**
 * @brief Clean up GPU resources
 * 
 * This function should be called at the end of your program to properly
 * release GPU resources and prevent memory leaks.
 */
inline void cleanupGPUAcceleration() {
    if (gpu_initialized) {
        try {
            GPUVecAdapter::cleanup();
            gpu_initialized = false;
            std::cout << "GPU resources cleaned up successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to clean up GPU resources: " << e.what() << std::endl;
        }
    }
}

// Automatically initialize GPU acceleration when this header is included
static bool _gpu_auto_init = initializeGPUAcceleration(); 