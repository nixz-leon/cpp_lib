#pragma once

/**
 * @file HardwareAcceleration.hpp
 * @brief Main entry point for hardware acceleration in vector and matrix operations
 * 
 * This file provides a unified interface for hardware-accelerated operations,
 * automatically detecting and utilizing available CPU and GPU capabilities.
 * Include this file in your vector and matrix classes to enable hardware acceleration.
 */

#include "HardwareAccelerator.hpp"
#include <iostream>

// Global variable to track initialization state
static bool hardware_initialized = false;

/**
 * @brief Initialize hardware acceleration
 * 
 * This function attempts to detect and initialize hardware capabilities
 * for linear algebra operations. It returns true if hardware acceleration is available.
 * 
 * @return bool True if hardware acceleration is available and initialized, false otherwise
 */
inline bool initializeHardwareAcceleration() {
    if (hardware_initialized) {
        return true;
    }
    
    try {
        bool success = HardwareAccelerator::initialize();
        hardware_initialized = success;
        
        if (success) {
            std::cout << "Hardware acceleration initialized successfully." << std::endl;
            
            // Print hardware capabilities
            std::cout << "CPU Cores: " << HardwareAccelerator::getCPUCores() << std::endl;
            std::cout << "L1 Cache Size: " << HardwareAccelerator::getL1CacheSize() / 1024 << " KB" << std::endl;
            std::cout << "Thread Threshold: " << HardwareAccelerator::getThreadThreshold() << std::endl;
            std::cout << "Cache Block Size: " << HardwareAccelerator::getCacheBlockSize() << std::endl;
            
            if (HardwareAccelerator::isGPUAvailable()) {
                std::cout << "GPU acceleration is available." << std::endl;
                std::cout << "GPU Threshold: " << HardwareAccelerator::getGPUThreshold() << std::endl;
            } else {
                std::cout << "GPU acceleration is not available. Using CPU only." << std::endl;
            }
        } else {
            std::cout << "Hardware acceleration not available. Using CPU fallbacks." << std::endl;
        }
        
        return success;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize hardware acceleration: " << e.what() << std::endl;
        std::cerr << "Operations will use CPU fallbacks." << std::endl;
        return false;
    }
}

/**
 * @brief Clean up hardware resources
 * 
 * This function should be called at the end of your program to properly
 * release hardware resources and prevent memory leaks.
 */
inline void cleanupHardwareAcceleration() {
    if (hardware_initialized) {
        try {
            HardwareAccelerator::cleanup();
            hardware_initialized = false;
            std::cout << "Hardware resources cleaned up successfully." << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to clean up hardware resources: " << e.what() << std::endl;
        }
    }
}

// Automatically initialize hardware acceleration when this header is included
static bool _hardware_auto_init = initializeHardwareAcceleration(); 