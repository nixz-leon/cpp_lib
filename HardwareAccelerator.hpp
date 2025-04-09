#pragma once

/**
 * @file HardwareAccelerator.hpp
 * @brief Centralized hardware acceleration for vector and matrix operations
 * 
 * This file provides a unified interface for hardware-accelerated operations,
 * automatically detecting and utilizing available CPU and GPU capabilities.
 */

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <future>
#include <cmath>
#include <type_traits>

// Forward declarations
template<typename T> class vec;
template<typename T> class matrix;

/**
 * HardwareAccelerator - A static utility class that provides hardware-accelerated operations
 * for vectors and matrices, automatically selecting the best available hardware.
 */
class HardwareAccelerator {
private:
    // CPU capabilities
    static bool cpu_capabilities_detected;
    static int cpu_cores;
    static size_t l1_cache_size;
    static int thread_threshold;
    static int cache_block_size;

    // GPU capabilities
    static bool gpu_capabilities_detected;
    static bool gpu_available;
    static GLFWwindow* window;
    static GLuint floatMatMulProgram;
    static GLuint doubleMatMulProgram;
    static GLuint floatVecAddProgram;
    static GLuint floatVecMultProgram;
    static bool glInitialized;
    static std::mutex gl_mutex;
    static std::atomic<bool> initialized;
    static int maxWorkGroupSizeX;
    static int maxWorkGroupSizeY;
    static size_t maxBufferSize;
    static int optimalGpuMatrixSize;
    static bool hasDoubleSupport;
    static int gpu_threshold;

    // Detect CPU capabilities
    static void detectCPUCapabilities() {
        if (cpu_capabilities_detected) return;

        // Detect number of CPU cores
        cpu_cores = std::thread::hardware_concurrency();
        
        // Default L1 cache size (32KB if detection fails)
        l1_cache_size = 32 * 1024;

        #ifdef __linux__
        // Try to read L1 cache size from sysfs
        std::ifstream cache_info("/sys/devices/system/cpu/cpu0/cache/index0/size");
        if (cache_info.is_open()) {
            std::string size_str;
            cache_info >> size_str;
            if (size_str.back() == 'K') {
                size_str.pop_back();
                try {
                    l1_cache_size = std::stoi(size_str) * 1024;
                } catch (...) {
                    // Keep default if parsing fails
                }
            }
        }
        #endif

        // Calculate optimal cache block size based on L1 cache
        cache_block_size = static_cast<int>(sqrt(l1_cache_size / (2 * sizeof(double))));
        cache_block_size = std::min(64, std::max(16, cache_block_size));
        
        // Calculate thread threshold based on CPU cores and cache size
        thread_threshold = static_cast<int>(sqrt(l1_cache_size / sizeof(double)) * cpu_cores);
        thread_threshold = std::max(128, std::min(512, thread_threshold));

        cpu_capabilities_detected = true;
    }

    // Initialize OpenGL for compute shaders
    static bool initGL() {
        std::lock_guard<std::mutex> lock(gl_mutex);
        
        if (glInitialized) return true;
        
        try {
            if (!glfwInit()) {
                return false;
            }
            
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            
            window = glfwCreateWindow(1, 1, "Compute", nullptr, nullptr);
            if (!window) {
                glfwTerminate();
                return false;
            }
            
            glfwMakeContextCurrent(window);
            
            if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
                glfwDestroyWindow(window);
                glfwTerminate();
                return false;
            }
            
            // Check for compute shader support
            GLint computeShaders = 0;
            glGetIntegerv(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &computeShaders);
            if (computeShaders <= 0) {
                glfwDestroyWindow(window);
                glfwTerminate();
                return false;
            }
            
            // Get maximum work group sizes and memory limits
            glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxWorkGroupSizeX);
            glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 1, &maxWorkGroupSizeY);
            glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, (GLint*)&maxBufferSize);
            
            // Check for double precision support
            GLint numExtensions;
            glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);
            hasDoubleSupport = false;
            
            for (GLint i = 0; i < numExtensions; ++i) {
                const GLubyte* extension = glGetStringi(GL_EXTENSIONS, i);
                if (std::strcmp(reinterpret_cast<const char*>(extension), "GL_ARB_gpu_shader_fp64") == 0) {
                    hasDoubleSupport = true;
                    break;
                }
            }
            
            // Estimate optimal GPU matrix size based on device capabilities
            optimalGpuMatrixSize = std::min(
                static_cast<int>(std::sqrt(maxBufferSize / sizeof(float))),
                maxWorkGroupSizeX * 32  // Assuming 32 work groups is optimal
            );
            
            // Set GPU threshold for when to use GPU acceleration
            gpu_threshold = std::max(256, optimalGpuMatrixSize);
            
            glInitialized = true;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "GL initialization failed: " << e.what() << std::endl;
            if (window) {
                glfwDestroyWindow(window);
                window = nullptr;
            }
            glfwTerminate();
            return false;
        }
    }

    // Load a compute shader from file
    static GLuint loadComputeShader(const char* filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open compute shader file");
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string shader_source = buffer.str();
        const char* sourcePtr = shader_source.c_str();

        GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(computeShader, 1, &sourcePtr, nullptr);
        glCompileShader(computeShader);

        GLint success;
        glGetShaderiv(computeShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            GLchar infoLog[512];
            glGetShaderInfoLog(computeShader, 512, nullptr, infoLog);
            throw std::runtime_error(std::string("Shader compilation failed: ") + infoLog);
        }

        GLuint program = glCreateProgram();
        glAttachShader(program, computeShader);
        glLinkProgram(program);

        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            GLchar infoLog[512];
            glGetProgramInfoLog(program, 512, nullptr, infoLog);
            throw std::runtime_error(std::string("Program linking failed: ") + infoLog);
        }

        glDeleteShader(computeShader);
        return program;
    }

    // Detect GPU capabilities
    static void detectGPUCapabilities() {
        if (gpu_capabilities_detected) return;
        
        gpu_available = initGL();
        gpu_capabilities_detected = true;
    }

public:
    // Initialize hardware acceleration
    static bool initialize() {
        if (initialized) return true;
        
        // Detect CPU capabilities
        detectCPUCapabilities();
        
        // Try to detect GPU capabilities
        detectGPUCapabilities();
        
        // If GPU is available, create shaders
        if (gpu_available) {
            try {
                createVectorShaders();
                initialized = true;
                return true;
            } catch (const std::exception& e) {
                std::cerr << "Failed to create GPU shaders: " << e.what() << std::endl;
                gpu_available = false;
                return false;
            }
        }
        
        initialized = true;
        return true;
    }

    // Create vector operation shaders
    static bool createVectorShaders() {
        if (!gpu_available) return false;
        
        try {
            // Load shaders from files
            floatVecAddProgram = loadComputeShader("include/shaders/vec_add_float.comp");
            floatVecMultProgram = loadComputeShader("include/shaders/vec_mult_float.comp");
            floatMatMulProgram = loadComputeShader("include/shaders/matmul_float.comp");
            
            if (hasDoubleSupport) {
                doubleMatMulProgram = loadComputeShader("include/shaders/matmul_double.comp");
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to create shaders: " << e.what() << std::endl;
            return false;
        }
    }

    // Check if GPU is available
    static bool isGPUAvailable() {
        if (!gpu_capabilities_detected) {
            detectGPUCapabilities();
        }
        return gpu_available;
    }

    // Get CPU cores
    static int getCPUCores() {
        if (!cpu_capabilities_detected) {
            detectCPUCapabilities();
        }
        return cpu_cores;
    }

    // Get L1 cache size
    static size_t getL1CacheSize() {
        if (!cpu_capabilities_detected) {
            detectCPUCapabilities();
        }
        return l1_cache_size;
    }

    // Get thread threshold
    static int getThreadThreshold() {
        if (!cpu_capabilities_detected) {
            detectCPUCapabilities();
        }
        return thread_threshold;
    }

    // Get cache block size
    static int getCacheBlockSize() {
        if (!cpu_capabilities_detected) {
            detectCPUCapabilities();
        }
        return cache_block_size;
    }

    // Get GPU threshold
    static int getGPUThreshold() {
        if (!gpu_capabilities_detected) {
            detectGPUCapabilities();
        }
        return gpu_threshold;
    }

    // Determine if GPU should be used for a given operation size
    static bool shouldUseGPU(int size) {
        if (!gpu_capabilities_detected) {
            detectGPUCapabilities();
        }
        return gpu_available && size >= gpu_threshold;
    }

    // Determine if threading should be used for a given operation size
    static bool shouldUseThreading(int size) {
        if (!cpu_capabilities_detected) {
            detectCPUCapabilities();
        }
        return size >= thread_threshold;
    }

    // Matrix multiplication with automatic hardware selection
    template<typename T>
    static bool multiplyMatrices(
        const T* a, int a_rows, int a_cols,
        const T* b, int b_rows, int b_cols,
        T* result)
    {
        // Check dimensions
        if (a_cols != b_rows) {
            return false;
        }
        
        // Determine operation size
        int operation_size = a_rows * a_cols * b_cols;
        
        // Try GPU if available and operation is large enough
        if (isGPUAvailable() && shouldUseGPU(operation_size)) {
            if constexpr (std::is_same_v<T, float>) {
                return multiplyMatricesGPU(a, a_rows, a_cols, b, b_rows, b_cols, result);
            } else if constexpr (std::is_same_v<T, double> && hasDoubleSupport) {
                return multiplyMatricesGPU(a, a_rows, a_cols, b, b_rows, b_cols, result);
            }
        }
        
        // Fall back to threaded CPU implementation
        multiplyMatricesThreaded(a, a_rows, a_cols, b, b_rows, b_cols, result);
        return true;
    }

    // Matrix-vector multiplication with automatic hardware selection
    template<typename T>
    static bool multiplyMatrixVector(
        const T* matrix, int rows, int cols,
        const T* vector, T* result)
    {
        // Determine operation size
        int operation_size = rows * cols;
        
        // Try GPU if available and operation is large enough
        if (isGPUAvailable() && shouldUseGPU(operation_size)) {
            if constexpr (std::is_same_v<T, float>) {
                return multiplyMatrixVectorGPU(matrix, rows, cols, vector, result);
            }
        }
        
        // Fall back to CPU implementation
        for (int i = 0; i < rows; i++) {
            T sum = 0;
            for (int j = 0; j < cols; j++) {
                sum += matrix[i * cols + j] * vector[j];
            }
            result[i] = sum;
        }
        return true;
    }

    // Vector dot product with automatic hardware selection
    template<typename T>
    static T dotProduct(const T* a, const T* b, int length) {
        T sum = 0;
        #pragma omp parallel for simd reduction(+:sum)
        for (int i = 0; i < length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    template<typename T>
    static bool dotProduct(const T* a, const T* b, int length, T& result) {
        try {
            result = dotProduct(a, b, length);
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Dot product failed: " << e.what() << std::endl;
            return false;
        }
    }

    // Vector addition with automatic hardware selection
    template<typename T>
    static bool addVectors(const T* a, const T* b, T* result, int length) {
        // Determine operation size
        int operation_size = length;
        
        // Try GPU if available and operation is large enough
        if (isGPUAvailable() && shouldUseGPU(operation_size)) {
            if constexpr (std::is_same_v<T, float>) {
                return addVectorsGPU(a, b, result, length);
            }
        }
        
        // Fall back to CPU implementation with SIMD hints
        #pragma omp simd
        for (int i = 0; i < length; i++) {
            result[i] = a[i] + b[i];
        }
        return true;
    }

    // Vector element-wise multiplication with automatic hardware selection
    template<typename T>
    static bool multiplyVectors(const T* a, const T* b, T* result, int length) {
        // Determine operation size
        int operation_size = length;
        
        // Try GPU if available and operation is large enough
        if (isGPUAvailable() && shouldUseGPU(operation_size)) {
            if constexpr (std::is_same_v<T, float>) {
                return multiplyVectorsGPU(a, b, result, length);
            }
        }
        
        // Fall back to CPU implementation with SIMD hints
        #pragma omp simd
        for (int i = 0; i < length; i++) {
            result[i] = a[i] * b[i];
        }
        return true;
    }

    // GPU-specific implementations
    template<typename T>
    static bool multiplyMatricesGPU(
        const T* a, int a_rows, int a_cols,
        const T* b, int b_rows, int b_cols,
        T* result)
    {
        if (!gpu_available) return false;
        
        std::lock_guard<std::mutex> lock(gl_mutex);
        
        try {
            // Select appropriate shader program
            GLuint program;
            if constexpr (std::is_same_v<T, float>) {
                program = floatMatMulProgram;
            } else if constexpr (std::is_same_v<T, double>) {
                if (!hasDoubleSupport) return false;
                program = doubleMatMulProgram;
            } else {
                return false;
            }
            
            // Create and bind buffers
            GLuint aBuffer, bBuffer, resultBuffer;
            glGenBuffers(1, &aBuffer);
            glGenBuffers(1, &bBuffer);
            glGenBuffers(1, &resultBuffer);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, aBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, a_rows * a_cols * sizeof(T), a, GL_STATIC_READ);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, b_rows * b_cols * sizeof(T), b, GL_STATIC_READ);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, a_rows * b_cols * sizeof(T), nullptr, GL_STATIC_COPY);
            
            // Use shader program
            glUseProgram(program);
            
            // Set uniforms
            glUniform1i(glGetUniformLocation(program, "a_rows"), a_rows);
            glUniform1i(glGetUniformLocation(program, "a_cols"), a_cols);
            glUniform1i(glGetUniformLocation(program, "b_cols"), b_cols);
            
            // Bind buffers to shader storage blocks
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, aBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, resultBuffer);
            
            // Dispatch compute shader
            glDispatchCompute(
                (a_rows + 15) / 16,
                (b_cols + 15) / 16,
                1
            );
            
            // Wait for completion
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // Read back results
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, a_rows * b_cols * sizeof(T), result);
            
            // Clean up
            glDeleteBuffers(1, &aBuffer);
            glDeleteBuffers(1, &bBuffer);
            glDeleteBuffers(1, &resultBuffer);
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "GPU matrix multiplication failed: " << e.what() << std::endl;
            return false;
        }
    }

    static bool multiplyMatrixVectorGPU(
        const float* matrix, int rows, int cols,
        const float* vector, float* result)
    {
        if (!gpu_available) return false;
        
        std::lock_guard<std::mutex> lock(gl_mutex);
        
        try {
            // Implementation similar to matrix multiplication but for matrix-vector
            // This is a simplified version - you would need to implement the actual shader
            return false;
        } catch (const std::exception& e) {
            std::cerr << "GPU matrix-vector multiplication failed: " << e.what() << std::endl;
            return false;
        }
    }

    static bool addVectorsGPU(const float* a, const float* b, float* result, int length) {
        if (!gpu_available) return false;
        
        std::lock_guard<std::mutex> lock(gl_mutex);
        
        try {
            // Use the vector addition shader
            glUseProgram(floatVecAddProgram);
            
            // Create and bind buffers
            GLuint aBuffer, bBuffer, resultBuffer;
            glGenBuffers(1, &aBuffer);
            glGenBuffers(1, &bBuffer);
            glGenBuffers(1, &resultBuffer);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, aBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), a, GL_STATIC_READ);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), b, GL_STATIC_READ);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), nullptr, GL_STATIC_COPY);
            
            // Set uniform
            glUniform1i(glGetUniformLocation(floatVecAddProgram, "length"), length);
            
            // Bind buffers to shader storage blocks
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, aBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, resultBuffer);
            
            // Dispatch compute shader
            glDispatchCompute((length + 255) / 256, 1, 1);
            
            // Wait for completion
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // Read back results
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, length * sizeof(float), result);
            
            // Clean up
            glDeleteBuffers(1, &aBuffer);
            glDeleteBuffers(1, &bBuffer);
            glDeleteBuffers(1, &resultBuffer);
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "GPU vector addition failed: " << e.what() << std::endl;
            return false;
        }
    }

    static bool multiplyVectorsGPU(const float* a, const float* b, float* result, int length) {
        if (!gpu_available) return false;
        
        std::lock_guard<std::mutex> lock(gl_mutex);
        
        try {
            // Use the vector multiplication shader
            glUseProgram(floatVecMultProgram);
            
            // Create and bind buffers
            GLuint aBuffer, bBuffer, resultBuffer;
            glGenBuffers(1, &aBuffer);
            glGenBuffers(1, &bBuffer);
            glGenBuffers(1, &resultBuffer);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, aBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), a, GL_STATIC_READ);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), b, GL_STATIC_READ);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), nullptr, GL_STATIC_COPY);
            
            // Set uniform
            glUniform1i(glGetUniformLocation(floatVecMultProgram, "length"), length);
            
            // Bind buffers to shader storage blocks
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, aBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bBuffer);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, resultBuffer);
            
            // Dispatch compute shader
            glDispatchCompute((length + 255) / 256, 1, 1);
            
            // Wait for completion
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // Read back results
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, resultBuffer);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, length * sizeof(float), result);
            
            // Clean up
            glDeleteBuffers(1, &aBuffer);
            glDeleteBuffers(1, &bBuffer);
            glDeleteBuffers(1, &resultBuffer);
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "GPU vector multiplication failed: " << e.what() << std::endl;
            return false;
        }
    }

    // CPU-specific implementations
    template<typename T>
    static void multiplyMatricesThreaded(
        const T* a, int a_rows, int a_cols,
        const T* b, int b_rows, int b_cols,
        T* result)
    {
        // Determine if threading should be used
        int operation_size = a_rows * a_cols * b_cols;
        bool use_threading = shouldUseThreading(operation_size);
        
        if (use_threading) {
            // Multi-threaded implementation
            int num_threads = std::min(
                static_cast<int>(std::thread::hardware_concurrency()),
                (a_rows + cache_block_size - 1) / cache_block_size
            );
            
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            
            auto multiply_block = [&](int start_row, int end_row) {
                for (int i = start_row; i < end_row; i += cache_block_size) {
                    for (int k = 0; k < a_cols; k += cache_block_size) {
                        for (int j = 0; j < b_cols; j += cache_block_size) {
                            // Block multiplication
                            for (int ii = i; ii < std::min(i + cache_block_size, end_row); ++ii) {
                                for (int kk = k; kk < std::min(k + cache_block_size, a_cols); ++kk) {
                                    T r = a[ii * a_cols + kk];
                                    for (int jj = j; jj < std::min(j + cache_block_size, b_cols); ++jj) {
                                        result[ii * b_cols + jj] += r * b[kk * b_cols + jj];
                                    }
                                }
                            }
                        }
                    }
                }
            };
            
            // Distribute work among threads
            const int rows_per_thread = (a_rows + num_threads - 1) / num_threads;
            
            for (int t = 0; t < num_threads; ++t) {
                const int start_row = t * rows_per_thread;
                const int end_row = std::min(start_row + rows_per_thread, a_rows);
                if (start_row < end_row) {
                    threads.emplace_back(multiply_block, start_row, end_row);
                }
            }
            
            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
        } else {
            // Sequential implementation with cache blocking
            for (int i = 0; i < a_rows; i += cache_block_size) {
                for (int k = 0; k < a_cols; k += cache_block_size) {
                    for (int j = 0; j < b_cols; j += cache_block_size) {
                        // Block multiplication
                        for (int ii = i; ii < std::min(i + cache_block_size, a_rows); ++ii) {
                            for (int kk = k; kk < std::min(k + cache_block_size, a_cols); ++kk) {
                                T r = a[ii * a_cols + kk];
                                for (int jj = j; jj < std::min(j + cache_block_size, b_cols); ++jj) {
                                    result[ii * b_cols + jj] += r * b[kk * b_cols + jj];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Clean up resources
    static void cleanup() {
        if (gpu_available) {
            std::lock_guard<std::mutex> lock(gl_mutex);
            
            if (floatMatMulProgram) {
                glDeleteProgram(floatMatMulProgram);
                floatMatMulProgram = 0;
            }
            
            if (doubleMatMulProgram) {
                glDeleteProgram(doubleMatMulProgram);
                doubleMatMulProgram = 0;
            }
            
            if (floatVecAddProgram) {
                glDeleteProgram(floatVecAddProgram);
                floatVecAddProgram = 0;
            }
            
            if (floatVecMultProgram) {
                glDeleteProgram(floatVecMultProgram);
                floatVecMultProgram = 0;
            }
            
            if (window) {
                glfwDestroyWindow(window);
                window = nullptr;
            }
            
            glfwTerminate();
            glInitialized = false;
        }
    }
};

// Initialize static members
bool HardwareAccelerator::cpu_capabilities_detected = false;
int HardwareAccelerator::cpu_cores = 0;
size_t HardwareAccelerator::l1_cache_size = 0;
int HardwareAccelerator::thread_threshold = 128;  // Default value
int HardwareAccelerator::cache_block_size = 32;   // Default value

bool HardwareAccelerator::gpu_capabilities_detected = false;
bool HardwareAccelerator::gpu_available = false;
GLFWwindow* HardwareAccelerator::window = nullptr;
GLuint HardwareAccelerator::floatMatMulProgram = 0;
GLuint HardwareAccelerator::doubleMatMulProgram = 0;
GLuint HardwareAccelerator::floatVecAddProgram = 0;
GLuint HardwareAccelerator::floatVecMultProgram = 0;
bool HardwareAccelerator::glInitialized = false;
std::mutex HardwareAccelerator::gl_mutex;
std::atomic<bool> HardwareAccelerator::initialized(false);
int HardwareAccelerator::maxWorkGroupSizeX = 0;
int HardwareAccelerator::maxWorkGroupSizeY = 0;
size_t HardwareAccelerator::maxBufferSize = 0;
int HardwareAccelerator::optimalGpuMatrixSize = 0;
bool HardwareAccelerator::hasDoubleSupport = false;
int HardwareAccelerator::gpu_threshold = 256;  // Default value 