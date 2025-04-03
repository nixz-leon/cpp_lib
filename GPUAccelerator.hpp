#pragma once
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
/**
 * GPUAccelerator - A static utility class that provides GPU-accelerated operations for vectors and matrices.
 * It works with raw pointers to avoid dependence on specific container classes.
 */
class GPUAccelerator {
private:
    // OpenGL context variables
    static GLFWwindow* window;
    static GLuint floatMatMulProgram;
    static GLuint doubleMatMulProgram;
    static GLuint floatVecAddProgram;
    static GLuint floatVecMultProgram;
    static bool glInitialized;
    static std::mutex gl_mutex;
    static std::atomic<bool> initialized;

    // Hardware capability detection
    static int maxWorkGroupSizeX;
    static int maxWorkGroupSizeY;
    static size_t maxBufferSize;
    static int optimalGpuMatrixSize;
    static int numThreads;
    static bool hasDoubleSupport;

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
            
            // Load compute shaders
            try {
                floatMatMulProgram = loadComputeShader("include/shaders/matmul_float.comp");
                if (hasDoubleSupport) {
                    doubleMatMulProgram = loadComputeShader("include/shaders/matmul_double.comp");
                }
                
                // We'll create these later
                // floatVecAddProgram = loadComputeShader("include/shaders/vec_add_float.comp");
                // floatVecMultProgram = loadComputeShader("include/shaders/vec_mult_float.comp");
                
                glInitialized = true;
                return true;
            } catch (const std::runtime_error& e) {
                std::cerr << "Shader loading failed: " << e.what() << std::endl;
                glfwDestroyWindow(window);
                glfwTerminate();
                return false;
            }
        } catch (const std::exception& e) {
            std::cerr << "GL initialization failed: " << e.what() << std::endl;
            return false;
        }
    }

    // Load shader from file
    static GLuint loadComputeShader(const char* filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error(std::string("Failed to open compute shader file: ") + filepath);
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

    // Initialize system capabilities (CPU detection)
    static void detectHardwareCapabilities() {
        if (initialized) return;
        
        // Initialize OpenGL for GPU operations
        bool gpuAvailable = initGL();

        // Detect number of threads for CPU operations
        numThreads = std::thread::hardware_concurrency();
        if (numThreads == 0) numThreads = 4; // Fallback if detection fails
        
        initialized.store(true);
    }

public:
    // Create vector shader programs
    static bool createVectorShaders() {
        if (!glInitialized && !initGL()) {
            return false;
        }
        
        // Create vector addition shader
        const char* vecAddSource = R"(
            #version 430
            layout(local_size_x = 256) in;
            
            layout(std430, binding = 0) readonly buffer InputA {
                float a[];
            };
            
            layout(std430, binding = 1) readonly buffer InputB {
                float b[];
            };
            
            layout(std430, binding = 2) writeonly buffer Output {
                float result[];
            };
            
            uniform int length;
            
            void main() {
                uint id = gl_GlobalInvocationID.x;
                if (id < length) {
                    result[id] = a[id] + b[id];
                }
            }
        )";
        
        // Create vector multiply shader
        const char* vecMultSource = R"(
            #version 430
            layout(local_size_x = 256) in;
            
            layout(std430, binding = 0) readonly buffer InputA {
                float a[];
            };
            
            layout(std430, binding = 1) readonly buffer InputB {
                float b[];
            };
            
            layout(std430, binding = 2) writeonly buffer Output {
                float result[];
            };
            
            uniform int length;
            
            void main() {
                uint id = gl_GlobalInvocationID.x;
                if (id < length) {
                    result[id] = a[id] * b[id];
                }
            }
        )";
        
        // Compile shaders from source
        const char* sources[1];
        
        // Vector addition shader
        sources[0] = vecAddSource;
        GLuint addShader = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(addShader, 1, sources, nullptr);
        glCompileShader(addShader);
        
        GLint success;
        glGetShaderiv(addShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            GLchar infoLog[512];
            glGetShaderInfoLog(addShader, 512, nullptr, infoLog);
            std::cerr << "Vector add shader compilation failed: " << infoLog << std::endl;
            return false;
        }
        
        floatVecAddProgram = glCreateProgram();
        glAttachShader(floatVecAddProgram, addShader);
        glLinkProgram(floatVecAddProgram);
        glDeleteShader(addShader);
        
        // Vector multiply shader
        sources[0] = vecMultSource;
        GLuint multShader = glCreateShader(GL_COMPUTE_SHADER);
        glShaderSource(multShader, 1, sources, nullptr);
        glCompileShader(multShader);
        
        glGetShaderiv(multShader, GL_COMPILE_STATUS, &success);
        if (!success) {
            GLchar infoLog[512];
            glGetShaderInfoLog(multShader, 512, nullptr, infoLog);
            std::cerr << "Vector mult shader compilation failed: " << infoLog << std::endl;
            return false;
        }
        
        floatVecMultProgram = glCreateProgram();
        glAttachShader(floatVecMultProgram, multShader);
        glLinkProgram(floatVecMultProgram);
        glDeleteShader(multShader);
        
        return true;
    }

    // Matrix multiplication using GPU (float version)
    template<typename T>
    static bool multiplyMatricesGPU(
        const T* a, int a_rows, int a_cols,
        const T* b, int b_rows, int b_cols,
        T* result)
    {
        if (!initialized) detectHardwareCapabilities();
        
        // Validate input
        if (a_cols != b_rows) {
            std::cerr << "Matrix dimensions don't match for multiplication" << std::endl;
            return false;
        }
        
        // Select appropriate program based on type
        GLuint program;
        if constexpr (std::is_same_v<T, float>) {
            program = floatMatMulProgram;
        } else if constexpr (std::is_same_v<T, double>) {
            if (!hasDoubleSupport) {
                std::cerr << "Double precision not supported by GPU" << std::endl;
                return false;
            }
            program = doubleMatMulProgram;
        } else {
            std::cerr << "Unsupported type for GPU matrix multiplication" << std::endl;
            return false;
        }
        
        if (!glInitialized || program == 0) {
            if (!initGL()) {
                return false;
            }
        }
        
        try {
            // Create buffers
            GLuint buffers[3];
            glGenBuffers(3, buffers);
            
            // Matrix A buffer
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, a_rows * a_cols * sizeof(T), a, GL_STATIC_DRAW);
            
            // Matrix B buffer
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, b_rows * b_cols * sizeof(T), b, GL_STATIC_DRAW);
            
            // Result buffer
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, a_rows * b_cols * sizeof(T), nullptr, GL_STATIC_DRAW);
            
            // Use the shader
            glUseProgram(program);
            
            // Bind the buffers
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[0]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers[2]);
            
            // Set uniforms
            glUniform1i(glGetUniformLocation(program, "width"), b_cols);
            glUniform1i(glGetUniformLocation(program, "height"), a_rows);
            glUniform1i(glGetUniformLocation(program, "depth"), a_cols);
            
            // Dispatch compute shader
            int work_group_size_x = 32;
            int work_group_size_y = 32;
            
            int num_groups_x = (b_cols + work_group_size_x - 1) / work_group_size_x;
            int num_groups_y = (a_rows + work_group_size_y - 1) / work_group_size_y;
            
            glDispatchCompute(num_groups_x, num_groups_y, 1);
            
            // Wait for compute shader to finish
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // Read back result
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, a_rows * b_cols * sizeof(T), result);
            
            // Cleanup
            glDeleteBuffers(3, buffers);
            
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "GPU matrix multiplication failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Vector addition using GPU (float version)
    static bool addVectorsGPU(const float* a, const float* b, float* result, int length) {
        if (!initialized) detectHardwareCapabilities();
        
        if (!glInitialized) {
            if (!initGL()) return false;
        }
        
        // Create vector operation shaders if they don't exist
        if (floatVecAddProgram == 0) {
            if (!createVectorShaders()) return false;
        }
        
        try {
            // Create buffers
            GLuint buffers[3];
            glGenBuffers(3, buffers);
            
            // Input vector buffers
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), a, GL_STATIC_DRAW);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), b, GL_STATIC_DRAW);
            
            // Result buffer
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), nullptr, GL_STATIC_DRAW);
            
            // Use the program
            glUseProgram(floatVecAddProgram);
            
            // Bind buffers
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[0]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers[2]);
            
            // Set uniform
            glUniform1i(glGetUniformLocation(floatVecAddProgram, "length"), length);
            
            // Dispatch compute shader
            int work_group_size = 256;
            int num_groups = (length + work_group_size - 1) / work_group_size;
            
            glDispatchCompute(num_groups, 1, 1);
            
            // Wait for compute shader to finish
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // Read back result
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, length * sizeof(float), result);
            
            // Cleanup
            glDeleteBuffers(3, buffers);
            
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "GPU vector addition failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Vector element-wise multiplication using GPU (float version)
    static bool multiplyVectorsGPU(const float* a, const float* b, float* result, int length) {
        if (!initialized) detectHardwareCapabilities();
        
        if (!glInitialized) {
            if (!initGL()) return false;
        }
        
        // Create vector operation shaders if they don't exist
        if (floatVecMultProgram == 0) {
            if (!createVectorShaders()) return false;
        }
        
        try {
            // Create buffers
            GLuint buffers[3];
            glGenBuffers(3, buffers);
            
            // Input vector buffers
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), a, GL_STATIC_DRAW);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), b, GL_STATIC_DRAW);
            
            // Result buffer
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, length * sizeof(float), nullptr, GL_STATIC_DRAW);
            
            // Use the program
            glUseProgram(floatVecMultProgram);
            
            // Bind buffers
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[0]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers[2]);
            
            // Set uniform
            glUniform1i(glGetUniformLocation(floatVecMultProgram, "length"), length);
            
            // Dispatch compute shader
            int work_group_size = 256;
            int num_groups = (length + work_group_size - 1) / work_group_size;
            
            glDispatchCompute(num_groups, 1, 1);
            
            // Wait for compute shader to finish
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // Read back result
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, length * sizeof(float), result);
            
            // Cleanup
            glDeleteBuffers(3, buffers);
            
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "GPU vector multiplication failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Matrix-Vector multiplication using GPU (float version)
    static bool multiplyMatrixVectorGPU(
        const float* matrix, int rows, int cols,
        const float* vector, float* result)
    {
        if (!initialized) detectHardwareCapabilities();
        
        if (!glInitialized) {
            if (!initGL()) return false;
        }
        
        // Create a temporary 1-column matrix for the vector
        float* matrix_b = new float[cols * 1];
        for (int i = 0; i < cols; i++) {
            matrix_b[i] = vector[i];
        }
        
        // Use matrix multiplication
        bool success = multiplyMatricesGPU(matrix, rows, cols, matrix_b, cols, 1, result);
        
        delete[] matrix_b;
        return success;
    }
    
    // Multi-threaded matrix multiplication for CPU fallback
    template<typename T>
    static void multiplyMatricesThreaded(
        const T* a, int a_rows, int a_cols,
        const T* b, int b_rows, int b_cols,
        T* result)
    {
        if (!initialized) detectHardwareCapabilities();
        
        // Validate dimensions
        if (a_cols != b_rows) {
            std::cerr << "Matrix dimensions don't match for multiplication" << std::endl;
            return;
        }
        
        // Clear result matrix
        std::memset(result, 0, a_rows * b_cols * sizeof(T));
        
        // Transpose B for better cache locality
        T* b_transposed = new T[b_rows * b_cols];
        for (int i = 0; i < b_rows; i++) {
            for (int j = 0; j < b_cols; j++) {
                b_transposed[j * b_rows + i] = b[i * b_cols + j];
            }
        }
        
        // Calculate optimal number of threads and block size
        int threads_to_use = std::min(numThreads, a_rows);
        int block_size = 32; // Cache-friendly block size
        
        std::vector<std::thread> threads;
        threads.reserve(threads_to_use);
        
        // Define the task for each thread
        auto multiply_block = [&](int start_row, int end_row) {
            for (int i = start_row; i < end_row; i++) {
                for (int j = 0; j < b_cols; j++) {
                    T sum = 0;
                    // Process in cache-friendly blocks
                    for (int k = 0; k < a_cols; k++) {
                        sum += a[i * a_cols + k] * b_transposed[j * b_rows + k];
                    }
                    result[i * b_cols + j] = sum;
                }
            }
        };
        
        // Start threads
        int rows_per_thread = (a_rows + threads_to_use - 1) / threads_to_use;
        for (int t = 0; t < threads_to_use; t++) {
            int start_row = t * rows_per_thread;
            int end_row = std::min(start_row + rows_per_thread, a_rows);
            if (start_row < end_row) {
                threads.emplace_back(multiply_block, start_row, end_row);
            }
        }
        
        // Join threads
        for (auto& thread : threads) {
            thread.join();
        }
        
        delete[] b_transposed;
    }
    
    // Check if GPU operations are recommended for this matrix size
    static bool shouldUseGPU(int size) {
        if (!initialized) detectHardwareCapabilities();
        return glInitialized && size >= optimalGpuMatrixSize;
    }
    
    // Clean up resources
    static void cleanup() {
        if (glInitialized) {
            // Delete programs
            if (floatMatMulProgram != 0) glDeleteProgram(floatMatMulProgram);
            if (doubleMatMulProgram != 0) glDeleteProgram(doubleMatMulProgram);
            if (floatVecAddProgram != 0) glDeleteProgram(floatVecAddProgram);
            if (floatVecMultProgram != 0) glDeleteProgram(floatVecMultProgram);
            
            // Terminate GLFW
            glfwDestroyWindow(window);
            glfwTerminate();
            
            glInitialized = false;
        }
    }
};

// Initialize static members
GLFWwindow* GPUAccelerator::window = nullptr;
GLuint GPUAccelerator::floatMatMulProgram = 0;
GLuint GPUAccelerator::doubleMatMulProgram = 0;
GLuint GPUAccelerator::floatVecAddProgram = 0;
GLuint GPUAccelerator::floatVecMultProgram = 0;
bool GPUAccelerator::glInitialized = false;
std::mutex GPUAccelerator::gl_mutex;
std::atomic<bool> GPUAccelerator::initialized(false);
int GPUAccelerator::maxWorkGroupSizeX = 0;
int GPUAccelerator::maxWorkGroupSizeY = 0;
size_t GPUAccelerator::maxBufferSize = 0;
int GPUAccelerator::optimalGpuMatrixSize = 512; // Default value
int GPUAccelerator::numThreads = 0;
bool GPUAccelerator::hasDoubleSupport = false; 