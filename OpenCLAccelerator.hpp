#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120  // Define OpenCL 1.2 as target version
#endif

#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <type_traits>

// Forward declare vec and matrix templates - don't include them here
template<typename T> class vec;
template<typename T> class matrix;

class OpenCLAccelerator {
private:
    // CPU capabilities
    static bool cpu_capabilities_detected;
    static int cpu_cores;
    static size_t l1_cache_size;
    static int thread_threshold;
    static int cache_block_size;

    // OpenCL objects
    static bool cl_initialized;
    static bool cl_available;
    static cl_platform_id platform;
    static cl_device_id device;
    static cl_context context;
    static cl_command_queue queue;
    static cl_program program;
    
    // Kernels
    static cl_kernel vec_add_float_kernel;
    static cl_kernel vec_add_double_kernel;
    static cl_kernel vec_mult_float_kernel;
    static cl_kernel vec_mult_double_kernel;
    static cl_kernel vec_dot_float_kernel;
    static cl_kernel vec_dot_double_kernel;
    static cl_kernel matmul_float_kernel;
    static cl_kernel matmul_double_kernel;
    
    static std::mutex cl_mutex;
    static std::atomic<bool> initialized;
    static size_t max_work_group_size;
    static size_t max_compute_units;
    static size_t max_mem_alloc_size;
    static bool has_double_support;
    static int gpu_threshold;

    // Detect CPU capabilities
    static void detectCPUCapabilities() {
        if (cpu_capabilities_detected) return;
        
        // Detect number of CPU cores
        cpu_cores = std::thread::hardware_concurrency();
        if (cpu_cores <= 0) cpu_cores = 2; // Default to 2 if detection fails
        
        // Estimate L1 cache size (common sizes: 32KB or 64KB per core)
        l1_cache_size = 32 * 1024; // Default to 32KB
        
        // Set thresholds based on detected capabilities
        thread_threshold = std::max(128, static_cast<int>(std::sqrt(l1_cache_size)));
        cache_block_size = 32; // Default cache block size
        
        cpu_capabilities_detected = true;
    }
    
    // Load OpenCL kernel source
    static std::string loadKernelSource(const char* filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Failed to open kernel file: " << filename << std::endl;
            return "";
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    
    // Initialize OpenCL
    static bool initOpenCL() {
        std::lock_guard<std::mutex> lock(cl_mutex);
        
        if (cl_initialized) return cl_available;
        
        try {
            // Get platform
            cl_uint num_platforms;
            cl_int error = clGetPlatformIDs(0, nullptr, &num_platforms);
            if (error != CL_SUCCESS || num_platforms == 0) {
                std::cerr << "No OpenCL platforms found" << std::endl;
                cl_initialized = true;
                return false;
            }
            
            std::vector<cl_platform_id> platforms(num_platforms);
            error = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to get OpenCL platform IDs" << std::endl;
                cl_initialized = true;
                return false;
            }
            
            // Choose first platform with GPU device
            cl_device_id selected_device = nullptr;
            for (cl_uint i = 0; i < num_platforms; i++) {
                // Try to get GPU device
                cl_uint num_devices;
                error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
                if (error == CL_SUCCESS && num_devices > 0) {
                    std::vector<cl_device_id> devices(num_devices);
                    error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
                    if (error == CL_SUCCESS) {
                        platform = platforms[i];
                        selected_device = devices[0];
                        break;
                    }
                }
            }
            
            // If no GPU found, try to find CPU device
            if (selected_device == nullptr) {
                for (cl_uint i = 0; i < num_platforms; i++) {
                    // Try to get CPU device
                    cl_uint num_devices;
                    error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices);
                    if (error == CL_SUCCESS && num_devices > 0) {
                        std::vector<cl_device_id> devices(num_devices);
                        error = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, num_devices, devices.data(), nullptr);
                        if (error == CL_SUCCESS) {
                            platform = platforms[i];
                            selected_device = devices[0];
                            break;
                        }
                    }
                }
            }
            
            if (selected_device == nullptr) {
                std::cerr << "No OpenCL devices found" << std::endl;
                cl_initialized = true;
                return false;
            }
            
            device = selected_device;
            
            // Create context
            cl_context_properties properties[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
                0
            };
            
            context = clCreateContext(properties, 1, &device, nullptr, nullptr, &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create OpenCL context" << std::endl;
                cl_initialized = true;
                return false;
            }
            
            // Create command queue
            #ifdef CL_VERSION_2_0
            queue = clCreateCommandQueueWithProperties(context, device, nullptr, &error);
            #else
            queue = clCreateCommandQueue(context, device, 0, &error);
            #endif
            
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create OpenCL command queue" << std::endl;
                clReleaseContext(context);
                cl_initialized = true;
                return false;
            }
            
            // Get device capabilities
            clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &max_compute_units, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &max_mem_alloc_size, nullptr);
            
            // Check for double precision support
            size_t ext_size;
            clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &ext_size);
            if (ext_size > 0) {
                std::vector<char> extensions(ext_size);
                clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_size, extensions.data(), nullptr);
                std::string ext_str(extensions.begin(), extensions.end());
                has_double_support = (ext_str.find("cl_khr_fp64") != std::string::npos);
            }
            
            // Set GPU threshold based on capabilities
            gpu_threshold = std::max(256, static_cast<int>(std::sqrt(max_mem_alloc_size / sizeof(float))));
            
            cl_available = true;
            cl_initialized = true;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "OpenCL initialization failed: " << e.what() << std::endl;
            cl_initialized = true;
            return false;
        }
    }
    
    // Create OpenCL kernels
    static bool createKernels() {
        if (!cl_available) return false;
        
        std::lock_guard<std::mutex> lock(cl_mutex);
        
        try {
            // Load kernel source
            std::string source = loadKernelSource("include/opencl/kernels.cl");
            if (source.empty()) {
                std::cerr << "Failed to load OpenCL kernel source" << std::endl;
                return false;
            }
            
            const char* source_ptr = source.c_str();
            size_t source_size = source.size();
            
            // Create program
            cl_int error;
            program = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create OpenCL program" << std::endl;
                return false;
            }
            
            // Build program
            error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                // Get build log
                size_t log_size;
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
                std::vector<char> log(log_size);
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
                std::cerr << "OpenCL program build failed: " << std::string(log.begin(), log.end()) << std::endl;
                clReleaseProgram(program);
                return false;
            }
            
            // Create kernels
            vec_add_float_kernel = clCreateKernel(program, "vec_add_float", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create vec_add_float kernel" << std::endl;
                clReleaseProgram(program);
                return false;
            }
            
            vec_mult_float_kernel = clCreateKernel(program, "vec_mult_float", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create vec_mult_float kernel" << std::endl;
                clReleaseKernel(vec_add_float_kernel);
                clReleaseProgram(program);
                return false;
            }
            
            vec_dot_float_kernel = clCreateKernel(program, "vec_dot_float", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create vec_dot_float kernel" << std::endl;
                clReleaseKernel(vec_add_float_kernel);
                clReleaseKernel(vec_mult_float_kernel);
                clReleaseProgram(program);
                return false;
            }
            
            matmul_float_kernel = clCreateKernel(program, "matmul_float", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create matmul_float kernel" << std::endl;
                clReleaseKernel(vec_add_float_kernel);
                clReleaseKernel(vec_mult_float_kernel);
                clReleaseKernel(vec_dot_float_kernel);
                clReleaseProgram(program);
                return false;
            }
            
            // Create double precision kernels if supported
            if (has_double_support) {
                vec_add_double_kernel = clCreateKernel(program, "vec_add_double", &error);
                if (error != CL_SUCCESS) {
                    std::cerr << "Failed to create vec_add_double kernel" << std::endl;
                    has_double_support = false;
                }
                
                vec_mult_double_kernel = clCreateKernel(program, "vec_mult_double", &error);
                if (error != CL_SUCCESS) {
                    std::cerr << "Failed to create vec_mult_double kernel" << std::endl;
                    has_double_support = false;
                }
                
                vec_dot_double_kernel = clCreateKernel(program, "vec_dot_double", &error);
                if (error != CL_SUCCESS) {
                    std::cerr << "Failed to create vec_dot_double kernel" << std::endl;
                    has_double_support = false;
                }
                
                matmul_double_kernel = clCreateKernel(program, "matmul_double", &error);
                if (error != CL_SUCCESS) {
                    std::cerr << "Failed to create matmul_double kernel" << std::endl;
                    has_double_support = false;
                }
            }
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to create OpenCL kernels: " << e.what() << std::endl;
            return false;
        }
    }

public:
    // Initialize hardware acceleration
    static bool initialize() {
        if (initialized) return true;
        
        // Detect CPU capabilities
        detectCPUCapabilities();
        
        // Try to initialize OpenCL
        if (initOpenCL()) {
            // Create kernels
            createKernels();
        }
        
        initialized = true;
        return true;
    }
    
    // Check if OpenCL is available
    static bool isOpenCLAvailable() {
        if (!initialized) initialize();
        return cl_available;
    }
    
    // Get CPU cores
    static int getCPUCores() {
        if (!cpu_capabilities_detected) detectCPUCapabilities();
        return cpu_cores;
    }
    
    // Get L1 cache size
    static size_t getL1CacheSize() {
        if (!cpu_capabilities_detected) detectCPUCapabilities();
        return l1_cache_size;
    }
    
    // Get thread threshold
    static int getThreadThreshold() {
        if (!cpu_capabilities_detected) detectCPUCapabilities();
        return thread_threshold;
    }
    
    // Get cache block size
    static int getCacheBlockSize() {
        if (!cpu_capabilities_detected) detectCPUCapabilities();
        return cache_block_size;
    }
    
    // Get GPU threshold
    static int getGPUThreshold() {
        if (!initialized) initialize();
        return gpu_threshold;
    }
    
    // Check if we should use GPU
    static bool shouldUseGPU(int size) {
        if (!initialized) initialize();
        return cl_available && size >= gpu_threshold;
    }
    
    // Check if we should use threading
    static bool shouldUseThreading(int size) {
        if (!cpu_capabilities_detected) detectCPUCapabilities();
        return size >= thread_threshold;
    }
    
    // Vector addition - float version
    static bool addVectors(const float* a, const float* b, float* result, int length) {
        if (!cl_available) return false;
        
        std::lock_guard<std::mutex> lock(cl_mutex);
        
        try {
            cl_int error;
            
            // Create buffers
            cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)a, &error);
            if (error != CL_SUCCESS) return false;
            
            cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)b, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                return false;
            }
            
            cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(float) * length, nullptr, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                return false;
            }
            
            // Set kernel arguments
            error = clSetKernelArg(vec_add_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_add_float_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_add_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_add_float_kernel, 3, sizeof(int), &length);
            
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Determine global and local work sizes
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;
            
            // Execute kernel
            error = clEnqueueNDRangeKernel(queue, vec_add_float_kernel, 1, nullptr,
                                          &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                                       sizeof(float) * length, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Clean up
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "OpenCL vector addition failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Vector addition - double version
    static bool addVectors(const double* a, const double* b, double* result, int length) {
        if (!cl_available || !has_double_support) return false;
        
        std::lock_guard<std::mutex> lock(cl_mutex);
        
        try {
            cl_int error;
            
            // Create buffers
            cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(double) * length, (void*)a, &error);
            if (error != CL_SUCCESS) return false;
            
            cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(double) * length, (void*)b, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                return false;
            }
            
            cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(double) * length, nullptr, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                return false;
            }
            
            // Set kernel arguments
            error = clSetKernelArg(vec_add_double_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_add_double_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_add_double_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_add_double_kernel, 3, sizeof(int), &length);
            
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Determine global and local work sizes
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;
            
            // Execute kernel
            error = clEnqueueNDRangeKernel(queue, vec_add_double_kernel, 1, nullptr,
                                          &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                                       sizeof(double) * length, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Clean up
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "OpenCL vector addition failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Vector multiplication - float version
    static bool multiplyVectors(const float* a, const float* b, float* result, int length) {
        if (!cl_available) return false;
        
        std::lock_guard<std::mutex> lock(cl_mutex);
        
        try {
            cl_int error;
            
            // Create buffers
            cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)a, &error);
            if (error != CL_SUCCESS) return false;
            
            cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)b, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                return false;
            }
            
            cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(float) * length, nullptr, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                return false;
            }
            
            // Set kernel arguments
            error = clSetKernelArg(vec_mult_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_mult_float_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_mult_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_mult_float_kernel, 3, sizeof(int), &length);
            
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Determine global and local work sizes
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;
            
            // Execute kernel
            error = clEnqueueNDRangeKernel(queue, vec_mult_float_kernel, 1, nullptr,
                                          &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                                       sizeof(float) * length, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Clean up
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "OpenCL vector multiplication failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Vector multiplication - double version
    static bool multiplyVectors(const double* a, const double* b, double* result, int length) {
        if (!cl_available || !has_double_support) return false;
        
        std::lock_guard<std::mutex> lock(cl_mutex);
        
        try {
            cl_int error;
            
            // Create buffers
            cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(double) * length, (void*)a, &error);
            if (error != CL_SUCCESS) return false;
            
            cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(double) * length, (void*)b, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                return false;
            }
            
            cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(double) * length, nullptr, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                return false;
            }
            
            // Set kernel arguments
            error = clSetKernelArg(vec_mult_double_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_mult_double_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_mult_double_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_mult_double_kernel, 3, sizeof(int), &length);
            
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Determine global and local work sizes
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;
            
            // Execute kernel
            error = clEnqueueNDRangeKernel(queue, vec_mult_double_kernel, 1, nullptr,
                                          &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                                       sizeof(double) * length, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Clean up
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "OpenCL vector multiplication failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Dot product - float version
    static bool dotProduct(const float* a, const float* b, int length, float& result) {
        if (!cl_available) return false;
        
        std::lock_guard<std::mutex> lock(cl_mutex);
        
        try {
            cl_int error;
            
            // Create buffers
            cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)a, &error);
            if (error != CL_SUCCESS) return false;
            
            cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)b, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                return false;
            }
            
            cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                sizeof(float), nullptr, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                return false;
            }
            
            // Initialize result to 0
            float zero = 0.0f;
            error = clEnqueueWriteBuffer(queue, result_buffer, CL_TRUE, 0,
                                       sizeof(float), &zero, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Determine local size
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;
            
            // Create local memory for reduction
            cl_int local_mem_size = sizeof(float) * local_size;
            
            // Set kernel arguments
            error = clSetKernelArg(vec_dot_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_dot_float_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_dot_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_dot_float_kernel, 3, sizeof(int), &length);
            error |= clSetKernelArg(vec_dot_float_kernel, 4, local_mem_size, nullptr);
            
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Execute kernel
            error = clEnqueueNDRangeKernel(queue, vec_dot_float_kernel, 1, nullptr,
                                          &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                                       sizeof(float), &result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Clean up
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "OpenCL dot product failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Dot product - double version
    static bool dotProduct(const double* a, const double* b, int length, double& result) {
        if (!cl_available || !has_double_support) return false;
        
        std::lock_guard<std::mutex> lock(cl_mutex);
        
        try {
            cl_int error;
            
            // Create buffers
            cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(double) * length, (void*)a, &error);
            if (error != CL_SUCCESS) return false;
            
            cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(double) * length, (void*)b, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                return false;
            }
            
            cl_mem result_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                sizeof(double), nullptr, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                return false;
            }
            
            // Initialize result to 0
            double zero = 0.0;
            error = clEnqueueWriteBuffer(queue, result_buffer, CL_TRUE, 0,
                                       sizeof(double), &zero, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Determine local size
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;
            
            // Create local memory for reduction
            cl_int local_mem_size = sizeof(double) * local_size;
            
            // Set kernel arguments
            error = clSetKernelArg(vec_dot_double_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_dot_double_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_dot_double_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_dot_double_kernel, 3, sizeof(int), &length);
            error |= clSetKernelArg(vec_dot_double_kernel, 4, local_mem_size, nullptr);
            
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Execute kernel
            error = clEnqueueNDRangeKernel(queue, vec_dot_double_kernel, 1, nullptr,
                                          &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                                       sizeof(double), &result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Clean up
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "OpenCL dot product failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Matrix-matrix multiplication - float version
    static bool multiplyMatrices(const float* a, int a_rows, int a_cols,
                               const float* b, int b_rows, int b_cols,
                               float* result) {
        if (!cl_available) return false;
        if (a_cols != b_rows) return false;
        
        std::lock_guard<std::mutex> lock(cl_mutex);
        
        try {
            cl_int error;
            
            // Create buffers
            cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * a_rows * a_cols, (void*)a, &error);
            if (error != CL_SUCCESS) return false;
            
            cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * b_rows * b_cols, (void*)b, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                return false;
            }
            
            cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(float) * a_rows * b_cols, nullptr, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                return false;
            }
            
            // Determine tile size for local memory (usually 16x16 or 32x32)
            size_t tile_size = 16;
            size_t local_size[2] = {tile_size, tile_size};
            size_t global_size[2] = {
                ((b_cols + tile_size - 1) / tile_size) * tile_size,
                ((a_rows + tile_size - 1) / tile_size) * tile_size
            };
            
            // Set kernel arguments
            error = clSetKernelArg(matmul_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(matmul_float_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(matmul_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(matmul_float_kernel, 3, sizeof(int), &a_rows);
            error |= clSetKernelArg(matmul_float_kernel, 4, sizeof(int), &a_cols);
            error |= clSetKernelArg(matmul_float_kernel, 5, sizeof(int), &b_cols);
            error |= clSetKernelArg(matmul_float_kernel, 6, sizeof(float) * tile_size * tile_size, nullptr);
            error |= clSetKernelArg(matmul_float_kernel, 7, sizeof(float) * tile_size * tile_size, nullptr);
            
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Execute kernel
            error = clEnqueueNDRangeKernel(queue, matmul_float_kernel, 2, nullptr,
                                          global_size, local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                                       sizeof(float) * a_rows * b_cols, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Clean up
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "OpenCL matrix multiplication failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Matrix-matrix multiplication - double version
    static bool multiplyMatrices(const double* a, int a_rows, int a_cols,
                               const double* b, int b_rows, int b_cols,
                               double* result) {
        if (!cl_available || !has_double_support) return false;
        if (a_cols != b_rows) return false;
        
        std::lock_guard<std::mutex> lock(cl_mutex);
        
        try {
            cl_int error;
            
            // Create buffers
            cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(double) * a_rows * a_cols, (void*)a, &error);
            if (error != CL_SUCCESS) return false;
            
            cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(double) * b_rows * b_cols, (void*)b, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                return false;
            }
            
            cl_mem result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(double) * a_rows * b_cols, nullptr, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                return false;
            }
            
            // Determine tile size for local memory (usually 16x16 or 32x32)
            size_t tile_size = 16;
            size_t local_size[2] = {tile_size, tile_size};
            size_t global_size[2] = {
                ((b_cols + tile_size - 1) / tile_size) * tile_size,
                ((a_rows + tile_size - 1) / tile_size) * tile_size
            };
            
            // Set kernel arguments
            error = clSetKernelArg(matmul_double_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(matmul_double_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(matmul_double_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(matmul_double_kernel, 3, sizeof(int), &a_rows);
            error |= clSetKernelArg(matmul_double_kernel, 4, sizeof(int), &a_cols);
            error |= clSetKernelArg(matmul_double_kernel, 5, sizeof(int), &b_cols);
            error |= clSetKernelArg(matmul_double_kernel, 6, sizeof(double) * tile_size * tile_size, nullptr);
            error |= clSetKernelArg(matmul_double_kernel, 7, sizeof(double) * tile_size * tile_size, nullptr);
            
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Execute kernel
            error = clEnqueueNDRangeKernel(queue, matmul_double_kernel, 2, nullptr,
                                          global_size, local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                                       sizeof(double) * a_rows * b_cols, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(a_buffer);
                clReleaseMemObject(b_buffer);
                clReleaseMemObject(result_buffer);
                return false;
            }
            
            // Clean up
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "OpenCL matrix multiplication failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Matrix-vector multiplication - float version
    static bool multiplyMatrixVector(const float* matrix, int rows, int cols,
                                   const float* vector, float* result) {
        if (!cl_available) return false;
        
        // Use the matrix-matrix multiplication function for simplicity
        return multiplyMatrices(matrix, rows, cols, vector, cols, 1, result);
    }
    
    // Matrix-vector multiplication - double version
    static bool multiplyMatrixVector(const double* matrix, int rows, int cols,
                                   const double* vector, double* result) {
        if (!cl_available || !has_double_support) return false;
        
        // Use the matrix-matrix multiplication function for simplicity
        return multiplyMatrices(matrix, rows, cols, vector, cols, 1, result);
    }
    
    // Clean up resources
    static void cleanup() {
        if (cl_initialized && cl_available) {
            std::lock_guard<std::mutex> lock(cl_mutex);
            
            if (vec_add_float_kernel) {
                clReleaseKernel(vec_add_float_kernel);
                vec_add_float_kernel = nullptr;
            }
            
            if (vec_mult_float_kernel) {
                clReleaseKernel(vec_mult_float_kernel);
                vec_mult_float_kernel = nullptr;
            }
            
            if (vec_dot_float_kernel) {
                clReleaseKernel(vec_dot_float_kernel);
                vec_dot_float_kernel = nullptr;
            }
            
            if (matmul_float_kernel) {
                clReleaseKernel(matmul_float_kernel);
                matmul_float_kernel = nullptr;
            }
            
            if (has_double_support) {
                if (vec_add_double_kernel) {
                    clReleaseKernel(vec_add_double_kernel);
                    vec_add_double_kernel = nullptr;
                }
                
                if (vec_mult_double_kernel) {
                    clReleaseKernel(vec_mult_double_kernel);
                    vec_mult_double_kernel = nullptr;
                }
                
                if (vec_dot_double_kernel) {
                    clReleaseKernel(vec_dot_double_kernel);
                    vec_dot_double_kernel = nullptr;
                }
                
                if (matmul_double_kernel) {
                    clReleaseKernel(matmul_double_kernel);
                    matmul_double_kernel = nullptr;
                }
            }
            
            if (program) {
                clReleaseProgram(program);
                program = nullptr;
            }
            
            if (queue) {
                clReleaseCommandQueue(queue);
                queue = nullptr;
            }
            
            if (context) {
                clReleaseContext(context);
                context = nullptr;
            }
            
            cl_available = false;
        }
    }
};

// Initialize static members
bool OpenCLAccelerator::cpu_capabilities_detected = false;
int OpenCLAccelerator::cpu_cores = 0;
size_t OpenCLAccelerator::l1_cache_size = 0;
int OpenCLAccelerator::thread_threshold = 128;  // Default value
int OpenCLAccelerator::cache_block_size = 32;   // Default value

bool OpenCLAccelerator::cl_initialized = false;
bool OpenCLAccelerator::cl_available = false;
cl_platform_id OpenCLAccelerator::platform = nullptr;
cl_device_id OpenCLAccelerator::device = nullptr;
cl_context OpenCLAccelerator::context = nullptr;
cl_command_queue OpenCLAccelerator::queue = nullptr;
cl_program OpenCLAccelerator::program = nullptr;

cl_kernel OpenCLAccelerator::vec_add_float_kernel = nullptr;
cl_kernel OpenCLAccelerator::vec_add_double_kernel = nullptr;
cl_kernel OpenCLAccelerator::vec_mult_float_kernel = nullptr;
cl_kernel OpenCLAccelerator::vec_mult_double_kernel = nullptr;
cl_kernel OpenCLAccelerator::vec_dot_float_kernel = nullptr;
cl_kernel OpenCLAccelerator::vec_dot_double_kernel = nullptr;
cl_kernel OpenCLAccelerator::matmul_float_kernel = nullptr;
cl_kernel OpenCLAccelerator::matmul_double_kernel = nullptr;

std::mutex OpenCLAccelerator::cl_mutex;
std::atomic<bool> OpenCLAccelerator::initialized(false);
size_t OpenCLAccelerator::max_work_group_size = 0;
size_t OpenCLAccelerator::max_compute_units = 0;
size_t OpenCLAccelerator::max_mem_alloc_size = 0;
bool OpenCLAccelerator::has_double_support = false;
int OpenCLAccelerator::gpu_threshold = 256;  // Default value 