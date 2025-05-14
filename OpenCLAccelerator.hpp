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
    static cl_kernel vec_sub_float_kernel;
    static cl_kernel vec_mult_float_kernel;
    static cl_kernel vec_dot_float_kernel;
    static cl_kernel matmul_float_kernel;
    static cl_kernel outer_prod_float_kernel;
    static cl_kernel prep_vec_float_kernel;
    static cl_kernel vec_scale_float_kernel;
    
    static std::mutex cl_mutex;
    static std::atomic<bool> initialized;
    static size_t max_work_group_size;
    static size_t max_compute_units;
    static size_t max_mem_alloc_size;
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
                std::cerr << "Failed to create vec_add_float kernel: " << error << std::endl;
                clReleaseProgram(program);
                return false;
            }
            
            vec_sub_float_kernel = clCreateKernel(program, "vec_sub_float", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create vec_sub_float kernel: " << error << std::endl;
                if (vec_add_float_kernel) clReleaseKernel(vec_add_float_kernel);
                clReleaseProgram(program);
                return false;
            }
            
            vec_mult_float_kernel = clCreateKernel(program, "vec_mult_float", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create vec_mult_float kernel: " << error << std::endl;
                if (vec_add_float_kernel) clReleaseKernel(vec_add_float_kernel);
                if (vec_sub_float_kernel) clReleaseKernel(vec_sub_float_kernel);
                clReleaseProgram(program);
                return false;
            }
            
            vec_scale_float_kernel = clCreateKernel(program, "vec_scale_float", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create vec_scale_float kernel: " << error << std::endl;
                if (vec_add_float_kernel) clReleaseKernel(vec_add_float_kernel);
                if (vec_sub_float_kernel) clReleaseKernel(vec_sub_float_kernel);
                if (vec_mult_float_kernel) clReleaseKernel(vec_mult_float_kernel);
                clReleaseProgram(program);
                return false;
            }
            
            vec_dot_float_kernel = clCreateKernel(program, "vec_dot_float", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create vec_dot_float kernel: " << error << std::endl;
                if (vec_add_float_kernel) clReleaseKernel(vec_add_float_kernel);
                if (vec_sub_float_kernel) clReleaseKernel(vec_sub_float_kernel);
                if (vec_mult_float_kernel) clReleaseKernel(vec_mult_float_kernel);
                if (vec_scale_float_kernel) clReleaseKernel(vec_scale_float_kernel);
                clReleaseProgram(program);
                return false;
            }
            
            matmul_float_kernel = clCreateKernel(program, "matmul_float", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create matmul_float kernel: " << error << std::endl;
                if (vec_add_float_kernel) clReleaseKernel(vec_add_float_kernel);
                if (vec_sub_float_kernel) clReleaseKernel(vec_sub_float_kernel);
                if (vec_mult_float_kernel) clReleaseKernel(vec_mult_float_kernel);
                if (vec_scale_float_kernel) clReleaseKernel(vec_scale_float_kernel);
                if (vec_dot_float_kernel) clReleaseKernel(vec_dot_float_kernel);
                clReleaseProgram(program);
                return false;
            }
            
            outer_prod_float_kernel = clCreateKernel(program, "outer_prod_float", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create outer_prod_float kernel: " << error << std::endl;
                if (vec_add_float_kernel) clReleaseKernel(vec_add_float_kernel);
                if (vec_sub_float_kernel) clReleaseKernel(vec_sub_float_kernel);
                if (vec_mult_float_kernel) clReleaseKernel(vec_mult_float_kernel);
                if (vec_scale_float_kernel) clReleaseKernel(vec_scale_float_kernel);
                if (vec_dot_float_kernel) clReleaseKernel(vec_dot_float_kernel);
                if (matmul_float_kernel) clReleaseKernel(matmul_float_kernel);
                clReleaseProgram(program);
                return false;
            }
            
            // Create prep_vec kernels
            prep_vec_float_kernel = clCreateKernel(program, "prep_vec_kernel_float", &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create prep_vec float kernel: " << error << std::endl;
                if (vec_add_float_kernel) clReleaseKernel(vec_add_float_kernel);
                if (vec_sub_float_kernel) clReleaseKernel(vec_sub_float_kernel);
                if (vec_mult_float_kernel) clReleaseKernel(vec_mult_float_kernel);
                if (vec_scale_float_kernel) clReleaseKernel(vec_scale_float_kernel);
                if (vec_dot_float_kernel) clReleaseKernel(vec_dot_float_kernel);
                if (matmul_float_kernel) clReleaseKernel(matmul_float_kernel);
                if (outer_prod_float_kernel) clReleaseKernel(outer_prod_float_kernel);
                clReleaseProgram(program);
                return false;
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
    
    // Vector subtraction - float version
    static bool subtractVectors(const float* a, const float* b, float* result, int length) {
        std::lock_guard<std::mutex> lock(cl_mutex);
        if (!cl_available || !shouldUseGPU(length)) return false;

        cl_int error = CL_SUCCESS;
        cl_mem a_buffer = nullptr;
        cl_mem b_buffer = nullptr;
        cl_mem result_buffer = nullptr;

        try {
            // Create buffers
            a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)a, &error);
            if (error != CL_SUCCESS) { std::cerr << "subtractVectors: Failed to create a_buffer: " << error << std::endl; throw; }

            b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)b, &error);
            if (error != CL_SUCCESS) { std::cerr << "subtractVectors: Failed to create b_buffer: " << error << std::endl; throw; }

            result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(float) * length, nullptr, &error);
            if (error != CL_SUCCESS) { std::cerr << "subtractVectors: Failed to create result_buffer: " << error << std::endl; throw; }

            // Set kernel arguments
            error = clSetKernelArg(vec_sub_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_sub_float_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_sub_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_sub_float_kernel, 3, sizeof(int), &length);
            if (error != CL_SUCCESS) { std::cerr << "subtractVectors: Failed to set kernel args: " << error << std::endl; throw; }

            // Determine global and local work sizes
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;

            // Execute kernel
            error = clEnqueueNDRangeKernel(queue, vec_sub_float_kernel, 1, nullptr,
                                 &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "subtractVectors: Failed to enqueue kernel: " << error << std::endl; throw; }

            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                              sizeof(float) * length, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "subtractVectors: Failed to read buffer: " << error << std::endl; throw; }

            // Cleanup
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);

            return true;
        } catch (...) {
            if (a_buffer) clReleaseMemObject(a_buffer);
            if (b_buffer) clReleaseMemObject(b_buffer);
            if (result_buffer) clReleaseMemObject(result_buffer);
            return false;
        }
    }

    static bool scaleVector(const float* a, float scalar, float* result, int length) {
        std::lock_guard<std::mutex> lock(cl_mutex);
        if (!cl_available || !shouldUseGPU(length)) return false;

        cl_int error = CL_SUCCESS;
        cl_mem a_buffer = nullptr;
        cl_mem result_buffer = nullptr;

        try {
            // Create buffers
            a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)a, &error);
            if (error != CL_SUCCESS) { std::cerr << "scaleVector: Failed to create a_buffer: " << error << std::endl; throw; }
            
            result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(float) * length, nullptr, &error);
            if (error != CL_SUCCESS) { std::cerr << "scaleVector: Failed to create result_buffer: " << error << std::endl; throw; }

            // Set kernel arguments
            error = clSetKernelArg(vec_scale_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_scale_float_kernel, 1, sizeof(float), &scalar);
            error |= clSetKernelArg(vec_scale_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_scale_float_kernel, 3, sizeof(int), &length);
            if (error != CL_SUCCESS) { std::cerr << "scaleVector: Failed to set kernel args: " << error << std::endl; throw; }

            // Determine global and local work sizes
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;

            // Execute kernel
            error = clEnqueueNDRangeKernel(queue, vec_scale_float_kernel, 1, nullptr,
                                 &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "scaleVector: Failed to enqueue kernel: " << error << std::endl; throw; }

            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                              sizeof(float) * length, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "scaleVector: Failed to read buffer: " << error << std::endl; throw; }

            // Cleanup
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(result_buffer);

            return true;
        } catch (...) {
            if (a_buffer) clReleaseMemObject(a_buffer);
            if (result_buffer) clReleaseMemObject(result_buffer);
            return false;
        }
    }

    static bool elementMultiply(const float* a, const float* b, float* result, int length) {
        std::lock_guard<std::mutex> lock(cl_mutex);
        if (!cl_available || !shouldUseGPU(length)) return false;

        cl_int error = CL_SUCCESS;
        cl_mem a_buffer = nullptr;
        cl_mem b_buffer = nullptr;
        cl_mem result_buffer = nullptr;
        
        try {
            // Create buffers
            a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)a, &error);
            if (error != CL_SUCCESS) { std::cerr << "elementMultiply: Failed to create a_buffer: " << error << std::endl; throw; }

            b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)b, &error);
            if (error != CL_SUCCESS) { std::cerr << "elementMultiply: Failed to create b_buffer: " << error << std::endl; throw; }
            
            result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(float) * length, nullptr, &error);
            if (error != CL_SUCCESS) { std::cerr << "elementMultiply: Failed to create result_buffer: " << error << std::endl; throw; }

            // Set kernel arguments
            error = clSetKernelArg(vec_mult_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_mult_float_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_mult_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_mult_float_kernel, 3, sizeof(int), &length);
            if (error != CL_SUCCESS) { std::cerr << "elementMultiply: Failed to set kernel args: " << error << std::endl; throw; }
            
            // Determine global and local work sizes
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;

            // Execute kernel
            error = clEnqueueNDRangeKernel(queue, vec_mult_float_kernel, 1, nullptr,
                                 &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "elementMultiply: Failed to enqueue kernel: " << error << std::endl; throw; }

            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                              sizeof(float) * length, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "elementMultiply: Failed to read buffer: " << error << std::endl; throw; }

            // Cleanup
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
        } catch (...) {
            if (a_buffer) clReleaseMemObject(a_buffer);
            if (b_buffer) clReleaseMemObject(b_buffer);
            if (result_buffer) clReleaseMemObject(result_buffer);
            return false;
        }
    }

    // Add new functions for matrix operations
    static bool addMatrices(const float* a, const float* b, float* result, int rows, int cols) {
        std::lock_guard<std::mutex> lock(cl_mutex);
        const int length = rows * cols;
        if (!cl_available || !shouldUseGPU(length)) return false;
        
        cl_int error = CL_SUCCESS;
        cl_mem a_buffer = nullptr;
        cl_mem b_buffer = nullptr;
        cl_mem result_buffer = nullptr;

        try {
            // Create buffers
            a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)a, &error);
            if (error != CL_SUCCESS) { std::cerr << "addMatrices: Failed to create a_buffer: " << error << std::endl; throw; }

            b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)b, &error);
            if (error != CL_SUCCESS) { std::cerr << "addMatrices: Failed to create b_buffer: " << error << std::endl; throw; }
            
            result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(float) * length, nullptr, &error);
            if (error != CL_SUCCESS) { std::cerr << "addMatrices: Failed to create result_buffer: " << error << std::endl; throw; }
            
            // Set kernel arguments
            error = clSetKernelArg(vec_add_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_add_float_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_add_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_add_float_kernel, 3, sizeof(int), &length);
            if (error != CL_SUCCESS) { std::cerr << "addMatrices: Failed to set kernel args: " << error << std::endl; throw; }
            
            // Execute kernel
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;
            error = clEnqueueNDRangeKernel(queue, vec_add_float_kernel, 1, nullptr,
                                 &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "addMatrices: Failed to enqueue kernel: " << error << std::endl; throw; }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                              sizeof(float) * length, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "addMatrices: Failed to read buffer: " << error << std::endl; throw; }
            
            // Cleanup
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
        } catch (...) {
            if (a_buffer) clReleaseMemObject(a_buffer);
            if (b_buffer) clReleaseMemObject(b_buffer);
            if (result_buffer) clReleaseMemObject(result_buffer);
            return false;
        }
    }

    static bool subtractMatrices(const float* a, const float* b, float* result, int rows, int cols) {
        std::lock_guard<std::mutex> lock(cl_mutex);
        const int length = rows * cols;
        if (!cl_available || !shouldUseGPU(length)) return false;
        
        cl_int error = CL_SUCCESS;
        cl_mem a_buffer = nullptr;
        cl_mem b_buffer = nullptr;
        cl_mem result_buffer = nullptr;

        try {
            // Create buffers
            a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)a, &error);
            if (error != CL_SUCCESS) { std::cerr << "subtractMatrices: Failed to create a_buffer: " << error << std::endl; throw; }

            b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)b, &error);
            if (error != CL_SUCCESS) { std::cerr << "subtractMatrices: Failed to create b_buffer: " << error << std::endl; throw; }
            
            result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(float) * length, nullptr, &error);
            if (error != CL_SUCCESS) { std::cerr << "subtractMatrices: Failed to create result_buffer: " << error << std::endl; throw; }
            
            // Set kernel arguments
            error = clSetKernelArg(vec_sub_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_sub_float_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_sub_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_sub_float_kernel, 3, sizeof(int), &length);
            if (error != CL_SUCCESS) { std::cerr << "subtractMatrices: Failed to set kernel args: " << error << std::endl; throw; }
            
            // Execute kernel
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;
            error = clEnqueueNDRangeKernel(queue, vec_sub_float_kernel, 1, nullptr,
                                 &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "subtractMatrices: Failed to enqueue kernel: " << error << std::endl; throw; }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                              sizeof(float) * length, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "subtractMatrices: Failed to read buffer: " << error << std::endl; throw; }
            
            // Cleanup
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
        } catch (...) {
            if (a_buffer) clReleaseMemObject(a_buffer);
            if (b_buffer) clReleaseMemObject(b_buffer);
            if (result_buffer) clReleaseMemObject(result_buffer);
            return false;
        }
    }

    static bool scaleMatrix(const float* a, float scalar, float* result, int rows, int cols) {
        std::lock_guard<std::mutex> lock(cl_mutex);
        const int length = rows * cols;
        if (!cl_available || !shouldUseGPU(length)) return false;
        
        cl_int error = CL_SUCCESS;
        cl_mem a_buffer = nullptr;
        cl_mem result_buffer = nullptr;

        try {
            // Create buffers
            a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)a, &error);
            if (error != CL_SUCCESS) { std::cerr << "scaleMatrix: Failed to create a_buffer: " << error << std::endl; throw; }
            
            result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(float) * length, nullptr, &error);
            if (error != CL_SUCCESS) { std::cerr << "scaleMatrix: Failed to create result_buffer: " << error << std::endl; throw; }
            
            // Set kernel arguments
            error = clSetKernelArg(vec_scale_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_scale_float_kernel, 1, sizeof(float), &scalar);
            error |= clSetKernelArg(vec_scale_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_scale_float_kernel, 3, sizeof(int), &length);
             if (error != CL_SUCCESS) { std::cerr << "scaleMatrix: Failed to set kernel args: " << error << std::endl; throw; }
            
            // Execute kernel
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;
            error = clEnqueueNDRangeKernel(queue, vec_scale_float_kernel, 1, nullptr,
                                 &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "scaleMatrix: Failed to enqueue kernel: " << error << std::endl; throw; }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                              sizeof(float) * length, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "scaleMatrix: Failed to read buffer: " << error << std::endl; throw; }
            
            // Cleanup
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
        } catch (...) {
            if (a_buffer) clReleaseMemObject(a_buffer);
            if (result_buffer) clReleaseMemObject(result_buffer);
            return false;
        }
    }

    static bool hadamardProduct(const float* a, const float* b, float* result, int rows, int cols) {
        std::lock_guard<std::mutex> lock(cl_mutex);
        const int length = rows * cols;
        if (!cl_available || !shouldUseGPU(length)) return false;
        
        cl_int error = CL_SUCCESS;
        cl_mem a_buffer = nullptr;
        cl_mem b_buffer = nullptr;
        cl_mem result_buffer = nullptr;

        try {
            // Create buffers
            a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)a, &error);
            if (error != CL_SUCCESS) { std::cerr << "hadamardProduct: Failed to create a_buffer: " << error << std::endl; throw; }

            b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           sizeof(float) * length, (void*)b, &error);
            if (error != CL_SUCCESS) { std::cerr << "hadamardProduct: Failed to create b_buffer: " << error << std::endl; throw; }
            
            result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                sizeof(float) * length, nullptr, &error);
            if (error != CL_SUCCESS) { std::cerr << "hadamardProduct: Failed to create result_buffer: " << error << std::endl; throw; }
            
            // Set kernel arguments
            error = clSetKernelArg(vec_mult_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_mult_float_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_mult_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(vec_mult_float_kernel, 3, sizeof(int), &length);
            if (error != CL_SUCCESS) { std::cerr << "hadamardProduct: Failed to set kernel args: " << error << std::endl; throw; }
            
            // Execute kernel
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            size_t global_size = ((length + local_size - 1) / local_size) * local_size;
            error = clEnqueueNDRangeKernel(queue, vec_mult_float_kernel, 1, nullptr,
                                 &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "hadamardProduct: Failed to enqueue kernel: " << error << std::endl; throw; }
            
            // Read result
            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0,
                              sizeof(float) * length, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "hadamardProduct: Failed to read buffer: " << error << std::endl; throw; }
            
            // Cleanup
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(result_buffer);
            
            return true;
        } catch (...) {
            if (a_buffer) clReleaseMemObject(a_buffer);
            if (b_buffer) clReleaseMemObject(b_buffer);
            if (result_buffer) clReleaseMemObject(result_buffer);
            return false;
        }
    }

    // Dot product - float version
    static bool dotProduct(const float* a, const float* b, int length, float& result) {
        std::lock_guard<std::mutex> lock(cl_mutex);
        if (!cl_available || !shouldUseGPU(length)) return false;

        cl_int error = CL_SUCCESS;
        cl_mem a_buffer = nullptr;
        cl_mem b_buffer = nullptr;
        cl_mem partial_results_buffer = nullptr;
        float* partial_results_host = nullptr;

        try {
            size_t local_size = std::min(max_work_group_size, (size_t)256);
            // Ensure global_size is a multiple of local_size and covers at least length elements
            // More accurately, num_groups should be (length + local_size -1) / local_size
            // but vec_dot_float kernel is designed such that get_global_id(0) is used to check id < length.
            // The number of groups determines the size of partial_results.
            size_t num_groups = (length + local_size - 1) / local_size;
            size_t global_size = num_groups * local_size;

            if (num_groups == 0) { // Handle length == 0 case
                result = 0.0f;
                return true;
            }
            
            partial_results_host = new float[num_groups];

            a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * length, (void*)a, &error);
            if (error != CL_SUCCESS) { std::cerr << "dotProduct: Failed to create a_buffer: " << error << std::endl; throw; }

            b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * length, (void*)b, &error);
            if (error != CL_SUCCESS) { std::cerr << "dotProduct: Failed to create b_buffer: " << error << std::endl; throw; }

            partial_results_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * num_groups, nullptr, &error);
            if (error != CL_SUCCESS) { std::cerr << "dotProduct: Failed to create partial_results_buffer: " << error << std::endl; throw; }

            error = clSetKernelArg(vec_dot_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(vec_dot_float_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(vec_dot_float_kernel, 2, sizeof(cl_mem), &partial_results_buffer);
            error |= clSetKernelArg(vec_dot_float_kernel, 3, sizeof(int), &length);
            error |= clSetKernelArg(vec_dot_float_kernel, 4, sizeof(float) * local_size, nullptr); // Local memory
            if (error != CL_SUCCESS) { std::cerr << "dotProduct: Failed to set kernel args: " << error << std::endl; throw; }

            error = clEnqueueNDRangeKernel(queue, vec_dot_float_kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "dotProduct: Failed to enqueue kernel: " << error << std::endl; throw; }

            error = clEnqueueReadBuffer(queue, partial_results_buffer, CL_TRUE, 0, sizeof(float) * num_groups, partial_results_host, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "dotProduct: Failed to read buffer: " << error << std::endl; throw; }

            result = 0.0f;
            for (size_t i = 0; i < num_groups; ++i) {
                result += partial_results_host[i];
            }

            clReleaseMemObject(a_buffer); a_buffer = nullptr;
            clReleaseMemObject(b_buffer); b_buffer = nullptr;
            clReleaseMemObject(partial_results_buffer); partial_results_buffer = nullptr;
            delete[] partial_results_host; partial_results_host = nullptr;
            return true;

        } catch (...) {
            if (a_buffer) clReleaseMemObject(a_buffer);
            if (b_buffer) clReleaseMemObject(b_buffer);
            if (partial_results_buffer) clReleaseMemObject(partial_results_buffer);
            delete[] partial_results_host;
            return false;
        }
    }

    // Matrix multiplication: C = A * B
    static bool multiplyMatrices(const float* a_data, int a_rows, int a_cols,
                                 const float* b_data, int b_rows, int b_cols, 
                                 float* result_data) {
        std::lock_guard<std::mutex> lock(cl_mutex);
        if (a_cols != b_rows) return false; // Dimension check
        if (!cl_available || !shouldUseGPU(a_rows * a_cols + b_rows * b_cols + a_rows * b_cols)) return false;

        cl_int error = CL_SUCCESS;
        cl_mem a_buffer = nullptr;
        cl_mem b_buffer = nullptr;
        cl_mem result_buffer = nullptr;

        try {
            size_t size_a = sizeof(float) * a_rows * a_cols;
            size_t size_b = sizeof(float) * b_rows * b_cols;
            size_t size_result = sizeof(float) * a_rows * b_cols;

            a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_a, (void*)a_data, &error);
            if (error != CL_SUCCESS) { std::cerr << "multiplyMatrices: Failed to create a_buffer: " << error << std::endl; throw; }

            b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_b, (void*)b_data, &error);
            if (error != CL_SUCCESS) { std::cerr << "multiplyMatrices: Failed to create b_buffer: " << error << std::endl; throw; }

            result_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_result, nullptr, &error);
            if (error != CL_SUCCESS) { std::cerr << "multiplyMatrices: Failed to create result_buffer: " << error << std::endl; throw; }

            error = clSetKernelArg(matmul_float_kernel, 0, sizeof(cl_mem), &a_buffer);
            error |= clSetKernelArg(matmul_float_kernel, 1, sizeof(cl_mem), &b_buffer);
            error |= clSetKernelArg(matmul_float_kernel, 2, sizeof(cl_mem), &result_buffer);
            error |= clSetKernelArg(matmul_float_kernel, 3, sizeof(int), &a_rows);
            error |= clSetKernelArg(matmul_float_kernel, 4, sizeof(int), &a_cols); // a_cols is also b_rows
            error |= clSetKernelArg(matmul_float_kernel, 5, sizeof(int), &b_cols);
            
            // Determine local work size for matmul_float_kernel (e.g., 16x16 tile)
            // The kernel uses get_local_size(0) as TILE_SIZE. This should be a power of 2.
            // Max work group size needs to be considered for total local items (TILE_SIZE*TILE_SIZE)
            size_t tile_size = 16; // Common tile size
            while(tile_size * tile_size > max_work_group_size && tile_size > 1) { // Ensure tile fits
                tile_size /= 2;
            }
            if (tile_size == 0) tile_size = 1; // Fallback if max_work_group_size is tiny

            error |= clSetKernelArg(matmul_float_kernel, 6, sizeof(float) * tile_size * tile_size, nullptr); // tileA
            error |= clSetKernelArg(matmul_float_kernel, 7, sizeof(float) * tile_size * tile_size, nullptr); // tileB
            if (error != CL_SUCCESS) { std::cerr << "multiplyMatrices: Failed to set kernel args: " << error << std::endl; throw; }

            size_t global_work_size[2] = { (size_t)((b_cols + tile_size - 1) / tile_size) * tile_size, 
                                           (size_t)((a_rows + tile_size - 1) / tile_size) * tile_size }; 
            size_t local_work_size[2] = { tile_size, tile_size };

            error = clEnqueueNDRangeKernel(queue, matmul_float_kernel, 2, nullptr, global_work_size, local_work_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "multiplyMatrices: Failed to enqueue kernel: " << error << std::endl; throw; }

            error = clEnqueueReadBuffer(queue, result_buffer, CL_TRUE, 0, size_result, result_data, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) { std::cerr << "multiplyMatrices: Failed to read buffer: " << error << std::endl; throw; }

            clReleaseMemObject(a_buffer); a_buffer = nullptr;
            clReleaseMemObject(b_buffer); b_buffer = nullptr;
            clReleaseMemObject(result_buffer); result_buffer = nullptr;
            return true;

        } catch (...) {
            if (a_buffer) clReleaseMemObject(a_buffer);
            if (b_buffer) clReleaseMemObject(b_buffer);
            if (result_buffer) clReleaseMemObject(result_buffer);
            return false;
        }
    }

    // Matrix-vector multiplication: result_vec = matrix * vector
    static bool multiplyMatrixVector(const float* matrix_data, int matrix_rows, int matrix_cols,
                                     const float* vector_data, float* result_vector_data) {
        // Reuse multiplyMatrices by treating vector as a matrix_cols x 1 matrix
        // The result will be a matrix_rows x 1 matrix (which is our result_vector_data)
        return multiplyMatrices(matrix_data, matrix_rows, matrix_cols, 
                              vector_data, matrix_cols, 1, 
                              result_vector_data);
    }

    // Execute prep_vec operation using OpenCL
    static bool prepVecGPU(const float* image_vec, int image_width, int image_height, 
                          float* result, int filter_size, int stride, int padding) {
        if (!initialize() || !cl_available) {
            return false;
        }
        
        std::lock_guard<std::mutex> lock(cl_mutex);
        
        try {
            // Calculate dimensions
            int num_submatrices_x = (image_width + 2 * padding - filter_size) / stride + 1;
            int num_submatrices_y = (image_height + 2 * padding - filter_size) / stride + 1;
            int total_submatrices = num_submatrices_x * num_submatrices_y;
            int result_size = total_submatrices * filter_size * filter_size;
            
            // Check if the operation exceeds device capabilities
            size_t input_size = sizeof(float) * image_width * image_height;
            size_t output_size = sizeof(float) * result_size;
            
            if (input_size > max_mem_alloc_size || output_size > max_mem_alloc_size) {
                return false;
            }
            
            // Create buffers
            cl_int error;
            cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                input_size, (void*)image_vec, &error);
            if (error != CL_SUCCESS) {
                std::cerr << "Failed to create input buffer for prep_vec: " << error << std::endl;
                return false;
            }
            
            cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                                output_size, nullptr, &error);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(input_buffer);
                std::cerr << "Failed to create output buffer for prep_vec: " << error << std::endl;
                return false;
            }
            
            // Set kernel arguments
            error = clSetKernelArg(prep_vec_float_kernel, 0, sizeof(cl_mem), &input_buffer);
            error |= clSetKernelArg(prep_vec_float_kernel, 1, sizeof(cl_mem), &output_buffer);
            error |= clSetKernelArg(prep_vec_float_kernel, 2, sizeof(int), &image_width);
            error |= clSetKernelArg(prep_vec_float_kernel, 3, sizeof(int), &image_height);
            error |= clSetKernelArg(prep_vec_float_kernel, 4, sizeof(int), &filter_size);
            error |= clSetKernelArg(prep_vec_float_kernel, 5, sizeof(int), &stride);
            error |= clSetKernelArg(prep_vec_float_kernel, 6, sizeof(int), &padding);
            error |= clSetKernelArg(prep_vec_float_kernel, 7, sizeof(int), &num_submatrices_x);
            error |= clSetKernelArg(prep_vec_float_kernel, 8, sizeof(int), &num_submatrices_y);
            
            if (error != CL_SUCCESS) {
                clReleaseMemObject(input_buffer);
                clReleaseMemObject(output_buffer);
                std::cerr << "Failed to set kernel arguments for prep_vec: " << error << std::endl;
                return false;
            }
            
            // Execute kernel
            size_t global_size = total_submatrices;
            size_t local_size = std::min(max_work_group_size, (size_t)256); // Adjust based on hardware
            
            error = clEnqueueNDRangeKernel(queue, prep_vec_float_kernel, 1, nullptr,
                                         &global_size, &local_size, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(input_buffer);
                clReleaseMemObject(output_buffer);
                std::cerr << "Failed to execute prep_vec kernel: " << error << std::endl;
                return false;
            }
            
            // Read results
            error = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
                                     output_size, result, 0, nullptr, nullptr);
            if (error != CL_SUCCESS) {
                clReleaseMemObject(input_buffer);
                clReleaseMemObject(output_buffer);
                std::cerr << "Failed to read results from prep_vec: " << error << std::endl;
                return false;
            }
            
            // Cleanup
            clReleaseMemObject(input_buffer);
            clReleaseMemObject(output_buffer);
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "OpenCL prepVecGPU operation failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Clean up resources
    static void cleanup() {
        if (cl_initialized && cl_available) {
            std::lock_guard<std::mutex> lock(cl_mutex);
            
            if (vec_add_float_kernel) {
                clReleaseKernel(vec_add_float_kernel);
                vec_add_float_kernel = nullptr;
            }
            if (vec_sub_float_kernel) {
                clReleaseKernel(vec_sub_float_kernel);
                vec_sub_float_kernel = nullptr;
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
            
            if (outer_prod_float_kernel) {
                clReleaseKernel(outer_prod_float_kernel);
                outer_prod_float_kernel = nullptr;
            }
            
            if (prep_vec_float_kernel) {
                clReleaseKernel(prep_vec_float_kernel);
                prep_vec_float_kernel = nullptr;
            }
            
            if (vec_scale_float_kernel) {
                clReleaseKernel(vec_scale_float_kernel);
                vec_scale_float_kernel = nullptr;
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
cl_kernel OpenCLAccelerator::vec_sub_float_kernel = nullptr;
cl_kernel OpenCLAccelerator::vec_mult_float_kernel = nullptr;
cl_kernel OpenCLAccelerator::vec_dot_float_kernel = nullptr;
cl_kernel OpenCLAccelerator::matmul_float_kernel = nullptr;
cl_kernel OpenCLAccelerator::outer_prod_float_kernel = nullptr;
cl_kernel OpenCLAccelerator::prep_vec_float_kernel = nullptr;
cl_kernel OpenCLAccelerator::vec_scale_float_kernel = nullptr;

std::mutex OpenCLAccelerator::cl_mutex;
std::atomic<bool> OpenCLAccelerator::initialized(false);
size_t OpenCLAccelerator::max_work_group_size = 0;
size_t OpenCLAccelerator::max_compute_units = 0;
size_t OpenCLAccelerator::max_mem_alloc_size = 0;
int OpenCLAccelerator::gpu_threshold = 256;  // Default value