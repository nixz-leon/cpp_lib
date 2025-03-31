#pragma once
#ifndef iostream
#include <iostream>
#endif
#ifndef string
#include <string>
#endif
#ifndef cstring
#include <cstring>
#endif
#ifndef vec_hpp
#include "vec.hpp"
#endif
#ifndef  random
#include <random>
#endif
#ifndef vector
#include <vector>
#endif 
#ifndef thread
#include <thread>
#endif 
#ifndef functional
#include <functional>
#endif
#ifndef glad
#include <glad/glad.h>
#endif
#ifndef glfw3
#include <GLFW/glfw3.h>
#endif
#ifndef algorithm
#include <algorithm>
#endif
#ifndef windows
#include <windows.h>
#endif

template <typename T>
class matrix {  
    public: //this are memory functions
        matrix();//default 
        matrix(int n, int m); //normal
        matrix(vecs<T> &vectors, bool row_major = false);
        matrix(const matrix<T> &other);// copy
        matrix<T>& operator=(matrix<T>& other); //copy assigment
        matrix(matrix<T> &&other) noexcept;//move
        matrix<T>& operator=(matrix<T>&& other) noexcept;// move assignment 
        T& operator()(int row, int col);
        const T& operator()(int row, int col)const;
        void printout();
        ~matrix();
        int row;
        int col;
    private:
        T *data;
        static GLFWwindow* window;
        static GLuint computeProgram;
        static bool glInitialized;
        static bool capabilities_detected;  // Add this line
        static int GPU_SIZE_THRESHOLD;
        static int THREAD_SIZE_THRESHOLD;
        static int CACHE_BLOCK_SIZE;

        static bool initGL() {
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
                
                // Load compute shader
                try {
                    if constexpr (std::is_same_v<T, float>) {
                    computeProgram = loadComputeShader("include/matmul_float.comp");
                } else {
                    computeProgram = loadComputeShader("include/matmul_double.comp");
                }
                    glInitialized = true;
                    return true;
                } catch (const std::runtime_error& e) {
                    //std::cerr << "Shader loading failed: " << e.what() << std::endl;
                    glfwDestroyWindow(window);
                    glfwTerminate();
                    return false;
                }
            } catch (const std::exception& e) {
                std::cerr << "GL initialization failed: " << e.what() << std::endl;
                return false;
            }
        };
        static GLuint loadComputeShader(const char* filepath) {
            std::ifstream file(filepath);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open compute shader file");
            }
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string shaderSource = buffer.str();
            const char* sourcePtr = shaderSource.c_str();
    
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
        };
        
        static void detectHardwareCapabilities() {
            // GPU capabilities
            if (glInitialized || initGL()) {
                GLint maxWorkGroupSize[3];
                GLint maxMemorySize;
                
                glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, 0, &maxWorkGroupSize[0]);
                glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxMemorySize);
                
                // Calculate GPU threshold based on work group size and memory
                int maxGPUMatrixDim = std::min(
                    static_cast<int>(sqrt(maxMemorySize / sizeof(T))),
                    maxWorkGroupSize[0] * 32  // Assuming 32 work groups is optimal
                );
                GPU_SIZE_THRESHOLD = std::max(256, maxGPUMatrixDim);
            } else {
                // If GPU not available, set threshold to maximum to disable GPU path
                GPU_SIZE_THRESHOLD = std::numeric_limits<int>::max();
            }

            // CPU capabilities
            SYSTEM_INFO sysInfo;
            GetSystemInfo(&sysInfo);
            
            // Get number of CPU cores
            unsigned int numCores = sysInfo.dwNumberOfProcessors;
            
            // Get L1 cache size
            DWORD bufferSize = 0;
            GetLogicalProcessorInformation(nullptr, &bufferSize);
            std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
            GetLogicalProcessorInformation(buffer.data(), &bufferSize);
            
            size_t l1CacheSize = 32 * 1024; // Default 32KB if detection fails
            for (const auto& info : buffer) {
                if (info.Relationship == RelationCache && 
                    info.Cache.Level == 1 &&
                    info.Cache.Type == CacheData) {
                    l1CacheSize = info.Cache.Size;
                    break;
                }
            }
            
            // Set cache block size based on L1 cache
            CACHE_BLOCK_SIZE = static_cast<int>(sqrt(l1CacheSize / (2 * sizeof(T))));
            CACHE_BLOCK_SIZE = std::min(64, std::max(16, CACHE_BLOCK_SIZE));
            
            // Set thread threshold based on number of cores
            THREAD_SIZE_THRESHOLD = static_cast<int>(sqrt(l1CacheSize / sizeof(T)) * numCores);
            THREAD_SIZE_THRESHOLD = std::max(128, std::min(512, THREAD_SIZE_THRESHOLD));
        };

        static inline matrix<T> multiplySequential(const matrix<T>& a, const matrix<T>& b) {
            matrix<T> result(a.row, b.col);
            const int block = CACHE_BLOCK_SIZE;
            
            // Triple nested loop with cache blocking
            #pragma omp simd
            for (int i = 0; i < a.row; i += block) {
                for (int k = 0; k < a.col; k += block) {
                    for (int j = 0; j < b.col; j += block) {
                        // Block multiplication
                        for (int ii = i; ii < std::min(i + block, a.row); ++ii) {
                            for (int kk = k; kk < std::min(k + block, a.col); ++kk) {
                                T r = a(ii, kk);
                                #pragma omp simd
                                for (int jj = j; jj < std::min(j + block, b.col); ++jj) {
                                    result(ii, jj) += r * b(kk, jj);
                                }
                            }
                        }
                    }
                }
            }
            return result;
        };

        static inline matrix<T> multiplyThreaded(const matrix<T>& a, const matrix<T>& b) {
            matrix<T> result(a.row, b.col);
            const int block = CACHE_BLOCK_SIZE;
            
            // Calculate optimal number of threads based on CPU cores and matrix size
            int num_threads = std::min(
                static_cast<int>(std::thread::hardware_concurrency()),
                (a.row + block - 1) / block
            );
            
            // Pre-allocate thread vector to avoid reallocations
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            
            // Aligned memory access optimization
            alignas(32) T* temp_result = new T[a.row * b.col]();
            
            auto multiply_block = [&](int start_i, int end_i) {
                // Thread-local accumulator for better cache usage
                alignas(32) T local_buffer[block * block];
                
                for (int i = start_i; i < end_i; i += block) {
                    for (int k = 0; k < a.col; k += block) {
                        // Zero local buffer
                        std::memset(local_buffer, 0, sizeof(T) * block * block);
                        
                        for (int j = 0; j < b.col; j += block) {
                            // Compute block boundaries
                            const int i_end = std::min(i + block, end_i);
                            const int k_end = std::min(k + block, a.col);
                            const int j_end = std::min(j + block, b.col);
                            
                            // Manual vectorization hints
                            #pragma omp simd collapse(2)
                            for (int ii = i; ii < i_end; ++ii) {
                                for (int kk = k; kk < k_end; ++kk) {
                                    const T r = a(ii, kk);
                                    const int row_offset = (ii - i) * block;
                                    
                                    #pragma omp simd
                                    for (int jj = j; jj < j_end; ++jj) {
                                        local_buffer[row_offset + (jj - j)] += r * b(kk, jj);
                                    }
                                }
                            }
                            
                            // Accumulate block results
                            for (int ii = i; ii < i_end; ++ii) {
                                const int row_offset = (ii - i) * block;
                                #pragma omp simd
                                for (int jj = j; jj < j_end; ++jj) {
                                    result(ii, jj) += local_buffer[row_offset + (jj - j)];
                                }
                            }
                        }
                    }
                }
            };
            
            // Distribute work among threads
            const int rows_per_thread = (a.row + num_threads - 1) / num_threads;
            
            for (int t = 0; t < num_threads; ++t) {
                const int start_row = t * rows_per_thread;
                const int end_row = std::min(start_row + rows_per_thread, a.row);
                if (start_row < end_row) {
                    threads.emplace_back(multiply_block, start_row, end_row);
                }
            }
            
            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
            
            // Cleanup
            delete[] temp_result;
            
            return result;
        };

        static inline matrix<T> multiplyGPU(const matrix<T>& a, const matrix<T>& b) {
            if (!glInitialized && !initGL()) {
                throw std::runtime_error("Failed to initialize OpenGL");
            }
        
            // Check if double precision is supported
            if constexpr (std::is_same_v<T, double>) {
                GLint doublePrecisionSupported;
                // Check for GL_ARB_gpu_shader_fp64 extension
                GLint numExtensions;
                glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);
                bool hasDoublePrecision = false;
                
                for (GLint i = 0; i < numExtensions; ++i) {
                    const GLubyte* extension = glGetStringi(GL_EXTENSIONS, i);
                    if (std::strcmp(reinterpret_cast<const char*>(extension), "GL_ARB_gpu_shader_fp64") == 0) {
                        hasDoublePrecision = true;
                        break;
                    }
                }
                
                if (!hasDoublePrecision) {
                    throw std::runtime_error("Double precision not supported by GPU");
                }
            }
        
            // Load appropriate shader version
            if (computeProgram == 0) {
                if constexpr (std::is_same_v<T, float>) {
                    computeProgram = loadComputeShader("include/matmul_float.comp");
                } else {
                    computeProgram = loadComputeShader("include/matmul_double.comp");
                }
            }
        
            matrix<T> result(a.row, b.col);
            
            // Create buffers
            GLuint buffers[3];
            glGenBuffers(3, buffers);
            
            // Matrix A buffer
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, a.row * a.col * sizeof(T), a.data, GL_STATIC_DRAW);
            
            // Matrix B buffer
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, b.row * b.col * sizeof(T), b.data, GL_STATIC_DRAW);
            
            // Result buffer
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, result.row * result.col * sizeof(T), nullptr, GL_DYNAMIC_COPY);
            
            // Bind compute shader and set uniforms
            glUseProgram(computeProgram);
            glUniform1i(glGetUniformLocation(computeProgram, "width"), b.col);
            glUniform1i(glGetUniformLocation(computeProgram, "height"), a.row);
            glUniform1i(glGetUniformLocation(computeProgram, "depth"), a.col);
            
            // Bind storage buffers
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[0]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers[2]);
            
            // Calculate optimal dispatch size
            const int WORK_GROUP_SIZE = 32;
            int num_groups_x = (b.col + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
            int num_groups_y = (a.row + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
            
            // Dispatch compute shader
            glDispatchCompute(num_groups_x, num_groups_y, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            // Read back results
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 
                              result.row * result.col * sizeof(T), 
                              result.data);
            
            // Cleanup
            glDeleteBuffers(3, buffers);
            
            return result;
        };
        
    public: //these will be the math operator functions and accessor functions
        
        inline void set_diag(T c);
        inline void row_swap(int a, int b);
        inline void row_add(int a, int b, T fac); 
        inline matrix<T> operator/=(T c);
        inline matrix<T> operator *=(T c);
        inline vec<T> operator*(vec<T> b);
        inline friend vec<T> vectorize(matrix<T> a){
            vec<T> temp(a.col * a.row);
            for(int i = 0; i < a.col; i++){
                for(int j = 0; j < a.row; j++){
                    temp((i*a.row)+j) = a(j,i);
                }
            }
            return temp;
        };
        inline friend matrix<T> transpose(matrix<T> a){
            matrix<T> transposed(a.col, a.row);
            for (int i = 0; i < a.row; i++) {
                for (int j = 0; j < a.col; j++) {
                    transposed(j, i) = a(i, j);
                }
            }
            return transposed;
        };
        inline friend matrix<T> operator+(matrix<T> a, matrix<T> b){
            if((a.col!=b.col)||(a.row!=b.row)){std::cout << "Tried to add a " << a.row << ','<<a.col<< " matrix with a " << b.row << ',' << b.col << " matrix\n";exit(0);}
            matrix temp(a);
            for(int i =0; i < a.row;i++){for(int j = 0; j < a.col;j++){temp(i,j)=a(i,j)+b(i,j);}}
            return temp;
        };
        inline friend matrix<T> operator-(matrix<T> a, matrix<T> b){
            if((a.col!=b.col)||(a.row!=b.row)){std::cout << "Tried to subtract a " << b.row << ','<<b.col<< " matrix from a " << a.row << ',' << a.col << " matrix\n";exit(0);}
            matrix temp(a);
            for(int i =0; i < a.row;i++){for(int j = 0; j < a.col;j++){temp(i,j)=a(i,j)-b(i,j);}}
            return temp;
        };
        inline friend matrix<T> operator*(matrix<T> a, matrix<T> b) {
            if (a.col != b.row) {
                std::cout << "Invalid matrix dimensions\n";
                exit(0);
            }
        
            const size_t total_elements = a.row * b.col;
            const size_t total_operations = total_elements * a.col;
        
            // Check if GPU is available and initialized
            static bool gpu_available = false;
            static bool first_check = true;
        
            if (first_check) {
                try {
                    if (initGL()) {
                        // Try to load and compile shaders
                        if constexpr (std::is_same_v<T, float>) {
                            computeProgram = loadComputeShader("include/matmul_float.comp");
                        } else {
                            computeProgram = loadComputeShader("include/matmul_double.comp");
                        }
                        gpu_available = true;
                    }
                } catch (const std::runtime_error& e) {
                    std::cerr << "GPU initialization failed, using CPU-only mode: " << e.what() << std::endl;
                    gpu_available = false;
                    GPU_SIZE_THRESHOLD = std::numeric_limits<int>::max(); // Disable GPU path
                }
                first_check = false;
            }
        
            // Only try GPU path if we know it's available
            if (gpu_available && total_operations >= GPU_SIZE_THRESHOLD * GPU_SIZE_THRESHOLD) {
                try {
                    return multiplyGPU(a, b);
                } catch (const std::runtime_error& e) {
                    std::cerr << "GPU multiplication failed, falling back to CPU: " << e.what() << std::endl;
                    gpu_available = false; // Disable GPU for future multiplications
                    GPU_SIZE_THRESHOLD = std::numeric_limits<int>::max();
                }
            }
        
            // CPU paths
            if (total_operations >= THREAD_SIZE_THRESHOLD * THREAD_SIZE_THRESHOLD) {
                return multiplyThreaded(a, b);
            }
        
            return multiplySequential(a, b);
        };
        inline friend matrix<T> operator*(matrix<T> a, T b){matrix<T>temp(a); temp*=b;return temp;};
        inline friend matrix<T> operator*(T b, matrix<T> a){matrix<T>temp(a); temp*=b;return temp;};
        inline T sum_elms(){
            T sum = 0;
            for(int i = 0; i < row; i++){
                for(int j = 0; j < col; j++){
                    sum += data[(col*i)+j];
                }
            }
            return sum;
        };
};

template <typename T> GLFWwindow* matrix<T>::window = nullptr;
template <typename T> GLuint matrix<T>::computeProgram = 0;
template <typename T> bool matrix<T>::glInitialized = false;
template <typename T> bool matrix<T>::capabilities_detected = false;
template <typename T> int matrix<T>::GPU_SIZE_THRESHOLD = 512;
template <typename T> int matrix<T>::THREAD_SIZE_THRESHOLD = 128;
template <typename T> int matrix<T>::CACHE_BLOCK_SIZE = 32;

template <typename T>
matrix<T>::matrix(){
    if (!capabilities_detected) {
        detectHardwareCapabilities();
        capabilities_detected = true;
    }
    data = nullptr;
    row = 0;
    col = 0;
}

template <typename T>
matrix<T>::matrix(int n, int m){
    if (!capabilities_detected) {
        detectHardwareCapabilities();
        capabilities_detected = true;
    }
    data = new T[n*m];
    std::memset(data, 0 , sizeof(T)*n*m);
    //data = (T*)std::calloc(n*m,sizeof(T));
    row = n;
    col = m;
}

template <typename T>
inline matrix<T>::matrix(vecs<T> &vectors, bool row_major){
    if (!capabilities_detected) {
        detectHardwareCapabilities();
        capabilities_detected = true;
    }
    if(row_major){
        row = vectors.size();
        col = vectors.num_of_vecs();
        data = new T[row*col];
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                data[(col*i)+j] = vectors(j)(i);
            }
        }
    }else{
        col = vectors.size();
        row = vectors.num_of_vecs();
        data = new T[row*col];
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
                data[(col*i)+j] = vectors(i)(j);
            }
        }
    }
};

template <typename T>
matrix<T>::matrix(const matrix<T> &other){
    data = new T[other.row*other.col];//suposed deepcopy
    std::memcpy(data, other.data, sizeof(T)*other.col*other.row);
    row = other.row;
    col = other.col;
};

template <typename T>
inline matrix<T> &matrix<T>::operator=(matrix<T> &other){
    this->row = other.row;
    this->col = other.col;
    std::memcpy(this->data, other.data, sizeof(T)*other.col*other.row);
    return *this;
}

template <typename T>
matrix<T>::matrix(matrix<T> &&other) noexcept:row(0),col(0),data(nullptr){
    data = other.data; // reassigning ownership of pointer
    row = other.row; // redefining rows
    col = other.col; 
    other.data = nullptr; //releasing ownership of pointer from object passed in
    other.col = 0; // setting size to 0 to relfect size;
    other.row = 0;
};

template <typename  T>
inline matrix<T> &matrix<T>::operator=(matrix<T> &&other)noexcept{
    if(this != &other){
        delete[] data;
        data  = other.data;
        row = other.row;
        col = other.col;
        other.data = nullptr;
        other.row = 0;
        other.col = 0;
    }
    return *this;
}

template <typename T>
inline T &matrix<T>::operator()(int r, int c)
{
    if(data == nullptr){std::cout << "no data to be accessed\n"; exit(0);}
    if(row*col == 0){std::cout << "empty matrix\n";exit(0);}//checking if the object can be accessed in the first place 
    if((r > row-1)||(c > col-1)){std::cout << "Tried to access matrix elm " << r << ',' << c << ", But size is " << row << ',' << col << '\n';exit(0);} // bounds checking
    if(r < 0){std::cout << "tried to access row " << r << " which is negative\n";}
    if(c < 0){std::cout << "tried to access row " << c << " which is negative\n";}
    return data[(r*col)+c];
}

template <typename T>
inline const T &matrix<T>::operator()(int r, int c) const
{
    if(data == nullptr){std::cout << "no data to be accessed\n"; exit(0);}
    if(row*col == 0){std::cout << "empty matrix\n";exit(0);}//checking if the object can be accessed in the first place 
    if((r > row-1)||(c > col-1)){std::cout << "Tried to access matrix elm " << r << ',' << c << ", But size is " << row << ',' << col << '\n';exit(0);} // bounds checking
    if(r < 0){std::cout << "tried to access row " << r << " which is negative\n";}
    if(c < 0){std::cout << "tried to access row " << c << " which is negative\n";}
    return data[(r*col)+c];;
}

template <typename T>
inline void matrix<T>::printout()
{
    if(data !=nullptr){
        std::cout << '\n';
        for(int i = 0; i < row; i++){
            for(int j = 0; j < col; j++){
            std::cout << data[(col*i)+j] << ' ';
            }
        std::cout << '\n';
        }
        std::cout << '\n';
    }
}

template <typename T>
inline matrix<T>::~matrix() {
    delete[] data;
    row = 0;
    col = 0;
    
    if (glInitialized) {
        if (window) {
            glfwDestroyWindow(window);
            window = nullptr;
        }
        if (computeProgram) {
            glDeleteProgram(computeProgram);
            computeProgram = 0;
        }
        glfwTerminate();
        glInitialized = false;
    }
}

template <typename T>
inline void matrix<T>::set_diag(T c)
{
    for(int i = 0; i < row; i++){
        data[(i*col)+i]=c;
    }
}

template <typename T>
inline void matrix<T>::row_swap(int a, int b)
{
    if(a > row-1){std::cout << "Tried to access elm " << a << ", But size is " << row-1 << '\n';exit(0);}
    if(b > row-1){std::cout << "Tried to access elm " << b << ", But size is " << row-1 << '\n';exit(0);}
    if(a==b){return;}
    for(int i =0; i < col; i++){
        T temp = data[(a*col)+i];
        data[(a*col)+i] = data[(b*col)+i];
        data[(b*col)+i] = temp;
    } 
    
}

template <typename T>
inline void matrix<T>::row_add(int a, int b, T fac)
{
    if(a > row-1){std::cout << "Tried to access elm " << a << ", But size is " << row-1 << '\n';exit(0);}
    if(b > row-1){std::cout << "Tried to access elm " << b << ", But size is " << row-1 << '\n';exit(0);}
    if(a==b){return;}
    for(int i =0; i < col; i++){
        data[(a*col)+i] += fac*data[(b*col)+i];
    }
}

template <typename T>
inline matrix<T> matrix<T>::operator/=(T c)
{
    for(int i=0;i<this->row*this->col;i++){this->data[i]/=c;}
    return *this;
}

template <typename T>
inline matrix<T> matrix<T>::operator*=(T c)
{
    for(int i=0;i<this->row*this->col;i++){this->data[i]*=c;}
    return *this;
}

template <typename T>
inline vec<T> matrix<T>::operator*(vec<T> b)
{
    if (this->data == nullptr) {
        std::cout << "Error: matrix is empty\n";
        exit(0);
    }
    if (b.size != this->col) { // Ensure the matrix columns match the vector size
        std::cout << "Error: Tried to multiply a matrix of size " << this->row << "x" << this->col
                  << " with a vector of size " << b.size << "\n";
        exit(0);
    }

    vec<T> temp(this->row); // Output vector should have the same number of rows as the matrix
    T sum = 0;

    for (int i = 0; i < this->row; i++) {  // Loop over matrix rows
        sum = 0;
        for (int j = 0; j < this->col; j++) {  // Loop over matrix columns
            sum += this->data[(i * this->col) + j] * b(j);
        }
        temp(i) = sum;
    }
    return temp;
};

