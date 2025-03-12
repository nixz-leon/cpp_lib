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

template <typename T> class matrix{
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
        
        static void initGL() {
            if (!glfwInit()) {
                throw std::runtime_error("Failed to initialize GLFW");
            }
            
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
            glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
            
            window = glfwCreateWindow(1, 1, "Compute", nullptr, nullptr);
            if (!window) {
                glfwTerminate();
                throw std::runtime_error("Failed to create GLFW window");
            }
            
            glfwMakeContextCurrent(window);
            
            if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
                throw std::runtime_error("Failed to initialize GLAD");
            }
            
            // Load compute shader
            computeProgram = loadComputeShader("include/matmul.comp");
            glInitialized = true;
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
        
        static int GPU_SIZE_THRESHOLD;
        static int THREAD_SIZE_THRESHOLD;
        static int CACHE_BLOCK_SIZE;

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

        static matrix<T> multiplySequential(const matrix<T>& a, const matrix<T>& b) {
            matrix<T> result(a.row, b.col);
            for (int i = 0; i < a.row; i += CACHE_BLOCK_SIZE) {
                for (int j = 0; j < b.col; j += CACHE_BLOCK_SIZE) {
                    for (int k = 0; k < a.col; k += CACHE_BLOCK_SIZE) {
                        for (int ii = i; ii < std::min(i + CACHE_BLOCK_SIZE, a.row); ++ii) {
                            for (int jj = j; jj < std::min(j + CACHE_BLOCK_SIZE, b.col); ++jj) {
                                T sum = result(ii, jj);
                                for (int kk = k; kk < std::min(k + CACHE_BLOCK_SIZE, a.col); ++kk) {
                                    sum += a(ii, kk) * b(kk, jj);
                                }
                                result(ii, jj) = sum;
                            }
                        }
                    }
                }
            }
            return result;
        };

        static matrix<T> multiplyThreaded(const matrix<T>& a, const matrix<T>& b) {
            matrix<T> temp(a.row, b.col);
            
            auto multiply_block = [&a, &b, &temp](int start_row, int end_row) {
                for (int i = start_row; i < end_row; ++i) {
                    for (int j = 0; j < b.col; ++j) {
                        temp(i, j) = 0;
                        for (int k = 0; k < b.row; ++k) {
                            temp(i, j) += a(i, k) * b(k, j);
                        }
                    }
                }
            };

            int num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) num_threads = 2;

            int rows_per_thread = a.row / num_threads;
            int remaining_rows = a.row % num_threads;

            std::vector<std::thread> threads;
            int start_row = 0;

            for (int t = 0; t < num_threads; ++t) {
                int end_row = start_row + rows_per_thread + (t < remaining_rows ? 1 : 0);
                threads.push_back(std::thread(multiply_block, start_row, end_row));
                start_row = end_row;
            }

            for (auto& t : threads) {
                t.join();
            }

            return temp;
        };

        static matrix<T> multiplyGPU(const matrix<T>& a, const matrix<T>& b) {
            if (!glInitialized) {
                initGL();
            }

            matrix<T> result(a.row, b.col);
            
            GLuint buffers[3];
            glGenBuffers(3, buffers);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[0]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, a.row * a.col * sizeof(T), a.data, GL_STATIC_DRAW);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[1]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, b.row * b.col * sizeof(T), b.data, GL_STATIC_DRAW);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
            glBufferData(GL_SHADER_STORAGE_BUFFER, result.row * result.col * sizeof(T), nullptr, GL_DYNAMIC_COPY);
            
            glUseProgram(computeProgram);
            glUniform1i(glGetUniformLocation(computeProgram, "width"), b.col);
            glUniform1i(glGetUniformLocation(computeProgram, "height"), a.row);
            glUniform1i(glGetUniformLocation(computeProgram, "depth"), a.col);
            
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers[0]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, buffers[1]);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers[2]);
            
            glDispatchCompute((a.row + 15) / 16, (b.col + 15) / 16, 1);
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
            
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers[2]);
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, result.row * result.col * sizeof(T), result.data);
            
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
        
            if (total_elements >= GPU_SIZE_THRESHOLD * GPU_SIZE_THRESHOLD) {
                try {
                    return multiplyGPU(a, b);
                } catch (const std::runtime_error& e) {
                    std::cerr << "GPU multiplication failed, falling back to CPU: " << e.what() << std::endl;
                }
            }
        
            if (total_elements >= THREAD_SIZE_THRESHOLD * THREAD_SIZE_THRESHOLD) {
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
template <typename T> int matrix<T>::GPU_SIZE_THRESHOLD = 512;
template <typename T> int matrix<T>::THREAD_SIZE_THRESHOLD = 128;
template <typename T> int matrix<T>::CACHE_BLOCK_SIZE = 32;

template <typename T>
matrix<T>::matrix(){
    static bool capabilities_detected = false;
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
    data = new T[n*m];
    std::memset(data, 0 , sizeof(T)*n*m);
    //data = (T*)std::calloc(n*m,sizeof(T));
    row = n;
    col = m;
}

template <typename T>
inline matrix<T>::matrix(vecs<T> &vectors, bool row_major){
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
}

