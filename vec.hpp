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
#ifndef ffteasyjj_cpp
#include "ffteasyjj.cpp"
#endif
#ifndef vector
#include <vector>
#endif
#ifndef cmath
#include <cmath>
#endif
#ifndef algorithm
#include <algorithm>
#endif
#ifndef functional
#include <functional>
#endif
#ifndef random
#include <random>
#endif
#ifndef thread
#include <thread>
#endif

// Include OpenCLAccelerator header
#include "OpenCLAccelerator.hpp"


// Forward declaration of FFT function used in the vec class
extern "C" void fftr1(double*, int, int);

template<class T>
struct _init_list_with_square_brackets {
    const std::initializer_list<T>& list;
    _init_list_with_square_brackets(const std::initializer_list<T>& _list): list(_list) {}
    T operator[](unsigned int index) {
        return *(list.begin() + index);
    }
    int size(){
        return list.size();
    }

};

template <typename T>
class vec {
    public:
        vec() : size(0),data(nullptr){} //default
        vec(int n) : size(n) {data = new T[size]();}; //normal
        vec(T *data, int n) : size(n){
            this->data = new T[size];
            std::memcpy(this->data, data, sizeof(T) * size);
        };
        vec(int n, T val) : size(n){
            data = new T[size];
            std::fill(data, data + size, val);
        }
        vec(std::initializer_list<T> list) : size(list.size()) {
            data = new T[size];
            std::copy(list.begin(), list.end(), data);
        }
        vec(const vec<T> &other);// copy
        vec<T>& operator=(vec<T>& other); //copy assigment
        vec<T>& operator=(const vec<T>& other); // const copy assignment
        vec(vec<T> &&other) noexcept;//move
        vec<T>& operator=(vec<T>&& other) noexcept;// move assignment 
        vec<T>& operator=(std::initializer_list<T> list) {
            if (size != (int)list.size()) {
                delete[] data;
                size = list.size();
                data = new T[size];
            }
            std::copy(list.begin(), list.end(), data);
            return *this;
        }
        T& operator()(int i);
        const T& operator()(int i)const;
        void printout();
        ~vec();
        int size;
        T *data; // Making public to allow GPU adapter direct access

    private:
        static inline vec<T> addSequential(const vec<T>& a, const vec<T>& b) {
            vec<T> result(a.size);
            #pragma omp simd
            for (int i = 0; i < a.size; i++) {
                result.data[i] = a.data[i] + b.data[i];
            }
            return result;
        }

        static inline vec<T> addThreaded(const vec<T>& a, const vec<T>& b) {
            vec<T> result(a.size);
            const int block = OpenCLAccelerator::getCacheBlockSize();
            
            // Calculate optimal number of threads based on CPU cores and vector size
            int num_threads = std::min(
                static_cast<int>(std::thread::hardware_concurrency()),
                (a.size + block - 1) / block
            );
            
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            
            auto add_block = [&](int start_idx, int end_idx) {
                #pragma omp simd
                for (int i = start_idx; i < end_idx; i++) {
                    result.data[i] = a.data[i] + b.data[i];
                }
            };
            
            // Distribute work among threads
            const int elements_per_thread = (a.size + num_threads - 1) / num_threads;
            
            for (int t = 0; t < num_threads; ++t) {
                const int start_idx = t * elements_per_thread;
                const int end_idx = std::min(start_idx + elements_per_thread, a.size);
                if (start_idx < end_idx) {
                    threads.emplace_back(add_block, start_idx, end_idx);
                }
            }
            
            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
            
            return result;
        }

        static inline vec<T> subtractSequential(const vec<T>& a, const vec<T>& b) {
            vec<T> result(a.size);
            #pragma omp simd
            for (int i = 0; i < a.size; i++) {
                result.data[i] = a.data[i] - b.data[i];
            }
            return result;
        }

        static inline vec<T> subtractThreaded(const vec<T>& a, const vec<T>& b) {
            vec<T> result(a.size);
            const int block = OpenCLAccelerator::getCacheBlockSize();
            
            // Calculate optimal number of threads based on CPU cores and vector size
            int num_threads = std::min(
                static_cast<int>(std::thread::hardware_concurrency()),
                (a.size + block - 1) / block
            );
            
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            
            auto subtract_block = [&](int start_idx, int end_idx) {
                #pragma omp simd
                for (int i = start_idx; i < end_idx; i++) {
                    result.data[i] = a.data[i] - b.data[i];
                }
            };
            
            // Distribute work among threads
            const int elements_per_thread = (a.size + num_threads - 1) / num_threads;
            
            for (int t = 0; t < num_threads; ++t) {
                const int start_idx = t * elements_per_thread;
                const int end_idx = std::min(start_idx + elements_per_thread, a.size);
                if (start_idx < end_idx) {
                    threads.emplace_back(subtract_block, start_idx, end_idx);
                }
            }
            
            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
            
            return result;
        }

    public:
        inline friend vec<T> operator+(const vec<T>& a, const vec<T>& b) {
            if (a.data == nullptr || b.data == nullptr) {
                std::cout << "Error: vector is empty\n";
                exit(0);
            }
            if (a.size != b.size) {
                std::cout << "Error: Tried to add a vector of size " << a.size
                          << " with a vector of size " << b.size << "\n";
                exit(0);
            }

            // Use hardware acceleration if available
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    if (OpenCLAccelerator::isOpenCLAvailable() && OpenCLAccelerator::shouldUseGPU(a.size)) {
                        vec<T> result(a.size);
                        bool success = OpenCLAccelerator::addVectors(
                            a.data, b.data, result.data, a.size
                        );
                        
                        if (success) {
                            return result;
                        }
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if hardware acceleration fails
                std::cerr << "OpenCL acceleration failed, using CPU: " << e.what() << std::endl;
            }

            // Use multithreaded CPU implementation for larger vectors
            if (a.size >= OpenCLAccelerator::getThreadThreshold()) {
                return addThreaded(a, b);
            }
            
            // Use sequential implementation for smaller vectors
            return addSequential(a, b);
        }

        inline vec<T> operator+=(const vec<T>& other){
            if (this->data == nullptr || other.data == nullptr) {
                std::cout << "Error: vector is empty\n";
                exit(0);
            }
            if (this->size != other.size) {
                std::cout << "Error: Tried to add a vector of size " << this->size
                          << " with a vector of size " << other.size << "\n";
                exit(0);
            }

            // Use hardware acceleration if available
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    if (OpenCLAccelerator::isOpenCLAvailable() && OpenCLAccelerator::shouldUseGPU(this->size)) {
                        // For in-place, we need a temporary buffer for 'this->data' if the OpenCL kernel
                        // cannot guarantee safe in-place operation or if we want to reuse the addVectors kernel
                        // which takes two inputs and one output.
                        // A true in-place OpenCL kernel would be ideal: void addInPlace(T* a, const T* b, int size);
                        // Assuming addVectors(a,b,result,size) cannot write result to a or b directly if they overlap.
                        // So, we use addVectors and copy back, or implement CPU path.
                        // For simplicity here, let's fall through to CPU if direct in-place OpenCL is not straightforwardly available
                        // OR, if addVectors can handle it, result.data could be this->data.
                        // Let's assume for now addVectors needs distinct output, so OpenCL path will be similar to operator+
                        // and then we copy. Or, we can just use CPU path for += to avoid this complexity if
                        // direct in-place OpenCL is not available.
                        // A more robust solution would be an OpenCL kernel specifically for in-place addition.
                        // Given the current OpenCLAccelerator::addVectors, it's safer to perform on CPU or use a temporary result.
                        // To avoid allocation here, let's implement the CPU paths directly for in-place.
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "OpenCL check failed during +=, proceeding with CPU: " << e.what() << std::endl;
            }


            // Use multithreaded CPU implementation for larger vectors
            if (this->size >= OpenCLAccelerator::getThreadThreshold()) {
                const int block = OpenCLAccelerator::getCacheBlockSize();
                int num_threads = std::min(
                    static_cast<int>(std::thread::hardware_concurrency()),
                    (this->size + block - 1) / block
                );
                std::vector<std::thread> threads;
                threads.reserve(num_threads);
                
                auto add_block_inplace = [&](int start_idx, int end_idx) {
                    #pragma omp simd
                    for (int i = start_idx; i < end_idx; i++) {
                        this->data[i] += other.data[i];
                    }
                };
                
                const int elements_per_thread = (this->size + num_threads - 1) / num_threads;
                for (int t = 0; t < num_threads; ++t) {
                    const int start_idx = t * elements_per_thread;
                    const int end_idx = std::min(start_idx + elements_per_thread, this->size);
                    if (start_idx < end_idx) {
                        threads.emplace_back(add_block_inplace, start_idx, end_idx);
                    }
                }
                for (auto& thread : threads) {
                    thread.join();
                }
            } else { // Use sequential implementation for smaller vectors
                #pragma omp simd
                for (int i = 0; i < this->size; i++) {
                    this->data[i] += other.data[i];
                }
            }
            return *this;
        }
        
        inline vec<T> operator-=(const vec<T>& other){
            if (this->data == nullptr || other.data == nullptr) {
                std::cout << "Error: vector is empty\n";
                exit(0);
            }
            if (this->size != other.size) {
                std::cout << "Error: Tried to subtract a vector of size " << this->size
                          << " from a vector of size " << other.size << "\n";
                exit(0);
            }

            // OpenCL path consideration: Similar to operator+=, a dedicated in-place OpenCL kernel
            // for subtraction (e.g., subtractInPlace(T* a, const T* b, int size)) would be optimal.
            // For now, we fall back to CPU for in-place to avoid allocation issues with addVectors-like kernels.
            // try {
            //     if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            //         if (OpenCLAccelerator::isOpenCLAvailable() && OpenCLAccelerator::shouldUseGPU(this->size)) {
            //             // Potentially use OpenCLAccelerator::addVectors with negated 'other' if it can operate in-place on 'this'.
            //             // Or, ideally, an OpenCLAccelerator::subtractVectorsInPlace(this->data, other.data, this->size)
            //             // If not possible without allocation, rely on CPU path below.
            //         }
            //     }
            // } catch (const std::exception& e) {
            //     std::cerr << "OpenCL check failed during -=, proceeding with CPU: " << e.what() << std::endl;
            // }

            // Use multithreaded CPU implementation for larger vectors
            if (this->size >= OpenCLAccelerator::getThreadThreshold()) {
                const int block = OpenCLAccelerator::getCacheBlockSize();
                int num_threads = std::min(
                    static_cast<int>(std::thread::hardware_concurrency()),
                    (this->size + block - 1) / block
                );
                std::vector<std::thread> threads;
                threads.reserve(num_threads);
                
                auto subtract_block_inplace = [&](int start_idx, int end_idx) {
                    #pragma omp simd
                    for (int i = start_idx; i < end_idx; i++) {
                        this->data[i] -= other.data[i];
                    }
                };
                
                const int elements_per_thread = (this->size + num_threads - 1) / num_threads;
                for (int t = 0; t < num_threads; ++t) {
                    const int start_idx = t * elements_per_thread;
                    const int end_idx = std::min(start_idx + elements_per_thread, this->size);
                    if (start_idx < end_idx) {
                        threads.emplace_back(subtract_block_inplace, start_idx, end_idx);
                    }
                }
                for (auto& thread : threads) {
                    thread.join();
                }
            } else { // Use sequential implementation for smaller vectors
                #pragma omp simd
                for (int i = 0; i < this->size; i++) {
                    this->data[i] -= other.data[i];
                }
            }
            return *this;
        }

        inline friend vec<T> operator-(const vec<T>& a, const vec<T>& b) {
            if (a.data == nullptr || b.data == nullptr) {
                std::cout << "Error: vector is empty\n";
                exit(0);
            }
            if (a.size != b.size) {
                std::cout << "Error: Tried to subtract a vector of size " << a.size
                          << " with a vector of size " << b.size << "\n";
                exit(0);
            }

            // Use hardware acceleration if available
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    if (OpenCLAccelerator::isOpenCLAvailable() && OpenCLAccelerator::shouldUseGPU(a.size)) {
                        vec<T> result(a.size);
                        vec<T> negated_b(b.size);
                        
                        // Negate b and then add
                        #pragma omp simd
                        for (int i = 0; i < b.size; i++) {
                            negated_b.data[i] = -b.data[i];
                        }
                        
                        bool success = OpenCLAccelerator::addVectors(
                            a.data, negated_b.data, result.data, a.size
                        );
                        
                        if (success) {
                            return result;
                        }
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if hardware acceleration fails
                std::cerr << "OpenCL acceleration failed, using CPU: " << e.what() << std::endl;
            }

            // Use multithreaded CPU implementation for larger vectors
            if (a.size >= OpenCLAccelerator::getThreadThreshold()) {
                return subtractThreaded(a, b);
            }
            
            // Use sequential implementation for smaller vectors
            return subtractSequential(a, b);
        }

        inline friend vec<T> operator*(const vec<T>& a, T b) {
            vec<T> result(a.size);
            #pragma omp simd
            for (int i = 0; i < a.size; i++) {
                result.data[i] = a.data[i] * b;
            }
            return result;
        }

        inline friend vec<T> operator*(T a, const vec<T>& b) {
            return b * a;
        }

        inline friend T operator*(const vec<T>& a, const vec<T>& b) {
            if (a.data == nullptr || b.data == nullptr) {
                std::cout << "Error: vector is empty\n";
                exit(0);
            }
            if (a.size != b.size) {
                std::cout << "Error: Tried to multiply a vector of size " << a.size
                          << " with a vector of size " << b.size << "\n";
                exit(0);
            }

            // Use hardware acceleration if available
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    if (OpenCLAccelerator::isOpenCLAvailable() && OpenCLAccelerator::shouldUseGPU(a.size)) {
                        T result = 0;
                        bool success = OpenCLAccelerator::dotProduct(
                            a.data, b.data, a.size, result
                        );
                        
                        if (success) {
                            return result;
                        }
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if hardware acceleration fails
                std::cerr << "OpenCL acceleration failed, using CPU: " << e.what() << std::endl;
            }

            // CPU implementation as fallback
            T sum = 0;
            #pragma omp parallel for simd reduction(+:sum)
            for (int i = 0; i < a.size; i++) {
                sum += a.data[i] * b.data[i];
            }
            return sum;
        }

        inline friend vec<T> element_mult(const vec<T>& a, const vec<T>& b) {
            if (a.data == nullptr || b.data == nullptr) {
                std::cout << "Error: vector is empty\n";
                exit(0);
            }
            if (a.size != b.size) {
                std::cout << "Error: Tried to multiply a vector of size " << a.size
                          << " with a vector of size " << b.size << "\n";
                exit(0);
            }

            // Use hardware acceleration if available
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    if (OpenCLAccelerator::isOpenCLAvailable() && OpenCLAccelerator::shouldUseGPU(a.size)) {
                        vec<T> result(a.size);
                        bool success = OpenCLAccelerator::elementMultiply(
                            a.data, b.data, result.data, a.size
                        );
                        
                        if (success) {
                            return result;
                        }
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if hardware acceleration fails
                std::cerr << "OpenCL acceleration failed, using CPU: " << e.what() << std::endl;
            }

            // CPU implementation as fallback
            vec<T> result(a.size);
            #pragma omp parallel for simd
            for (int i = 0; i < a.size; i++) {
                result.data[i] = a.data[i] * b.data[i];
            }
            return result;
        };
        inline T max(){
            T max_val = data[0];
            for (int i = 1; i < size; i++) {
                if (data[i] > max_val) {
                    max_val = data[i];
                }
            }
            return max_val;
        }

        inline T mean(){
            T sum = 0;
            for (int i = 0; i < size; i++) {
                sum += data[i];
            }
            return sum / size;
        }

        inline T min(){
            T min_val = data[0];
            for (int i = 1; i < size; i++) {
                if (data[i] < min_val) {
                    min_val = data[i];
                }
            }
            return min_val;
        }
        inline T sum(){
            T sum = 0;
            for(int i =0; i < size; i++){
                sum+=data[i];
            }
            return sum;
        }
        inline friend vec<T> operator-(vec<T> a, T c){
            vec<T> result(a.size);
            for(int i = 0; i < a.size; i++){
                result(i) = a(i) - c;
            }
            return result;
        }
};

template <typename T>
vec<T>::vec(const vec<T> &other) {
    data = new T[other.size];
    std::memcpy(data, other.data, sizeof(T) * other.size);
    size = other.size;
}

template <typename T>
vec<T> &vec<T>::operator=(vec<T>& other) {
    if (this != &other) {
        delete[] data;
        data = new T[other.size];
        std::memcpy(data, other.data, sizeof(T) * other.size);
        size = other.size;
    }
    return *this;
}

template <typename T>
vec<T> &vec<T>::operator=(const vec<T>& other) {
    if (this != &other) {
        delete[] data;
        data = new T[other.size];
        std::memcpy(data, other.data, sizeof(T) * other.size);
        size = other.size;
    }
    return *this;
}

template <typename T>
vec<T>::vec(vec<T> &&other) noexcept : size(0), data(nullptr) {
    data = other.data;
    size = other.size;
    other.data = nullptr;
    other.size = 0;
}

template <typename T>
vec<T> &vec<T>::operator=(vec<T>&& other) noexcept {
    if (this != &other) {
        delete[] data;
        data = other.data;
        size = other.size;
        other.data = nullptr;
        other.size = 0;
    }
    return *this;
}

template <typename T>
T &vec<T>::operator()(int i) {
    if (data == nullptr) {
        std::cout << "Error: Vector is empty\n";
        exit(0);
    }
    if (i < 0 || i >= size) {
        std::cout << "Error: Tried to access element " << i 
                  << " in vector of size " << size << "\n";
        exit(0);
    }
    return data[i];
}

template <typename T>
const T &vec<T>::operator()(int i) const {
    if (data == nullptr) {
        std::cout << "Error: Vector is empty\n";
        exit(0);
    }
    if (i < 0 || i >= size) {
        std::cout << "Error: Tried to access element " << i 
                  << " in vector of size " << size << "\n";
        exit(0);
    }
    return data[i];
}

template <typename T>
void vec<T>::printout() {
    if (data == nullptr) {
        std::cout << "Vector is empty\n";
        return;
    }
    
    for (int i = 0; i < size; i++) {
        std::cout << data[i] << '\n';
    }
    std::cout << '\n';
}

template <typename T>
vec<T>::~vec() {
    delete[] data;
    data = nullptr;
    size = 0;
}
