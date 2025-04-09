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

// Include HardwareAccelerator header
#include "HardwareAccelerator.hpp"


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
        vec(const vec<T> &other);// copy
        vec<T>& operator=(vec<T>& other); //copy assigment
        vec(vec<T> &&other) noexcept;//move
        vec<T>& operator=(vec<T>&& other) noexcept;// move assignment 
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
            const int block = HardwareAccelerator::getCacheBlockSize();
            
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
            const int block = HardwareAccelerator::getCacheBlockSize();
            
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
            #ifdef HARDWARE_ACCELERATION_ENABLED
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    vec<T> result(a.size);
                    bool success = HardwareAccelerator::addVectors(
                        a.data, b.data, result.data, a.size
                    );
                    
                    if (success) {
                        return result;
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if hardware acceleration fails
                std::cerr << "Hardware acceleration failed, using CPU: " << e.what() << std::endl;
            }
            #endif

            // Use multithreaded CPU implementation for larger vectors
            if (a.size >= HardwareAccelerator::getThreadThreshold()) {
                return addThreaded(a, b);
            }
            
            // Use sequential implementation for smaller vectors
            return addSequential(a, b);
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
            #ifdef HARDWARE_ACCELERATION_ENABLED
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    vec<T> result(a.size);
                    vec<T> negated_b(b.size);
                    
                    // Negate b and then add
                    #pragma omp simd
                    for (int i = 0; i < b.size; i++) {
                        negated_b.data[i] = -b.data[i];
                    }
                    
                    bool success = HardwareAccelerator::addVectors(
                        a.data, negated_b.data, result.data, a.size
                    );
                    
                    if (success) {
                        return result;
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if hardware acceleration fails
                std::cerr << "Hardware acceleration failed, using CPU: " << e.what() << std::endl;
            }
            #endif

            // Use multithreaded CPU implementation for larger vectors
            if (a.size >= HardwareAccelerator::getThreadThreshold()) {
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
            #ifdef HARDWARE_ACCELERATION_ENABLED
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    T result = 0;
                    bool success = HardwareAccelerator::dotProduct(
                        a.data, b.data, a.size, result
                    );
                    
                    if (success) {
                        return result;
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if hardware acceleration fails
                std::cerr << "Hardware acceleration failed, using CPU: " << e.what() << std::endl;
            }
            #endif

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
            #ifdef HARDWARE_ACCELERATION_ENABLED
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    vec<T> result(a.size);
                    bool success = HardwareAccelerator::multiplyVectors(
                        a.data, b.data, result.data, a.size
                    );
                    
                    if (success) {
                        return result;
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if hardware acceleration fails
                std::cerr << "Hardware acceleration failed, using CPU: " << e.what() << std::endl;
            }
            #endif

            // CPU implementation as fallback
            vec<T> result(a.size);
            #pragma omp parallel for simd
            for (int i = 0; i < a.size; i++) {
                result.data[i] = a.data[i] * b.data[i];
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

// Container class for multiple vectors
template <class T>
class vecs {
private:
    vec<T>* vectors;
    int num_vecs;
    int size_of_vecs;

public:
    vecs(); // Default constructor
    vecs(int num_vec, int size); // Normal constructor
    vecs(const vecs<T>& other); // Copy constructor
    vecs<T>& operator=(const vecs<T>& other); // Copy assignment
    vecs(vecs<T>&& other) noexcept; // Move constructor
    vecs<T>& operator=(vecs<T>&& other) noexcept; // Move assignment
    ~vecs(); // Destructor
    vec<T>& operator()(int index); // Access operator
    const vec<T>& operator()(int index) const;
    int num_of_vecs() { return num_vecs; };
    int size() { return size_of_vecs; };
    inline vec<T>& back() { return vectors[num_vecs-1]; };
    void printout();
    vecs<T> subset(int start_col, int end_col); // Subset function
};

// Default constructor
template <class T>
vecs<T>::vecs() : vectors(nullptr), num_vecs(0), size_of_vecs(0) {}

// Parameterized constructor
template <class T>
vecs<T>::vecs(int num_vec, int size) : num_vecs(num_vec), size_of_vecs(size) {
    vectors = new vec<T>[num_vecs];  // Use new instead of calloc
    for (int i = 0; i < num_vecs; ++i) {
        vectors[i] = vec<T>(size);  // Initialize each vector with proper size
    }
}

// Copy constructor
template <class T>
vecs<T>::vecs(const vecs<T>& other) : num_vecs(other.num_vecs), size_of_vecs(other.size_of_vecs) {
    vectors = new vec<T>[num_vecs];  // Use new instead of calloc
    for (int i = 0; i < num_vecs; ++i) {
        vectors[i] = other.vectors[i];
    }
}

// Copy assignment operator
template <class T>
vecs<T>& vecs<T>::operator=(const vecs<T>& other) {
    if (this != &other) {
        if (vectors != nullptr) {
            delete[] vectors;
        }
        num_vecs = other.num_vecs;
        size_of_vecs = other.size_of_vecs;        
        vectors = new vec<T>[num_vecs];  // Use new instead of calloc
        
        for (int i = 0; i < num_vecs; ++i) {
            vectors[i] = other.vectors[i];
        }
    }
    return *this;
}

// Move constructor
template <class T>
vecs<T>::vecs(vecs<T>&& other) noexcept : vectors(other.vectors), num_vecs(other.num_vecs), size_of_vecs(other.size_of_vecs) {
    other.vectors = nullptr;
    other.num_vecs = 0;
    other.size_of_vecs = 0;
}

// Move assignment operator
template <class T>
vecs<T>& vecs<T>::operator=(vecs<T>&& other) noexcept {
    if (this != &other) {
        if (vectors != nullptr) {
            delete[] vectors;
        }
        vectors = other.vectors;
        num_vecs = other.num_vecs;
        size_of_vecs = other.size_of_vecs;
        other.vectors = nullptr;
        other.num_vecs = 0;
        other.size_of_vecs = 0;
    }
    return *this;
}

// Destructor
template <class T>
vecs<T>::~vecs() {
    if (vectors != nullptr) {
        delete[] vectors;
        vectors = nullptr;
    }
}

// Access operator
template <class T>
vec<T>& vecs<T>::operator()(int index) {
    if (index >= num_vecs || index < 0) {
        throw std::out_of_range("Index out of range");
    }
    return vectors[index];
}

template <class T>
const vec<T>& vecs<T>::operator()(int index) const {
    if (index >= num_vecs || index < 0) {
        throw std::out_of_range("Index out of range");
    }
    return vectors[index];
}

template <class T>
void vecs<T>::printout() {
    std::string sep = "";
    for (int i = 0; i < size_of_vecs; i++) {
        std::cout << i << "  ";
        sep += "---";
    }
    sep.pop_back();
    std::cout << '\n' << sep << '\n';   
    for (int i = 0; i < num_vecs; i++) {
        for (int j = 0; j < size_of_vecs; j++) {
            std::cout << vectors[i](j) << "  ";
        }
        std::cout << '\n';
    }
}

template <class T>
vecs<T> vecs<T>::subset(int start_index, int end_index) {
    int new_num = end_index - start_index + 1;
    vecs<T> result(new_num, size_of_vecs);
    for (int i = start_index; i <= end_index; i++) {
        result(i - start_index) = vectors[i];
    }
    return result;
}



template <typename T>
    std::pair<vecs<T>, vecs<T>> train_test_split(vecs<T>& data, double test_size = 0.2) {
        int total_size = data.size();
        int test_count = static_cast<int>(total_size * test_size);
        int train_count = total_size - test_count;
        
        // Create random indices
        std::vector<int> indices(total_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        // Create training and test sets
        vecs<T> train_set(data.num_of_vecs(), train_count);
        vecs<T> test_set(data.num_of_vecs(), test_count);
        
        // Fill training set
        for (int i = 0; i < train_count; i++) {
            for (int j = 0; j < data.num_of_vecs(); j++) {
                train_set(j)(i) = data(j)(indices[i]);
            }
        }
        
        // Fill test set
        for (int i = 0; i < test_count; i++) {
            for (int j = 0; j < data.num_of_vecs(); j++) {
                test_set(j)(i) = data(j)(indices[i + train_count]);
            }
        }
        
        return {train_set, test_set};
    };