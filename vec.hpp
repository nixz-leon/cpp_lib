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
#ifndef  random
#include <random>
#endif

// GPU acceleration check - include adapter if enabled
#ifdef GPU_ACCELERATION_ENABLED
#include "GPUVecAdapter.hpp"
#endif

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
    public: // Memory management functions
        vec();
        vec(const int size);
        vec(const vec<T> &other);
        vec<T>& operator=(const vec<T> &other);
        vec(vec<T> &&other) noexcept;
        vec<T>& operator=(vec<T> &&other) noexcept;
        vec<T> operator=(_init_list_with_square_brackets<T> other);
        T& operator()(int i);
        const T& operator()(int i) const;
        void printout();
        inline void fft();
        inline void ifft();
        inline const T back();
        ~vec();
        T *data; // Making public to allow GPU adapter direct access
        int size;
        T max() const;

    public: // Mathematical operations and utility functions
        // Vector operations
        inline vec<T> operator+=(const vec<T> &other);
        inline vec<T> operator-=(const vec<T> &other);
        inline vec<T> operator*=(T c);
        inline vec<T> operator/=(T c);
        inline vec<T> operator++();
        inline void row_swap(int a, int b);
        inline void row_add(int a, int b, T fac);
        inline T sum();
        inline void set(T num);
        
        // Vector-vector operations with GPU acceleration where available
        inline friend vec<T> operator+(vec<T> a, const vec<T> &b) {
            if (a.size != b.size) {
                std::cout << "Error: Tried to add vectors of different sizes: " 
                          << a.size << " and " << b.size << "\n";
                exit(0);
            }
            
            #ifdef GPU_ACCELERATION_ENABLED
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    // Use GPU for larger vectors
                    if (a.size >= 100000) {
                        return GPUVecAdapter::add(a, b);
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if GPU acceleration fails
                std::cerr << "GPU acceleration failed, using CPU: " << e.what() << std::endl;
            }
            #endif
            
            // CPU implementation as fallback
            vec<T> temp(a.size);
            #pragma omp simd
            for (int i = 0; i < a.size; i++) {
                temp(i) = a(i) + b(i);
            }
            return temp;
        }
        
        inline friend vec<T> operator-(vec<T> a, const vec<T> &b) {
            if (a.size != b.size) {
                std::cout << "Error: Tried to subtract vectors of different sizes: " 
                          << a.size << " and " << b.size << "\n";
                exit(0);
            }
            
            vec<T> temp(a.size);
            for (int i = 0; i < a.size; i++) {
                temp(i) = a(i) - b(i);
            }
            return temp;
        }
        
        // Dot product with GPU acceleration
        inline friend T operator*(const vec<T> &a, const vec<T> &b) {
            if (a.size != b.size) {
                std::cout << "Error: Tried to multiply a vector of size " << b.size 
                          << " with a vector of size " << a.size << "\n";
                exit(0);
            }
            
            #ifdef GPU_ACCELERATION_ENABLED
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    // Use GPU for larger vectors
                    if (a.size >= 100000) {
                        return GPUVecAdapter::dotProduct(a, b);
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if GPU acceleration fails
                std::cerr << "GPU acceleration failed, using CPU: " << e.what() << std::endl;
            }
            #endif
            
            // CPU implementation with optimizations
            T sum = 0;
            
            // Use SIMD instructions via compiler hints
            #pragma omp simd reduction(+:sum)
            for (int i = 0; i < a.size; i++) {
                sum += a(i) * b(i);
            }
            return sum;
        }
        
        // Element-wise multiplication
        inline friend vec<T> element_mult(const vec<T> &a, const vec<T> &b) {
            if (a.size != b.size) {
                std::cout << "Error: Tried element-wise multiplication of vectors with different sizes: " 
                          << a.size << " and " << b.size << "\n";
                exit(0);
            }
            
            #ifdef GPU_ACCELERATION_ENABLED
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    // Use GPU for larger vectors
                    if (a.size >= 100000) {
                        return GPUVecAdapter::elementMult(a, b);
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if GPU acceleration fails
                std::cerr << "GPU acceleration failed, using CPU: " << e.what() << std::endl;
            }
            #endif
            
            // CPU implementation
            vec<T> temp(a.size);
            for (int i = 0; i < a.size; i++) {
                temp(i) = a(i) * b(i);
            }
            return temp;
        }
        
        // Scalar operations
        inline friend vec<T> operator*(vec<T> a, T b) {
            vec<T> temp(a);
            temp *= b;
            return temp;
        }
        
        inline friend vec<T> operator*(T a, vec<T> b) {
            vec<T> temp(b);
            temp *= a;
            return temp;
        }
        
        inline friend vec<T> operator/(vec<T> a, T b) {
            vec<T> temp(a);
            temp /= b;
            return temp;
        }
        
        // Comparison operators
        inline friend bool operator==(vec<T> a, vec<T> b) {
            if (a.size != b.size) {
                return false;
            }
            for (int i = 0; i < b.size; i++) {
                if (a(i) != b(i)) return false;
            }
            return true;
        }
        
        inline friend bool operator!=(vec<T> a, vec<T> b) {
            if (a.size != b.size) {
                return true;
            }
            for (int i = 0; i < b.size; i++) {
                if (a(i) != b(i)) return true;
            }
            return false;
        }
        
        // Fill vector with random values
        inline void randomize(T min = -1.0, T max = 1.0) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dis(min, max);
            
            for (int i = 0; i < size; i++) {
                data[i] = dis(gen);
            }
        }
};

template <typename T>
vec<T>::vec() {
    data = nullptr;
    size = 0;
}

template <typename T>
vec<T>::vec(const int s) {
    data = new T[s];
    std::memset(data, 0, sizeof(T) * s);
    size = s;
}

template <typename T>
vec<T>::vec(const vec<T> &other) {
    data = new T[other.size];
    std::memcpy(data, other.data, sizeof(T) * other.size);
    size = other.size;
}

template <typename T>
vec<T> &vec<T>::operator=(const vec<T> &other) {
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
vec<T> &vec<T>::operator=(vec<T> &&other) noexcept {
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
vec<T> vec<T>::operator=(_init_list_with_square_brackets<T> other) {
    this->size = other.size();
    for (int i = 0; i < other.size(); i++) {
        this->data[i] = other[i];
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
inline void vec<T>::fft() {
    fftr1(this->data, this->size, 1);
}

template <typename T>
inline void vec<T>::ifft() {
    fftr1(this->data, this->size, -1);
}

template <typename T>
inline const T vec<T>::back() {
    return data[size-1];
}

template <typename T>
vec<T>::~vec() {
    delete[] data;
    data = nullptr;
    size = 0;
}

template <typename T>
inline vec<T> vec<T>::operator+=(const vec<T> &other) {
    if (size != other.size) {
        std::cout << "Error: Tried to add vectors of different sizes\n";
        exit(0);
    }
    
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        data[i] += other(i);
    }
    return *this;
}

template <typename T>
inline vec<T> vec<T>::operator-=(const vec<T> &other) {
    if (size != other.size) {
        std::cout << "Error: Tried to subtract vectors of different sizes\n";
        exit(0);
    }
    
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        data[i] -= other(i);
    }
    return *this;
}

template <typename T>
inline vec<T> vec<T>::operator*=(T c) {
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        data[i] *= c;
    }
    return *this;
}

template <typename T>
inline vec<T> vec<T>::operator/=(T c) {
    if (c == 0) {
        std::cout << "Error: Division by zero\n";
        exit(0);
    }
    
    T inv = 1.0 / c;
    #pragma omp simd
    for (int i = 0; i < size; i++) {
        data[i] *= inv;  // Multiply by inverse is faster than division
    }
    return *this;
}

template <typename T>
inline vec<T> vec<T>::operator++() {
    for (int i = 0; i < this->size; i++) {
        this->data[i]++;
    }
    return *this;
}

template <typename T>
inline void vec<T>::row_swap(int a, int b) {
    if (a > size-1) {
        std::cout << "Tried to access elm " << a << ", But size is " << size-1 << '\n';
        exit(0);
    }
    if (b > size-1) {
        std::cout << "Tried to access elm " << b << ", But size is " << size-1 << '\n';
        exit(0);
    }
    if (a == b) {
        return;
    }
    T temp = data[a];
    data[a] = data[b];
    data[b] = temp;
}

template <typename T>
inline void vec<T>::row_add(int a, int b, T fac) {
    if (a > size-1) {
        std::cout << "Tried to access elm " << a << ", But size is " << size-1 << '\n';
        exit(0);
    }
    if (b > size-1) {
        std::cout << "Tried to access elm " << b << ", But size is " << size-1 << '\n';
        exit(0);
    }
    if (a == b) {
        return;
    }
    data[a] += fac * data[b];
}

template <typename T>
inline T vec<T>::sum() {
    T sum = data[0];
    for (int i = 1; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

template <typename T>
inline void vec<T>::set(T num) {
    for (int i = 0; i < size; i++) {
        data[i] = num;
    }
}

template <typename T>
T vec<T>::max() const {
    if (size == 0) {
        throw std::runtime_error("Cannot find max of an empty vector");
    }
    T max_value = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > max_value) {
            max_value = data[i];
        }
    }
    return max_value;
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
    inline vec<T> back() { return vectors[num_vecs-1]; };
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
        delete[] vectors;  // Use delete[] instead of free
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
    delete[] vectors;  // Use delete[] instead of free
    vectors = nullptr;
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

template <class T>
inline vec<T> per_elm(vec<T> a, vec<T> b){
    vec<T> result(a.size);
    for(int i =0; i < a.size; i++){
        result(i) = a(i) * b(i);
    } 
    return result;
};


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