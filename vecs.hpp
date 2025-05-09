#pragma once
#ifndef vecs_hpp
#include "vec.hpp"
#endif

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
    int num_of_vecs() const { return num_vecs; };
    int size() { return size_of_vecs; };
    int size() const { return size_of_vecs; };
    inline vec<T>& back() { return vectors[num_vecs-1]; };
    inline const vec<T>& back() const { return vectors[num_vecs-1]; };
    void printout();
    vecs<T> subset(int start_col, int end_col); // Subset function
    vecs<T> subset(int start_col, int end_col) const;
    vec<T> vectorize();
    vec<T> vectorize() const;
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
template <class T>
vecs<T> vecs<T>::subset(int start_index, int end_index) const{
    int new_num = end_index - start_index + 1;
    vecs<T> result(new_num, size_of_vecs);
    for (int i = start_index; i <= end_index; i++) {
        result(i - start_index) = vectors[i];
    }
    return result;
}

template <class T>
vec<T> vecs<T>::vectorize() {
    vec<T> result(num_vecs * size_of_vecs);
    for (int i = 0; i < num_vecs; i++) {
        for (int j = 0; j < size_of_vecs; j++) {
            result(i * size_of_vecs + j) = vectors[i](j);
        }
    }
    return result;  
}

template <class T>
vec<T> vecs<T>::vectorize() const {
    vec<T> result(num_vecs * size_of_vecs);
    for (int i = 0; i < num_vecs; i++) {
        for (int j = 0; j < size_of_vecs; j++) {
            result(i * size_of_vecs + j) = vectors[i](j);
        }
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