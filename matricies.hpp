#pragma once 
#ifndef matrix_hpp
#include "matrix.hpp"
#endif

template <class T>
class matricies {
private:
    int num_matricies;
    int size_n;
    int size_m;
    matrix<T>* mats;
public:
    matricies(); // Default constructor
    matricies(int num_matricies, int n, int m); // Normal constructor
    matricies(const matricies<T>& other); // Copy constructor
    matricies<T>& operator=(const matricies<T>& other); // Copy assignment
    matricies(matricies<T>&& other) noexcept; // Move constructor
    matricies<T>& operator=(matricies<T>&& other) noexcept; // Move assignment
    ~matricies(); // Destructor
    matrix<T>& operator()(int index); // Access operator
    const matrix<T>& operator()(int index) const;

    int size(){return num_matricies;};
    int size_col(){return size_m;};
    int size_row(){return size_n;};

    inline matrix<T> back(){return mats[num_matricies-1];};
    void printout();
    matricies<T> subset(int start_index, int end_index); // Subset function
};

//default
template <class T>
matricies<T>::matricies() : num_matricies(0), size_n(0), size_m(0), mats(nullptr) {};

//normal constructor
template <class T>
matricies<T>::matricies(int num_mats, int n, int m) : num_matricies(num_mats), size_n(n), size_m(m) {
    matrix<T> temp(n,m);
    mats = new matrix<T>[num_mats];  // Use new instead of calloc
    for(int i = 0; i < num_mats; ++i){
        mats[i] = temp;
    }
}
//copy constructor
template <class T>
inline matricies<T>::matricies(const matricies<T> &other) 
    : num_matricies(other.num_matricies), size_n(other.size_n), size_m(other.size_m) {
    mats = new matrix<T>[num_matricies];  // Use new[] instead of calloc
    for(int i = 0; i < num_matricies; ++i) {
        mats[i] = other.mats[i];
    }
}
//copy assignment
template <class T>
matricies<T>& matricies<T>::operator=(const matricies<T>& other) {
    if(this != &other) {
        matrix<T>* new_mats = new matrix<T>[other.num_matricies];
        for(int i = 0; i < other.num_matricies; ++i) {
            new_mats[i] = other.mats[i];
        }
        
        if(mats != nullptr) {
            delete[] mats;
        }
        
        mats = new_mats;
        num_matricies = other.num_matricies;
        size_n = other.size_n;
        size_m = other.size_m;
    }
    return *this;
}
//move constructor
template <class T>
inline matricies<T>::matricies(matricies<T> &&other) noexcept : num_matricies(other.num_matricies), size_n(other.size_n), size_m(other.size_m), mats(other.mats)  {
    other.mats = nullptr;
    other.num_matricies = 0;
    other.size_m = 0;
    other.size_n = 0;
};

//move assignment
template <class T>
matricies<T>& matricies<T>::operator=(matricies<T>&& other) noexcept {
    if(this != &other) {
        if(mats != nullptr) {
            delete[] mats;  // Use delete[] instead of free
        }
        mats = other.mats;
        num_matricies = other.num_matricies;
        size_m = other.size_m;
        size_n = other.size_n;
        other.mats = nullptr;
        other.num_matricies = 0;
        other.size_m = 0;
        other.size_n = 0;
    }
    return *this;
}
template <class T>
inline matricies<T>::~matricies() {
    if(mats != nullptr) {
        delete[] mats;  // Use delete[] instead of free
        mats = nullptr;
    }
}
template <class T>
inline matrix<T> &matricies<T>::operator()(int index) {
    if (index >= num_matricies || index < 0) {
        throw std::out_of_range("Index out of range");
    }
    return mats[index];
}
template <class T>
inline const matrix<T> &matricies<T>::operator()(int index) const {
    if (index >= num_matricies || index < 0) {
        throw std::out_of_range("Index out of range");
    }
    return mats[index];
};
template <class T>
inline void matricies<T>::printout() {
    for(int i = 0; i < num_matricies; ++i) {
        std::cout << "Matrix " << i << ":\n";
        mats[i].printout();
        std::cout << "\n";
    }
}
template <class T>
inline matricies<T> matricies<T>::subset(int start_index, int end_index)
{
    int new_num = end_index - start_index +1;
    matricies<T> result(new_num, size_n, size_m);
    for(int i = start_index; i <= end_index; i++){
        result(i-start_index) = mats[i];
    }    
    return result;
};
