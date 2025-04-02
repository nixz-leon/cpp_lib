#pragma once 
#ifndef matrix_hpp
#include "matrix.hpp"
#endif

template <class T>
class matricies {
private:
    matrix<T>* mats;
    int num_matricies;
    int size_n;
    int size_m;
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

    inline matricies<T> back(){return mats[num_matricies-1];};
    void printout();
    matricies<T> subset(int start_index, int end_index); // Subset function
};

//default
template <class T>
matricies<T>::matricies() : mats(nullptr), num_matricies(0), size_n(0), size_m(0) {};

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
inline matricies<T>::matricies(const matricies<T> &other) : num_matricies(other.num_matricies), size_n(other.size_n), size_m(other.size_m){
    mats = (matrix<T>*)std::calloc(num_matricies, sizeof(other.mats[0]));
    for(int i =0; i < num_matricies; ++i){
        mats[i] = other.mats[i];
    }

};
//copy assignment
template <class T>
matricies<T>& matricies<T>::operator=(const matricies<T>& other){
    if(this != &other){
        if(mats != nullptr){
            delete[] mats;
        }
        num_matricies = other.num_matricies;
        size_n = other.size_n;
        size_m = other.size_m;
        mats = (matrix<T>*)std::calloc(num_matricies, sizeof(other.mats[0]));
        if(!mats){
            throw std::runtime_error("Memory allocation failed");
        }
        for(int i = 0; i < num_matricies; ++i){
            mats[i] = other.mats[i];
        }
    }
    return *this;

};
//move constructor
template <class T>
inline matricies<T>::matricies(matricies<T> &&other) noexcept : mats(other.mats), size_n(other.size_n), size_m(other.size_m), num_matricies(other.num_matricies) {
    other.mats = nullptr;
    other.num_matricies = 0;
    other.size_m = 0;
    other.size_n = 0;
};

//move assignment
template <class T>
matricies<T>& matricies<T>::operator=(matricies<T>&& other) noexcept{
    if(this != &other){
        if(mats != nullptr){
            for(int i =0; i < num_matricies; ++i){
                mats[i].~matrix();
            }
            std::free(mats);
            mats = nullptr;
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
inline matricies<T>::~matricies(){
    if(mats != nullptr){
        for(int i =0; i < num_matricies; ++i){
            mats[i].~matrix();
        }
        std::free(mats);
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
