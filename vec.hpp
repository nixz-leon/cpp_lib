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

template <typename T> class vec{
    public: //this are memory functions
        vec();//default 
        vec(int n); //normal
        vec(const vec<T> &other);// copy
        vec<T>& operator=(vec<T>& other); //copy assigment
        vec(vec<T> &&other) noexcept;//move
        vec<T>& operator=(vec<T>&& other) noexcept;// move assignment 
        vec<T> operator=(_init_list_with_square_brackets<T> other);
        T& operator()(int a);
        const T& operator()(int a)const;
        inline void fft();
        inline void ifft();
        ~vec();
        inline void printout();
        inline const T back();
        int size;
    private:
        T *data;
        
    public: //these will be the math operator functions
        
        inline vec<T> operator+=(vec<T> &other);
        inline vec<T> operator-=(vec<T> &other);
        inline vec<T> operator*=(T &c);
        inline vec<T> operator*=(const T &c);
        inline vec<T> operator++();
        inline void row_swap(int a, int b);
        inline void row_add(int a, int b, T fac);
        inline T sum();
        inline void set(T num);

        inline friend vec<T> operator+(vec<T> a, vec<T> b){
            if(a.size != b.size){std::cout << "Tried to add a vector of size " << b.size << "  to a vector of size " << a.size << '\n';/*exit(0)*/;}
            vec<T> temp(a.size);for(int i=0; i < a.size;i++){temp(i) = a(i) + b(i);}return temp;
        };
        inline friend vec<T> operator-(vec<T> a, vec<T> b){
            if(a.size != b.size){std::cout << "Tried to subtract a vector of size " << b.size << "  from a vector of size " << a.size << '\n';exit(0);}
            vec<T> temp(a.size);for(int i=0; i < a.size;i++){temp(i) = a(i) - b(i);}return temp;
        };
        inline friend T operator*(vec<T> a, vec<T> b){
            if(a.size != b.size){std::cout << "Tried to multiply a vector of size " << b.size << "  with a vector of size " << a.size << '\n';exit(0);}
            T sum =0;for(int i=0; i < a.size;i++){sum += a(i) * b(i);}return sum;
        };
        inline friend vec<T> operator*(vec<T> a, T b){vec<T> temp(a);temp*=b;return temp;};
        inline friend vec<T> operator*(T a, vec<T> b){vec<T> temp(b);temp*=a;return temp;};
        inline friend vec<T> operator/(vec<T> a, T b){vec<T> temp(a);temp/=b;return temp;};

};

template <class T>
vec<T>::vec(){
    std::cout << "null_constructor\n";
    data = nullptr;
    size = 0;
};

template <class T>
vec<T>::vec(int n){
    data = new T[n];
    std::memset(data, 0, sizeof(T)*n);
    //data = (T*)std::calloc(n,sizeof(T));
    size = n;
};

template <class T>
vec<T>::vec(const vec<T> &other){
    size = other.size;
    data = new T[size];
    std::memcpy(data, other.data, sizeof(T)*other.size);
};

template <class T>
inline vec<T> &vec<T>::operator=(vec<T> &other){
    if(this != &other){
        delete[] data;
        size = other.size;
        data = new T[size];
        std::memcpy(data, other.data, sizeof(T)*size);
    }
    return *this;
};

template <class T>
vec<T>::vec(vec<T> &&other)noexcept
    :size(0)
    ,data(nullptr)
{
    data = other.data; // reassigning ownership of pointer
    size = other.size; // redefining size 
    //std::cout << size << " new size\n";
    other.data = nullptr; //releasing ownership of pointer from object passed in
    size = 0; // setting size to 0 to relfect size;
};

template <class  T>
inline vec<T> &vec<T>::operator=(vec<T> &&other)noexcept{
    if(this != &other){
        delete[] data;
        data  = other.data;
        size = other.size;
        //std::cout << size <<" new size\n";
        other.data = nullptr;
        other.size = 0;
    }
    return *this;
};

template <class T>
inline vec<T> vec<T>::operator=(_init_list_with_square_brackets<T> other)
{
    this->size = other.size();
    for(int i = 0; i < other.size(); i ++){
        this->data[i] = other[i];
    }
    return *this;
};

template <class T>
inline T &vec<T>::operator()(int a)
{
    if(data == nullptr){std::cout << "no data to be accessed\n"; exit(0);}
    if(size == 0){std::cout << "empty vec\n";exit(0);}//checking if the object can be accessed in the first place 
    if(a > size-1){std::cout << "Tried to access vec elm " << a << ", But size is " << size-1 << '\n';exit(0);} // bounds checking
    return data[a];
};

template <class T>
inline const T& vec<T>::operator()(int a) const
{
    if(data == nullptr){std::cout << "no data to be accessed\n"; exit(0);}
    if(size == 0){std::cout << "empty vec\n";exit(0);}//checking if the object can be accessed in the first place 
    if(a > size-1){std::cout << "Tried to access vec elm " << a << ", But size is " << size-1 << '\n';exit(0);} //bounds checking
    return data[a];
};

template <class T>
inline void vec<T>::fft(){
    fftr1(this->data, this->size, 1);
};

template <class T>
inline void vec<T>::ifft(){
    fftr1(this->data, this->size, -1);
};

template <class T>
inline vec<T> fft(vec<T> a){
    vec<T> temp(a);
    temp.fft();
    return temp;
};

template <class T>
inline vec<T> ifft(vec<T> a){
    vec<T> temp(a);
    temp.fft();
    return temp;
};

template <class T>
inline vec<T>::~vec()
{
    if(data != nullptr){
        delete[] data;
        size = 0;
    }
};

template <class T>
inline vec<T> per_elm(vec<T> a, vec<T> b){
    vec<T> result(a.size);
    for(int i =0; i < a.size; i++){
        result(i) = a(i) * b(i);
    } 
    return result;
};

template <class T>
inline void vec<T>::printout(){
    if(data != nullptr){
        for(int i=0; i < size; i++){
        std::cout << data[i] << '\n';
        }
        std::cout << '\n';
    }
};

template <class T>
inline const T vec<T>::back()
{
    return data[size-1];
};

template <class T>
inline void vec<T>::row_swap(int a, int b){
    if(a > size-1){std::cout << "Tried to access elm " << a << ", But size is " << size-1 << '\n';exit(0);}
    if(b > size-1){std::cout << "Tried to access elm " << b << ", But size is " << size-1 << '\n';exit(0);}
    if(a==b){return;}
    T temp = data[a];data[a] = b; data[b]= temp;
};

template <class T>
inline void vec<T>::row_add(int a, int b, T fac){
    if(a > size-1){std::cout << "Tried to access elm " << a << ", But size is " << size-1 << '\n';exit(0);}
    if(b > size-1){std::cout << "Tried to access elm " << b << ", But size is " << size-1 << '\n';exit(0);}
    if(a==b){return;}
    data[a] += fac*data[b];
};

template <class T>
inline T vec<T>::sum(){
    T sum;
    sum = data[0];
    for(int i =1; i < size; i++){
        sum += data[i];
    }
    return sum;
}

template <class T>
inline void vec<T>::set(T num){
    for(int i = 0; i< size; i++){
        data[i] = num;
    }
};

template <class T>
inline vec<T> vec<T>::operator+=(vec<T> &other)
{
    if(this->size != other){std::cout << "Tried to add a vector of size " << other.size << "  to a vector of size " << this->size << '\n';/*exit(0);*/}
    for(int i =0; i< this->size; i++){
        this->data[i] += other(i);
    }
    return *this;
};

template <class T>
inline vec<T> vec<T>::operator-=(vec<T> &other)
{
    if(this->size != other){std::cout << "Tried to subtract a vector of size " << other.size << "  from a vector of size " << this->size << '\n';exit(0);}
    for(int i =0; i< this->size; i++){
        this->data[i] -= other(i);
    }
    return *this;
};

template <class T>
inline vec<T> vec<T>::operator*=(T &c)
{
    for(int i=0; i < this->size;i++){
        this->data[i] *= c;
    }
    return *this;
};

template <class T>
inline vec<T> vec<T>::operator*=(const T &c) { // Change here
    for (int i = 0; i < this->size; i++) {
        this->data[i] *= c;
    }
    return *this;
};

template <class T>
inline vec<T> vec<T>::operator++()
{
    for(int i =0; i < this->size; i++){
        this->data[i]++;
    }
};



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

    int num_of_vecs(){return num_vecs;};
    int size(){return size_of_vecs;};

    void printout();

};

// Default constructor
template <class T>
vecs<T>::vecs() : vectors(nullptr), num_vecs(0), size_of_vecs(0) {}

// Parameterized constructor
template <class T>
vecs<T>::vecs(int num_vec, int size) : num_vecs(num_vec), size_of_vecs(size) {
    //vectors = new vec<T>[num_vecs]; // Allocate array of vec<T>
    vec<T> temp(size);
    vectors = (vec<T>*)std::calloc(num_vec,sizeof(temp));
    for (int i = 0; i < num_vecs; ++i) {
        vectors[i] = temp; // Initialize each vec<T>
    }
}


// Copy constructor
template <class T>
vecs<T>::vecs(const vecs<T>& other) : num_vecs(other.num_vecs), size_of_vecs(other.size_of_vecs) {
    std::cout << "copy\n";
    vec<T> temp(size_of_vecs);
    vectors = (vec<T>*)std::calloc(num_vecs,sizeof(temp));
    for (int i = 0; i < num_vecs; ++i) {
        vectors[i] = other.vectors[i];
    }
}

// Copy assignment operator
template <class T>
vecs<T>& vecs<T>::operator=(const vecs<T>& other) {
    if (this != &other) {
        if(vectors != nullptr) {
            delete[] vectors;
        }
        num_vecs = other.num_vecs;
        size_of_vecs = other.size_of_vecs;        
        vectors = (vec<T>*)std::calloc(num_vecs,sizeof(vec<T>));
        if (!vectors) {
            throw std::runtime_error("Memory allocation failed");
        }
        
        for (int i = 0; i < num_vecs; ++i) {
            new (&vectors[i]) vec<T>(size_of_vecs);
            vectors[i] = other.vectors[i];
        }
    }
    return *this;
}

// Move constructor
template <class T>
vecs<T>::vecs(vecs<T>&& other) noexcept : vectors(other.vectors), num_vecs(other.num_vecs), size_of_vecs(other.size_of_vecs) {
    std::cout << "move\n";
    other.vectors = nullptr;
    other.num_vecs = 0;
    other.size_of_vecs = 0;
}

// Move assignment operator
template <class T>
vecs<T>& vecs<T>::operator=(vecs<T>&& other) noexcept {
    std::cout << "move assignment\n";
    if (this != &other) {
        delete[] vectors;
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
        for (int i = 0; i < num_vecs; ++i) {
            vectors[i].~vec<T>();
        }
        std::free(vectors);
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
void vecs<T>::printout(){
    std::string sep = "";
    for(int i =0; i < size_of_vecs;i++){
        std::cout << i << "  ";
        sep += "---";
    }
    sep.pop_back();
    std::cout << '\n' << sep << '\n';   
    for(int i =0; i < num_vecs; i++){
        for(int j =0; j < size_of_vecs; j++){
            std::cout << vectors[i](j) << "  ";
        }
        std::cout << '\n';
    }
}