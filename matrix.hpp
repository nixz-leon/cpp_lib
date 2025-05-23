#pragma once
#ifndef vecs_hpp
#include "vecs.hpp"
#endif
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "OpenCLAccelerator.hpp"

// Define constants for hardware acceleration
#define CACHE_BLOCK_SIZE OpenCLAccelerator::getCacheBlockSize()
#define THREAD_SIZE_THRESHOLD OpenCLAccelerator::getThreadThreshold()
#define GPU_SIZE_THRESHOLD OpenCLAccelerator::getGPUThreshold()

template <typename T>
class matrix {  
    public:
        matrix() : row(0), col(0), data(nullptr) {} //default 
        matrix(int n, int m) : row(n), col(m) {data = new T[row * col]();}; //normal
        matrix(vecs<T> &vectors, bool row_major = false);
        matrix(vec<T> &vector, bool col_matrix = false);
        matrix(const matrix<T> &other);// copy
        matrix<T>& operator=(matrix<T>& other); //copy assigment
        matrix<T>& operator=(const matrix<T>& other);
        matrix(matrix<T> &&other) noexcept;//move
        matrix<T>& operator=(matrix<T>&& other) noexcept;// move assignment 
        T& operator()(int row, int col);
        const T& operator()(int row, int col)const;
        vec<T> get_row(int r);
        vec<T> set_row(int r, const vec<T> &v);
        void printout();
        ~matrix();
        int row;
        int col;
        T *data; // Making public to allow GPU adapter direct access

    private:
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

    public:
        inline void set_diag(T c);
        inline void row_swap(int a, int b);
        inline void row_add(int a, int b, T fac); 
        inline matrix<T> operator/=(T c);
        inline matrix<T> operator *=(T c);
        inline matrix<T>& operator+=(const matrix<T>& other);
        inline matrix<T>& operator-=(const matrix<T>& other);
        inline vec<T> operator*(vec<T> b);
        inline vec<T> operator*(vec<T> b) const;
        inline friend vec<T> vectorize(matrix<T> a){
            vec<T> temp(a.col * a.row);
            for(int i = 0; i < a.col; i++){
                for(int j = 0; j < a.row; j++){
                    temp((i*a.row)+j) = a(i,j);
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
            matrix<T> result(a.row, a.col);
            for(int i = 0; i < a.row * a.col; ++i){
                result.data[i] = a.data[i] + b.data[i];
            }
            return result;
        };
        inline friend matrix<T> operator-(matrix<T> a, matrix<T> b){
            if((a.col!=b.col)||(a.row!=b.row)){std::cout << "Tried to subtract a " << b.row << ','<<b.col<< " matrix from a " << a.row << ',' << a.col << " matrix\n";exit(0);}
            matrix<T> result(a.row, a.col);
            for(int i = 0; i < a.row * a.col; ++i){
                result.data[i] = a.data[i] - b.data[i];
            }
            return result;
        };
        inline friend matrix<T> operator*(matrix<T> a, matrix<T> b) {
            if (a.col != b.row) {
                std::cout << "Invalid matrix dimensions\n";
                exit(0);
            }
        
            const size_t total_elements = a.row * b.col;
            const size_t total_operations = total_elements * a.col;
            
            // Use hardware acceleration if available
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    if (OpenCLAccelerator::isOpenCLAvailable() && OpenCLAccelerator::shouldUseGPU(total_operations)) {
                        matrix<T> result(a.row, b.col);
                        bool success = OpenCLAccelerator::multiplyMatrices(
                            a.data, a.row, a.col,
                            b.data, b.row, b.col,
                            result.data
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
            
            // Use multithreaded CPU implementation for larger matrices
            if ((int)total_operations >= THREAD_SIZE_THRESHOLD * THREAD_SIZE_THRESHOLD) {
                return multiplyThreaded(a, b);
            }
            
            // Use sequential implementation for smaller matrices
            return multiplySequential(a, b);
        };
        inline friend matrix<T> operator*(matrix<T> a, T b){matrix<T>temp(a); temp*=b;return temp;};
        inline friend matrix<T> operator*(T b, matrix<T> a){matrix<T>temp(a); temp*=b;return temp;};
        inline void add_per_row(vec<T> v);
        inline void add_per_col(vec<T> v);
        inline T sum_elms(){
            T sum = 0;
            for(int i = 0; i < row; i++){
                for(int j = 0; j < col; j++){
                    sum += data[(col*i)+j];
                }
            }
            return sum;
        };

        inline void multiply(const matrix<T>& other, matrix<T>& result) const {
            if (col != other.row || result.row != row || result.col != other.col) {
                throw std::invalid_argument("Matrix dimensions don't match for multiplication");
            }

            // Use hardware acceleration if available
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    if (OpenCLAccelerator::isOpenCLAvailable() && OpenCLAccelerator::shouldUseGPU(row * other.col * col)) {
                        bool success = OpenCLAccelerator::multiplyMatrices(
                            data, row, col,
                            other.data, other.row, other.col,
                            result.data
                        );
                        
                        if (success) {
                            return;
                        }
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if hardware acceleration fails
                std::cerr << "OpenCL acceleration failed, using CPU: " << e.what() << std::endl;
            }

            const int block = CACHE_BLOCK_SIZE;
            
            // Use cache-friendly blocking
            for (int i = 0; i < row; i += block) {
                for (int k = 0; k < col; k += block) {
                    for (int j = 0; j < other.col; j += block) {
                        // Block multiplication
                        for (int ii = i; ii < std::min(i + block, row); ++ii) {
                            for (int kk = k; kk < std::min(k + block, col); ++kk) {
                                T r = data[ii * col + kk];
                                for (int jj = j; jj < std::min(j + block, other.col); ++jj) {
                                    result.data[ii * other.col + jj] += r * other.data[kk * other.col + jj];
                                }
                            }
                        }
                    }
                }
            }
        }

        // For parallel operations
        inline bool shouldUseThreads() const {
            return (row * col) >= THREAD_SIZE_THRESHOLD;
        }

        // Computes transpose(a) * b without explicitly creating the transpose
        template <typename U>
        friend matrix<U> transpose_multiply(const matrix<U>& a, const matrix<U>& b);
        
        // Computes transpose(a) * v without explicitly creating the transpose
        template <typename U>
        friend vec<U> transpose_multiply_vec(const matrix<U>& a, const vec<U>& v);
};

template <typename T>
matrix<T>::matrix(vecs<T> &vectors, bool row_major){
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
}
template <typename T>
inline matrix<T>::matrix(vec<T> &vector, bool col_matrix){
    if(col_matrix){
        row = vector.size;
        col = 1;
        data = new T[row*col];
        for(int i = 0; i < row; i++){
            data[(col*i)+0] = vector(i);
        }
    }else{
        row = 1;
        col = vector.size;
        data = new T[row*col];
        for(int i = 0; i < col; i++){
            data[(col*0)+i] = vector(i);
        }
    }
};

template <typename T>
matrix<T>::matrix(const matrix<T> &other){
    data = new T[other.row*other.col];
    std::memcpy(data, other.data, sizeof(T)*other.col*other.row);
    row = other.row;
    col = other.col;
};

template <typename T>
inline matrix<T> &matrix<T>::operator=(matrix<T> &other){
    if (this != &other) {  // Self-assignment check
        delete[] data;  // Free existing memory
        data = new T[other.row * other.col];  // Allocate new memory
        row = other.row;
        col = other.col;
        std::memcpy(data, other.data, sizeof(T)*col*row);
    }
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
inline vec<T> r(matrix<T> &a, int r)
{
    if(r > a.row-1){std::cout << "Tried to access row " << r << ", But size is " << a.row-1 << '\n';exit(0);}
    if(r < 0){std::cout << "tried to access row " << r << " which is negative\n";}
    return vec<T>(a.data + r*a.col, a.col);
}

template <typename T>
inline const vec<T> r(matrix<T> &a, int r)
{
    if(r > a.row-1){std::cout << "Tried to access row " << r << ", But size is " << a.row-1 << '\n';exit(0);}
    if(r < 0){std::cout << "tried to access row " << r << " which is negative\n";}
    return vec<T>(a.data + r*a.col, a.col);
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
    if (data != nullptr) {
        delete[] data;
        data = nullptr;
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
inline matrix<T>& matrix<T>::operator+=(const matrix<T>& other)
{
    if (this->row != other.row || this->col != other.col) {
        std::cout << "Error: Tried to add-assign matrices of incompatible dimensions. "
                  << " LHS: " << this->row << 'x' << this->col
                  << ", RHS: " << other.row << 'x' << other.col << "\n";
        exit(0); // Or throw an exception
    }
    if(this->data == nullptr || other.data == nullptr){
        std::cout << "Error: One or both matrices in += have null data\n";
        exit(0); // Or throw an exception
    }
    #pragma omp parallel for simd // Basic parallelization
    for (int i = 0; i < this->row * this->col; ++i) {
        this->data[i] += other.data[i];
    }
    return *this;
}

template <typename T>
inline matrix<T>& matrix<T>::operator-=(const matrix<T>& other)
{
    if (this->row != other.row || this->col != other.col) {
        std::cout << "Error: Tried to subtract-assign matrices of incompatible dimensions. "
                  << " LHS: " << this->row << 'x' << this->col
                  << ", RHS: " << other.row << 'x' << other.col << "\n";
        exit(0); // Or throw an exception
    }
    if(this->data == nullptr || other.data == nullptr){
        std::cout << "Error: One or both matrices in -= have null data\n";
        exit(0); // Or throw an exception
    }
    #pragma omp parallel for simd // Basic parallelization
    for (int i = 0; i < this->row * this->col; ++i) {
        this->data[i] -= other.data[i];
    }
    return *this;
}

template <typename T>
inline vec<T> matrix<T>::operator*(vec<T> b)
{
    if (this->data == nullptr) {
        std::cout << "Error: matrix is empty\n";
        exit(0);
    }
    if (b.size != this->col) {
        std::cout << "Error: Tried to multiply a matrix of size " << this->row << "x" << this->col
                  << " with a vector of size " << b.size << "\n";
        exit(0);
    }

    // Use hardware acceleration if available
    try {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            if (OpenCLAccelerator::isOpenCLAvailable() && OpenCLAccelerator::shouldUseGPU(this->row * this->col)) {
                vec<T> result(this->row);
                bool success = OpenCLAccelerator::multiplyMatrixVector(
                    this->data, this->row, this->col,
                    b.data, result.data
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
    vec<T> temp(this->row);
    T sum = 0;

    // Use cache-friendly loop ordering and vectorization hints
    #pragma omp parallel for simd
    for (int i = 0; i < this->row; i++) {
        sum = 0;
        for (int j = 0; j < this->col; j++) {
            sum += this->data[(i * this->col) + j] * b.data[j];
        }
        temp.data[i] = sum;
    }
    return temp;
}

template <typename T>
inline vec<T> matrix<T>::operator*(vec<T> b) const
{
    if (this->data == nullptr) {
        std::cout << "Error: matrix is empty\n";
        exit(0);
    }
    if (b.size != this->col) {
        std::cout << "Error: Tried to multiply a matrix of size " << this->row << "x" << this->col
                  << " with a vector of size " << b.size << "\n";
        exit(0);
    }

    // Use hardware acceleration if available
    try {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            if (OpenCLAccelerator::isOpenCLAvailable() && OpenCLAccelerator::shouldUseGPU(this->row * this->col)) {
                vec<T> result(this->row);
                bool success = OpenCLAccelerator::multiplyMatrixVector(
                    this->data, this->row, this->col,
                    b.data, result.data
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
    vec<T> temp(this->row);
    T sum = 0;

    // Use cache-friendly loop ordering and vectorization hints
    #pragma omp parallel for simd
    for (int i = 0; i < this->row; i++) {
        sum = 0;
        for (int j = 0; j < this->col; j++) {
            sum += this->data[(i * this->col) + j] * b.data[j];
        }
        temp.data[i] = sum;
    }
    return temp;
}

template <typename T>
inline matrix<T> outer_product(vec<T> a, vec<T> b){
    matrix<T> result(a.size, b.size);
    for(int i = 0; i < a.size; i++){
        for(int j = 0; j < b.size; j++){
            result(i, j) = a(i) * b(j);
        }
    }
    return result;
};
//need to add hardware acceleration for this
template <typename T>
inline matrix<T> hadamard(matrix<T> a, matrix<T> b){
    if((a.row != b.row)||(a.col != b.col)){
        std::cout << "Tried to multiply a " << a.row << ','<<a.col<< " matrix with a " << b.row << ',' << b.col << " matrix\n";
        exit(0);
    }
    matrix<T> result(a.row, a.col);
    for(int i = 0; i < a.row; i++){
        for(int j = 0; j < a.col; j++){
            result(i, j) = a(i, j) * b(i, j);
        }
    }
    return result;  
}

template <typename T>
inline vec<T> matrix<T>::get_row(int r)
{
    if(r > row-1){std::cout << "Tried to access row " << r << ", But size is " << row-1 << '\n';exit(0);}
    if(r < 0){std::cout << "tried to access row " << r << " which is negative\n";}
    return vec<T>(data + r*col, col);
}

template <typename T>
inline vec<T> matrix<T>::set_row(int r, const vec<T> &v)
{
    if(r > row-1){std::cout << "Tried to access row " << r << ", But size is " << row-1 << '\n';exit(0);}
    if(r < 0){std::cout << "tried to access row " << r << " which is negative\n";}
    if(v.size != col){std::cout << "Vector size " << v.size << " does not match matrix column size " << col << '\n';exit(0);}
    
    // Copy the vector data into the matrix row
    for(int i = 0; i < col; i++) {
        data[r*col + i] = v(i);
    }
    return v;
}

template<typename T>
inline void matrix<T>::add_per_row(vec<T> v){
    if(col != v.size){
        std::cerr << "vector not the right size\n";
    }
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            data[(i*col)+j] += v(j);
        }
    }
};
template <typename T>
inline void matrix<T>::add_per_col(vec<T> v){
    if(row != v.size){
        std::cerr << "vector not the right size\n";
    }
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            data[(i*col)+j] += v(i);
        }
    }
}

template <typename T>
matrix<T> transpose_multiply(const matrix<T>& a, const matrix<T>& b) {
    // Check dimensions for compatibility
    if (a.row != b.row) {
        std::cout << "Error: Incompatible dimensions for transpose_multiply. "
                  << "Matrix A: " << a.row << "x" << a.col 
                  << ", Matrix B: " << b.row << "x" << b.col << std::endl;
        exit(0);
    }
    
    // Result will have dimensions (a.col x b.col)
    matrix<T> result(a.col, b.col);
    
    // Use hardware acceleration if available for float type
    if constexpr (std::is_same_v<T, float>) {
        try {
            if (OpenCLAccelerator::isOpenCLAvailable() && 
                OpenCLAccelerator::shouldUseGPU(a.col * b.col * a.row)) {
                bool success = OpenCLAccelerator::transposeMultiplyMatrices(
                    a.data, a.row, a.col,
                    b.data, b.row, b.col,
                    result.data
                );
                
                if (success) {
                    return result;
                }
                // Fall through to CPU implementation if GPU failed
            }
        } catch (const std::exception& e) {
            std::cerr << "OpenCL acceleration failed, using CPU: " << e.what() << std::endl;
            // Fall through to CPU implementation
        }
    }
    
    const int block = CACHE_BLOCK_SIZE;
    
    // Check if we should use threaded implementation
    if ((a.col * b.col * a.row) >= THREAD_SIZE_THRESHOLD * THREAD_SIZE_THRESHOLD) {
        // Multi-threaded implementation
        const int num_threads = std::min(
            static_cast<int>(std::thread::hardware_concurrency()),
            (a.col + block - 1) / block
        );
        
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        auto multiply_block = [&](int start_i, int end_i) {
            // Cache blocking for better memory access patterns
            for (int i = start_i; i < end_i; i += block) {
                for (int j = 0; j < b.col; j += block) {
                    for (int k = 0; k < a.row; k += block) {
                        // Process each block
                        for (int ii = i; ii < std::min(i + block, end_i); ++ii) {
                            for (int jj = j; jj < std::min(j + block, b.col); ++jj) {
                                T sum = 0;
                                // The key difference: access a(k, i) instead of a(i, k)
                                #pragma omp simd reduction(+:sum)
                                for (int kk = k; kk < std::min(k + block, a.row); ++kk) {
                                    sum += a.data[kk * a.col + ii] * b.data[kk * b.col + jj];
                                }
                                result.data[ii * b.col + jj] += sum;
                            }
                        }
                    }
                }
            }
        };
        
        // Distribute work among threads
        const int cols_per_thread = (a.col + num_threads - 1) / num_threads;
        
        for (int t = 0; t < num_threads; ++t) {
            const int start_col = t * cols_per_thread;
            const int end_col = std::min(start_col + cols_per_thread, a.col);
            if (start_col < end_col) {
                threads.emplace_back(multiply_block, start_col, end_col);
            }
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Single-threaded implementation with cache blocking
        for (int i = 0; i < a.col; i += block) {
            for (int j = 0; j < b.col; j += block) {
                for (int k = 0; k < a.row; k += block) {
                    // Process each block
                    for (int ii = i; ii < std::min(i + block, a.col); ++ii) {
                        for (int jj = j; jj < std::min(j + block, b.col); ++jj) {
                            T sum = 0;
                            // The key difference: access a(k, i) instead of a(i, k)
                            #pragma omp simd reduction(+:sum)
                            for (int kk = k; kk < std::min(k + block, a.row); ++kk) {
                                sum += a.data[kk * a.col + ii] * b.data[kk * b.col + jj];
                            }
                            result.data[ii * b.col + jj] += sum;
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

// Implementation of transpose_multiply_vec
template <typename T>
vec<T> transpose_multiply_vec(const matrix<T>& a, const vec<T>& v) {
    // Check dimensions for compatibility
    if (a.row != v.size) {
        std::cout << "Error: Incompatible dimensions for transpose_multiply_vec. "
                  << "Matrix A: " << a.row << "x" << a.col 
                  << ", Vector v: " << v.size << std::endl;
        exit(0);
    }
    
    // Result will have dimensions a.col
    vec<T> result(a.col);
    
    // Use hardware acceleration if available for float type
    if constexpr (std::is_same_v<T, float>) {
        try {
            if (OpenCLAccelerator::isOpenCLAvailable() && 
                OpenCLAccelerator::shouldUseGPU(a.col * a.row)) {
                bool success = OpenCLAccelerator::transposeMultiplyMatrixVector(
                    a.data, v.data, result.data, a.row, a.col
                );
                
                if (success) {
                    return result;
                }
                // Fall through to CPU implementation if GPU failed
            }
        } catch (const std::exception& e) {
            std::cerr << "OpenCL acceleration failed, using CPU: " << e.what() << std::endl;
            // Fall through to CPU implementation
        }
    }
    
    // Check if we should use threaded implementation
    if (a.col * a.row >= THREAD_SIZE_THRESHOLD) {
        // Multi-threaded implementation
        const int num_threads = std::min(
            static_cast<int>(std::thread::hardware_concurrency()),
            (a.col + 63) / 64  // Use chunks of reasonable size
        );
        
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        
        auto process_columns = [&](int start_col, int end_col) {
            for (int j = start_col; j < end_col; ++j) {
                T sum = 0;
                // For each element in column j, multiply by corresponding element in vector v
                #pragma omp simd reduction(+:sum)
                for (int i = 0; i < a.row; ++i) {
                    // The key part - accessing a(i,j) which is the (j,i) element of the transpose
                    sum += a.data[i * a.col + j] * v.data[i];
                }
                result.data[j] = sum;
            }
        };
        
        // Distribute work among threads
        const int cols_per_thread = (a.col + num_threads - 1) / num_threads;
        
        for (int t = 0; t < num_threads; ++t) {
            const int start_col = t * cols_per_thread;
            const int end_col = std::min(start_col + cols_per_thread, a.col);
            if (start_col < end_col) {
                threads.emplace_back(process_columns, start_col, end_col);
            }
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
    } else {
        // Single-threaded optimized implementation
        // For best cache performance, process one output element at a time
        #pragma omp simd
        for (int j = 0; j < a.col; ++j) {
            T sum = 0;
            // For each element in column j
            for (int i = 0; i < a.row; ++i) {
                // The key part - accessing a(i,j) which is the (j,i) element of the transpose
                sum += a.data[i * a.col + j] * v.data[i];
            }
            result.data[j] = sum;
        }
    }
    
    return result;
}
