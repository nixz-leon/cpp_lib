#pragma once
#ifndef vecs_hpp
#include "vecs.hpp"
#endif
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "HardwareAccelerator.hpp"

// Define constants for hardware acceleration
#define CACHE_BLOCK_SIZE HardwareAccelerator::getCacheBlockSize()
#define THREAD_SIZE_THRESHOLD HardwareAccelerator::getThreadThreshold()
#define GPU_SIZE_THRESHOLD HardwareAccelerator::getGPUThreshold()

template <typename T>
class matrix {  
    public:
        matrix() : row(0), col(0), data(nullptr) {} //default 
        matrix(int n, int m) : row(n), col(m) {data = new T[row * col]();}; //normal
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
            
            // Use hardware acceleration if available
            #ifdef HARDWARE_ACCELERATION_ENABLED
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    matrix<T> result(a.row, b.col);
                    bool success = HardwareAccelerator::multiplyMatrices(
                        a.data, a.row, a.col,
                        b.data, b.row, b.col,
                        result.data
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
            
            // Use multithreaded CPU implementation for larger matrices
            if (total_operations >= THREAD_SIZE_THRESHOLD * THREAD_SIZE_THRESHOLD) {
                return multiplyThreaded(a, b);
            }
            
            // Use sequential implementation for smaller matrices
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

        inline void multiply(const matrix<T>& other, matrix<T>& result) const {
            if (col != other.row || result.row != row || result.col != other.col) {
                throw std::invalid_argument("Matrix dimensions don't match for multiplication");
            }

            // Use hardware acceleration if available
            #ifdef HARDWARE_ACCELERATION_ENABLED
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    bool success = HardwareAccelerator::multiplyMatrices(
                        data, row, col,
                        other.data, other.row, other.col,
                        result.data
                    );
                    
                    if (success) {
                        return;
                    }
                }
            } catch (const std::exception& e) {
                // Fall back to CPU if hardware acceleration fails
                std::cerr << "Hardware acceleration failed, using CPU: " << e.what() << std::endl;
            }
            #endif

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
    #ifdef HARDWARE_ACCELERATION_ENABLED
    try {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            vec<T> result(this->row);
            bool success = HardwareAccelerator::multiplyMatrixVector(
                this->data, this->row, this->col,
                b.data, result.data
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
};

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
