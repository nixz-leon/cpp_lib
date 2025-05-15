# C++ Linear Algebra Classes Documentation

This document provides an overview of the functions available in the custom linear algebra library, along with the potential errors each function might throw.

## Table of Contents
- [vec Class](#vec-class)
- [matrix Class](#matrix-class)
- [vecs Class](#vecs-class)
- [matricies Class](#matricies-class)

## vec Class

### Constructors

```cpp
vec(); // Default constructor
vec(int n); // Allocates vector of size n with zero values
vec(T *data, int n); // Constructs from array
vec(int n, T val); // Fills vector with specified value
vec(std::initializer_list<T> list); // Initializes from list
vec(const vec<T> &other); // Copy constructor
vec(vec<T> &&other) noexcept; // Move constructor
```

### Assignment Operators

```cpp
vec<T>& operator=(vec<T>& other); // Copy assignment
vec<T>& operator=(const vec<T>& other); // Const copy assignment
vec<T>& operator=(vec<T>&& other) noexcept; // Move assignment
vec<T>& operator=(std::initializer_list<T> list); // List assignment
```

### Element Access

```cpp
T& operator()(int i); // Access element
const T& operator()(int i) const; // Const element access
```

**Potential Errors:**
- "Error: Vector is empty" - If the vector's data pointer is null
- "Error: Tried to access element [i] in vector of size [size]" - If index is out of bounds

### Arithmetic Operations

```cpp
friend vec<T> operator+(const vec<T>& a, const vec<T>& b);
friend vec<T> operator-(const vec<T>& a, const vec<T>& b);
friend vec<T> operator*(const vec<T>& a, T b); // Scalar multiplication
friend vec<T> operator*(T a, const vec<T>& b); // Scalar multiplication
friend T operator*(const vec<T>& a, const vec<T>& b); // Dot product
friend vec<T> element_mult(const vec<T>& a, const vec<T>& b); // Element-wise multiplication
vec<T> operator+=(const vec<T>& other);
vec<T> operator-=(const vec<T>& other);
friend vec<T> operator-(vec<T> a, T c); // Subtract scalar
```

**Potential Errors:**
- "Error: vector is empty" - If either vector's data is null
- "Error: Tried to add/subtract/multiply a vector of size [a.size] with a vector of size [b.size]" - If vector dimensions don't match

### Statistical Operations

```cpp
T max(); // Maximum value
T mean(); // Average of elements
T min(); // Minimum value
T sum(); // Sum of elements
```

### Utility Functions

```cpp
void printout(); // Print vector contents
```

### OpenCL Acceleration
The vec class uses OpenCLAccelerator for hardware acceleration of operations when available, with fallback to multi-threaded or sequential CPU implementation.

## matrix Class

### Constructors

```cpp
matrix(); // Default constructor
matrix(int n, int m); // n rows, m columns
matrix(vecs<T> &vectors, bool row_major = false); // From vectors
matrix(vec<T> &vector, bool col_matrix = false); // From vector
matrix(const matrix<T> &other); // Copy constructor
matrix(matrix<T> &&other) noexcept; // Move constructor
```

### Assignment Operators

```cpp
matrix<T>& operator=(matrix<T>& other); // Copy assignment
matrix<T>& operator=(const matrix<T>& other); // Const copy assignment
matrix<T>& operator=(matrix<T>&& other) noexcept; // Move assignment
```

### Element Access

```cpp
T& operator()(int row, int col); // Access element
const T& operator()(int row, int col) const; // Const element access
vec<T> get_row(int r); // Get row as vector
vec<T> set_row(int r, const vec<T> &v); // Set row from vector
```

**Potential Errors:**
- "no data to be accessed" - If matrix data is null
- "empty matrix" - If matrix dimensions are zero
- "Tried to access matrix elm [r],[c], But size is [row],[col]" - If indices are out of bounds
- "tried to access row [r] which is negative" - If row index is negative
- "tried to access row [c] which is negative" - If column index is negative
- "Vector size [v.size] does not match matrix column size [col]" - In set_row if vector size doesn't match

### Matrix Operations

```cpp
void set_diag(T c); // Set diagonal elements
void row_swap(int a, int b); // Swap two rows
void row_add(int a, int b, T fac); // Add fac * row b to row a
matrix<T> operator/=(T c); // Divide by scalar
matrix<T> operator*=(T c); // Multiply by scalar
matrix<T>& operator+=(const matrix<T>& other); // Add in-place
matrix<T>& operator-=(const matrix<T>& other); // Subtract in-place
vec<T> operator*(vec<T> b); // Matrix-vector multiplication
vec<T> operator*(vec<T> b) const; // Const matrix-vector multiplication
void multiply(const matrix<T>& other, matrix<T>& result) const; // Matrix multiplication into result
```

**Potential Errors:**
- "Error: Tried to add-assign matrices of incompatible dimensions" - In += if dimensions mismatch
- "Error: One or both matrices in += have null data" - In += if data is null
- "Error: Tried to subtract-assign matrices of incompatible dimensions" - In -= if dimensions mismatch
- "Error: One or both matrices in -= have null data" - In -= if data is null
- "Error: matrix is empty" - In matrix*vector if matrix data is null
- "Error: Tried to multiply a matrix of size [row]x[col] with a vector of size [b.size]" - In matrix*vector if dimensions mismatch
- "Matrix dimensions don't match for multiplication" - In multiply() if dimensions aren't compatible

### Static Matrix Operations

```cpp
friend matrix<T> transpose(matrix<T> a); // Matrix transpose
friend matrix<T> operator+(matrix<T> a, matrix<T> b); // Matrix addition
friend matrix<T> operator-(matrix<T> a, matrix<T> b); // Matrix subtraction
friend matrix<T> operator*(matrix<T> a, matrix<T> b); // Matrix multiplication
friend matrix<T> operator*(matrix<T> a, T b); // Scalar multiplication
friend matrix<T> operator*(T b, matrix<T> a); // Scalar multiplication
friend matrix<T> outer_product(vec<T> a, vec<T> b); // Outer product
friend matrix<T> hadamard(matrix<T> a, matrix<T> b); // Element-wise multiplication
friend matrix<T> transpose_multiply(const matrix<T>& a, const matrix<T>& b); // A^T * B
friend vec<T> transpose_multiply_vec(const matrix<T>& a, const vec<T>& v); // A^T * v
```

**Potential Errors:**
- "Tried to add/subtract a [a.row],[a.col] matrix with a [b.row],[b.col] matrix" - If matrix dimensions don't match
- "Invalid matrix dimensions" - In matrix multiplication if a.col != b.row
- "Error: Incompatible dimensions for transpose_multiply" - If a.row != b.row
- "Error: Incompatible dimensions for transpose_multiply_vec" - If a.row != v.size

### Utility Functions

```cpp
void printout(); // Print matrix contents
T sum_elms(); // Sum of all elements
void add_per_row(vec<T> v); // Add vector to each row
void add_per_col(vec<T> v); // Add vector to each column
bool shouldUseThreads() const; // Check if threading should be used
```

**Potential Errors:**
- "vector not the right size" - In add_per_row/add_per_col if vector dimensions don't match

### OpenCL Acceleration
The matrix class uses OpenCLAccelerator for hardware acceleration of operations when available, with fallback to multi-threaded or sequential CPU implementation.

## vecs Class

### Constructors

```cpp
vecs(); // Default constructor
vecs(int num_vec, int size); // Normal constructor
vecs(const vecs<T>& other); // Copy constructor
vecs(vecs<T>&& other) noexcept; // Move constructor
```

### Assignment Operators

```cpp
vecs<T>& operator=(const vecs<T>& other); // Copy assignment
vecs<T>& operator=(vecs<T>&& other) noexcept; // Move assignment
```

### Element Access

```cpp
vec<T>& operator()(int index); // Access operator
const vec<T>& operator()(int index) const; // Const access operator
vec<T>& back(); // Access last vector
const vec<T>& back() const; // Const access last vector
```

**Potential Errors:**
- "Index out of range" exception - If index is out of bounds

### Information Methods

```cpp
int num_of_vecs(); // Number of vectors
int num_of_vecs() const; // Const number of vectors
int size(); // Size of each vector
int size() const; // Const size of each vector
```

### Operations

```cpp
vecs<T> subset(int start_col, int end_col); // Subset function
vecs<T> subset(int start_col, int end_col) const; // Const subset function
vec<T> vectorize(); // Flatten to single vector
vec<T> vectorize() const; // Const flatten to single vector
T sum() const; // Sum all elements
```

### Utility Functions

```cpp
void printout(); // Print contents
```

### Global Functions

```cpp
template <typename T>
std::pair<vecs<T>, vecs<T>> train_test_split(vecs<T>& data, double test_size = 0.2);
```

## matricies Class

### Constructors

```cpp
matricies(); // Default constructor
matricies(int num_matricies, int n, int m); // Normal constructor
matricies(const matricies<T>& other); // Copy constructor
matricies(matricies<T>&& other) noexcept; // Move constructor
```

### Assignment Operators

```cpp
matricies<T>& operator=(const matricies<T>& other); // Copy assignment
matricies<T>& operator=(matricies<T>&& other) noexcept; // Move assignment
```

### Element Access

```cpp
matrix<T>& operator()(int index); // Access operator
const matrix<T>& operator()(int index) const; // Const access operator
matrix<T> back(); // Access last matrix
```

**Potential Errors:**
- "Index out of range" exception - If index is out of bounds

### Information Methods

```cpp
int size(); // Number of matrices
int size_col(); // Number of columns in each matrix
int size_row(); // Number of rows in each matrix
int size() const; // Const number of matrices
int size_col() const; // Const number of columns in each matrix
int size_row() const; // Const number of rows in each matrix
```

### Operations

```cpp
matricies<T> subset(int start_index, int end_index); // Subset function
```

### Utility Functions

```cpp
void printout(); // Print contents
```

## Common Error Patterns

1. **Null or empty containers:**
   - "Error: vector/matrix is empty"
   - "no data to be accessed"

2. **Index out of bounds:**
   - "Tried to access element [i] in vector of size [size]"
   - "Tried to access matrix elm [r],[c], But size is [row],[col]"
   - "Index out of range" exception

3. **Dimension mismatches:**
   - "Tried to add/subtract/multiply a vector of size [a.size] with a vector of size [b.size]"
   - "Tried to add/subtract a [a.row],[a.col] matrix with a [b.row],[b.col] matrix"
   - "Invalid matrix dimensions"
   - "vector not the right size"

4. **Hardware acceleration failures:**
   - "OpenCL acceleration failed, using CPU: [error message]" 