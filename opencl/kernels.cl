// OpenCL Kernels for Linear Algebra Operations

// Vector addition - float
__kernel void vec_add_float(__global const float* a,
                           __global const float* b,
                           __global float* result,
                           const int length) {
    const size_t id = get_global_id(0);
    if (id < length) {
        result[id] = a[id] + b[id];
    }
}

// Vector subtraction - float
__kernel void vec_sub_float(__global const float* a,
                           __global const float* b,
                           __global float* result,
                           const int length) {
    const size_t id = get_global_id(0);
    if (id < length) {
        result[id] = a[id] - b[id];
    }
}

// Vector multiplication - float
__kernel void vec_mult_float(__global const float* a,
                            __global const float* b,
                            __global float* result,
                            const int length) {
    const size_t id = get_global_id(0);
    if (id < length) {
        result[id] = a[id] * b[id];
    }
}

// Vector scaling - float
__kernel void vec_scale_float(__global const float* a,
                             const float scalar,
                             __global float* result,
                             const int length) {
    const size_t id = get_global_id(0);
    if (id < length) {
        result[id] = a[id] * scalar;
    }
}

// Dot product - float
__kernel void vec_dot_float(__global const float* a,
                           __global const float* b,
                           __global float* partial_results,
                           const int length,
                           __local float* temp) {
    const size_t id = get_global_id(0);
    const size_t lid = get_local_id(0);
    const size_t local_size = get_local_size(0);
    const size_t group_id = get_group_id(0);
    
    // Initialize local memory for this work-item's contribution
    temp[lid] = 0.0f;
    
    // Compute partial dot product for elements this work-item is responsible for
    // This loop structure assumes global_size might be larger than length,
    // and each work-item processes one element if id < length.
    // A strided loop (for (int i = id; i < length; i += global_size)) can also be used
    // if global_size is smaller than length but still want to use all work-items.
    // Given the C++ side calculates global_size based on length, this simple check is fine.
    if (id < length) {
        temp[lid] = a[id] * b[id];
    }
    
    // Wait for all threads in the work-group to complete their individual calculations
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            temp[lid] += temp[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // First thread in each work-group writes its partial sum to global memory
    if (lid == 0) {
        partial_results[group_id] = temp[0];
    }
}

// Matrix multiplication - float (using local memory for tiling)
__kernel void matmul_float(__global const float* a,
                          __global const float* b,
                          __global float* result,
                          const int a_rows,
                          const int a_cols,
                          const int b_cols,
                          __local float* tileA,
                          __local float* tileB) {
    // Get global IDs
    const int row = get_global_id(1);
    const int col = get_global_id(0);
    
    // Get local IDs
    const int localRow = get_local_id(1);
    const int localCol = get_local_id(0);
    
    // Get local size
    const int TILE_SIZE = get_local_size(0); // Assuming square local workgroups
    
    float sum = 0.0f;
    
    // Loop over tiles
    const int numTiles = (a_cols + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile from matrix A into local memory
        const int aRow = row;
        const int aCol = t * TILE_SIZE + localCol;
        
        if (aRow < a_rows && aCol < a_cols) {
            tileA[localRow * TILE_SIZE + localCol] = a[aRow * a_cols + aCol];
        } else {
            tileA[localRow * TILE_SIZE + localCol] = 0.0f;
        }
        
        // Load tile from matrix B into local memory
        const int bRow = t * TILE_SIZE + localRow;
        const int bCol = col;
        
        if (bRow < a_cols && bCol < b_cols) {
            tileB[localRow * TILE_SIZE + localCol] = b[bRow * b_cols + bCol];
        } else {
            tileB[localRow * TILE_SIZE + localCol] = 0.0f;
        }
        
        // Synchronize to make sure the tiles are loaded
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow * TILE_SIZE + k] * tileB[k * TILE_SIZE + localCol];
        }
        
        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result to global memory
    if (row < a_rows && col < b_cols) {
        result[row * b_cols + col] = sum;
    }
}

// Matrix transpose
__kernel void matrix_transpose(__global const float* input,
                              __global float* output,
                              const int rows,
                              const int cols) {
    const size_t row = get_global_id(0);
    const size_t col = get_global_id(1);
    
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

// Outer product - float version
__kernel void outer_prod_float(__global const float* a,
                             __global const float* b,
                             __global float* result,
                             const int a_size,
                             const int b_size) {
    const size_t i = get_global_id(0);
    
    if (i < a_size) {
        const float a_val = a[i];
        
        // Each work item computes one row of the result matrix
        for (int j = 0; j < b_size; j++) {
            result[i * b_size + j] = a_val * b[j];
        }
    }
}

// Kernel for extracting and processing image patches for prep_vec - float version
__kernel void prep_vec_kernel_float(
    __global const float* image_vec,      // Input image vector
    __global float* result,              // Output matrix (flattened)
    const int image_width,               // Width of input image
    const int image_height,              // Height of input image
    const int filter_size,               // Size of filter/patch
    const int stride,                    // Stride value
    const int padding,                   // Padding value
    const int num_submatrices_x,         // Number of submatrices in x direction
    const int num_submatrices_y          // Number of submatrices in y direction
) {
    // Calculate global position based on submatrix index
    const int submatrix_idx = get_global_id(0); // Represents which submatrix we're processing
    
    // Check bounds
    if (submatrix_idx >= num_submatrices_x * num_submatrices_y) {
        return;
    }
    
    // Calculate the starting position for this submatrix in the original image
    const int submatrix_x = (submatrix_idx / num_submatrices_y) * stride - padding;
    const int submatrix_y = (submatrix_idx % num_submatrices_y) * stride - padding;
    
    // Process this submatrix
    for (int i = 0; i < filter_size; i++) {
        for (int j = 0; j < filter_size; j++) {
            // Coordinates in original image
            const int img_x = submatrix_x + i;
            const int img_y = submatrix_y + j;
            
            // Value to store
            float value = 0.0f;
            
            // Check if within image bounds
            if (img_x >= 0 && img_x < image_width && 
                img_y >= 0 && img_y < image_height) {
                value = image_vec[img_x * image_height + img_y];
            }
            
            // Store in result at the right position
            // Each submatrix takes filter_size * filter_size floats
            const int result_idx = submatrix_idx * (filter_size * filter_size) + i * filter_size + j;
            result[result_idx] = value;
        }
    }
} 