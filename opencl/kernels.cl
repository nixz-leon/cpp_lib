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

// Vector addition - double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void vec_add_double(__global const double* a,
                            __global const double* b,
                            __global double* result,
                            const int length) {
    const size_t id = get_global_id(0);
    if (id < length) {
        result[id] = a[id] + b[id];
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

// Vector multiplication - double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void vec_mult_double(__global const double* a,
                             __global const double* b,
                             __global double* result,
                             const int length) {
    const size_t id = get_global_id(0);
    if (id < length) {
        result[id] = a[id] * b[id];
    }
}

// Dot product - float
__kernel void vec_dot_float(__global const float* a,
                           __global const float* b,
                           __global float* result,
                           const int length,
                           __local float* temp) {
    const size_t id = get_global_id(0);
    const size_t lid = get_local_id(0);
    const size_t local_size = get_local_size(0);
    
    // Initialize local memory
    temp[lid] = 0.0f;
    
    // Compute partial dot product
    if (id < length) {
        temp[lid] = a[id] * b[id];
    }
    
    // Wait for all threads
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            temp[lid] += temp[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // First thread writes result to global memory
    if (lid == 0) {
        // For OpenCL 1.2, we need to just use a manual approach
        // since atomic operations for floats aren't directly supported
        *result += temp[0];
    }
}

// Dot product - double
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void vec_dot_double(__global const double* a,
                            __global const double* b,
                            __global double* result,
                            const int length,
                            __local double* temp) {
    const size_t id = get_global_id(0);
    const size_t lid = get_local_id(0);
    const size_t local_size = get_local_size(0);
    
    // Initialize local memory
    temp[lid] = 0.0;
    
    // Compute partial dot product
    if (id < length) {
        temp[lid] = a[id] * b[id];
    }
    
    // Wait for all threads
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduction in local memory
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            temp[lid] += temp[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // First thread writes result to global memory
    if (lid == 0) {
        // For OpenCL 1.2, we need to just use a manual approach
        // since atomic operations for doubles aren't directly supported
        *result += temp[0];
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

// Matrix multiplication - double (using local memory for tiling)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void matmul_double(__global const double* a,
                           __global const double* b,
                           __global double* result,
                           const int a_rows,
                           const int a_cols,
                           const int b_cols,
                           __local double* tileA,
                           __local double* tileB) {
    // Get global IDs
    const int row = get_global_id(1);
    const int col = get_global_id(0);
    
    // Get local IDs
    const int localRow = get_local_id(1);
    const int localCol = get_local_id(0);
    
    // Get local size
    const int TILE_SIZE = get_local_size(0); // Assuming square local workgroups
    
    double sum = 0.0;
    
    // Loop over tiles
    const int numTiles = (a_cols + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile from matrix A into local memory
        const int aRow = row;
        const int aCol = t * TILE_SIZE + localCol;
        
        if (aRow < a_rows && aCol < a_cols) {
            tileA[localRow * TILE_SIZE + localCol] = a[aRow * a_cols + aCol];
        } else {
            tileA[localRow * TILE_SIZE + localCol] = 0.0;
        }
        
        // Load tile from matrix B into local memory
        const int bRow = t * TILE_SIZE + localRow;
        const int bCol = col;
        
        if (bRow < a_cols && bCol < b_cols) {
            tileB[localRow * TILE_SIZE + localCol] = b[bRow * b_cols + bCol];
        } else {
            tileB[localRow * TILE_SIZE + localCol] = 0.0;
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