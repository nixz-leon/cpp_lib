#include "stats_method.hpp"
#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <cstdarg>
#include <iomanip>
#include <cmath>  // For isnan() and isinf()
#include <random>
#include <memory>
#include <unordered_map>




class act_func_vec{
    private:
        static vec<float> relu(vec<float> &v){
            vec<float> result(v.size);
            for(int i = 0; i < v.size; i++){
                result(i) = v(i) > 0 ? v(i) : 0;
            }
            return result;
        }
        static vec<float> relu_deriv(vec<float> &v){
            vec<float> result(v.size);
            for(int i = 0; i < v.size; i++){
                result(i) = v(i) > 0 ? 1 : 0;
            }
            return result;
        }
        static vec<float> leaky_relu(vec<float> &v){
            vec<float> result(v.size);
            for(int i = 0; i < v.size; i++){
                result(i) = v(i) > 0 ? v(i) : 0.01f*v(i);
            }
            return result;
        }
        static vec<float> leaky_relu_deriv(vec<float> &v){
            vec<float> result(v.size);
            for(int i = 0; i < v.size; i++){
                result(i) = v(i) > 0 ? 1 : 0.01f;
            }
            return result;
        }
        static vec<float> sigmoid(vec<float> &v){
            vec<float> result(v.size);
            for(int i = 0; i < v.size; i++){
                result(i) = 1/(1+exp(-v(i)));
            }
            return result;
        }
        static vec<float> sigmoid_deriv(vec<float> &v){
            vec<float> result(v.size, 1.0f);
            vec<float> sigmoid_result = sigmoid(v);
            result = element_mult(sigmoid_result, result-sigmoid_result);
            return result;
        }
        static vec<float> softmax(vec<float> &v){
            vec<float> result(v.size);
            float max_val = v(0);
            
            // Find max value for numerical stability
            for(int i = 1; i < v.size; i++){
                if(v(i) > max_val) max_val = v(i);
            }
            
            float sum = 0.0f;
            // Compute exponentials with subtracted max value
            for(int i = 0; i < v.size; i++){
                result(i) = exp(v(i) - max_val);
                sum += result(i);
            }
            
            // Normalize
            float inv_sum = 1.0f / sum;
            for(int i = 0; i < v.size; i++){
                result(i) *= inv_sum;
            }
            
            return result;
        }
        
        // Calculate softmax and its derivative together for efficiency
        static std::pair<vec<float>, vec<float>> softmax_and_deriv(vec<float> &v){
            vec<float> softmax_result = softmax(v);
            vec<float> deriv_result(v.size, 1.0f);
            deriv_result = element_mult(softmax_result, deriv_result-softmax_result);
            return {softmax_result, deriv_result};
        }
        
        static vec<float> softmax_deriv(vec<float> &v){
            // Instead of recalculating softmax, we could memoize the result
            // but for now, just calculate the derivative directly
            vec<float> softmax_result = softmax(v);
            vec<float> result(v.size, 1.0f);
            result = element_mult(softmax_result, result-softmax_result);
            return result;
        }
        static vec<float> linear(vec<float> &v){
            return v;
        }
        static vec<float> linear_deriv(vec<float> &v){
            return vec<float>(v.size, 1.0f);
        }
        vec<float> (*act_funcs[5])(vec<float>&)={relu, leaky_relu, sigmoid, softmax, linear};
        vec<float> (*act_derivs[5])(vec<float>&)={relu_deriv, leaky_relu_deriv, sigmoid_deriv, softmax_deriv, linear_deriv};
        
        // Cache for storing previously computed activations to avoid recalculation
        mutable std::unordered_map<int, vec<float>> activation_cache;
        
    public:
        act_func_vec(){
            // Clear cache on construction
            activation_cache.clear();
        }
        
        vec<float> operator()(vec<float> &v, int call){
            return act_funcs[call](v);
        }
        
        vec<float> deriv(vec<float> &v, int call){
            return act_derivs[call](v);
        }
        
        vecs<float> operator()(vecs<float> &v, int call){
            // Process in batches for better performance
            #pragma omp parallel for
            for(int i = 0; i < v.num_of_vecs(); i++){
                v(i) = act_funcs[call](v(i));
            }
            return v;
        }
        
        vecs<float> deriv(vecs<float> &v, int call){
            // Process in batches for better performance
            #pragma omp parallel for
            for(int i = 0; i < v.num_of_vecs(); i++){
                v(i) = act_derivs[call](v(i));
            }
            return v;
        }
};

class act_func_mat{
    private:
        static matrix<float> relu(matrix<float> &m){
            matrix<float> result(m.row, m.col);
            for(int i = 0; i < m.row; i++){
                for(int j = 0; j < m.col; j++){
                    result(i, j) = m(i, j) > 0 ? m(i, j) : 0;
                }
            }
            return result;
        }
        static matrix<float> relu_deriv(matrix<float> &m){
            matrix<float> result(m.row, m.col);
            for(int i = 0; i < m.row; i++){
                for(int j = 0; j < m.col; j++){
                    result(i, j) = m(i, j) > 0 ? 1 : 0;
                }
            }
            return result;
        }
        static matrix<float> leaky_relu(matrix<float> &m){
            matrix<float> result(m.row, m.col);
            for(int i = 0; i < m.row; i++){
                for(int j = 0; j < m.col; j++){
                    result(i, j) = m(i, j) > 0 ? m(i, j) : 0.01f*m(i, j);
                }
            }
            return result;
        }
        static matrix<float> leaky_relu_deriv(matrix<float> &m){
            matrix<float> result(m.row, m.col);
            for(int i = 0; i < m.row; i++){
                for(int j = 0; j < m.col; j++){
                    result(i, j) = m(i, j) > 0 ? 1 : 0.01f;
                }
            }
            return result;
        }
        
        matrix<float> (*act_funcs[2])(matrix<float>&)={relu, leaky_relu};
        matrix<float> (*act_derivs[2])(matrix<float>&)={relu_deriv, leaky_relu_deriv};
    public:
        matrix<float> operator()(matrix<float> &m, int call){
            return act_funcs[call](m);
        }
        matrix<float> deriv(matrix<float> &m, int call){
            return act_derivs[call](m);
        }   
};

class layer{
    public:
    virtual ~layer() noexcept = default;
    virtual vecs<float> forward(const vecs<float>& input) = 0;
    virtual vecs<float> calc_gradient(const vecs<float>& prev_delta, const matricies<float>& prev_weight, std::string prev_layer_type) = 0;
    virtual vecs<float> calc_gradient_last(const vec<float>& actual, std::string loss_func) = 0;
    virtual void update_params(float learning_rate) = 0;
    virtual matricies<float> get_weights() = 0;
    virtual int get_output_size() const = 0;
    virtual std::string get_layer_type() const = 0;
};

void init_mat(matrix<float> &mat){
    std::random_device rd;
    std::mt19937 gen(rd());
    // Use He initialization for better training with ReLU/LeakyReLU
    float scale = sqrt(2.0f / mat.row);
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (int i = 0; i < mat.row; i++) {
        for (int j = 0; j < mat.col; j++) {
            mat(i, j) = dist(gen);
        }
    }
}

void init_vec(vec<float> &vec){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (int i = 0; i < vec.size; i++) {
        // Initialize biases with small positive values to help with activation
        vec(i) = 0.01f + dist(gen) * 0.1f;
    }
}


matrix<float> prep_vec(const vec<float>& image_vec, int image_width, int image_height, int filter_size, int stride, int padding) {
    // Calculate dimensions
    int num_submatrices_x = (image_width + 2 * padding - filter_size) / stride + 1;
    int num_submatrices_y = (image_height + 2 * padding - filter_size) / stride + 1;
    int total_submatrices = num_submatrices_x * num_submatrices_y;
    
    // Validate input parameters
    if (image_vec.size < image_width * image_height) {
        std::cerr << "Error: Image vector size (" << image_vec.size 
                  << ") smaller than expected (" << image_width * image_height << ")" << std::endl;
        // Return empty matrix or handle error
        return matrix<float>(1, 1);
    }
    
    if (num_submatrices_x <= 0 || num_submatrices_y <= 0 || total_submatrices <= 0) {
        std::cerr << "Error: Invalid submatrix dimensions calculated" << std::endl;
        // Return empty matrix or handle error
        return matrix<float>(1, 1);
    }
    
    // Create the result matrix directly - each submatrix becomes a column
    // Rows = filter_size * filter_size, Cols = total_submatrices
    matrix<float> result(filter_size * filter_size, total_submatrices);
    
    try {
        int start_index_x = 0 - padding;
        int start_index_y = 0 - padding;
        int submatrix_idx = 0;
        for (int i = start_index_x; i + filter_size <= image_width + padding; i += stride) {
            for (int j = start_index_y; j + filter_size <= image_height + padding; j += stride) {
                // Process this submatrix directly into the appropriate column
                for (int fi = 0; fi < filter_size; fi++) {
                    for (int fj = 0; fj < filter_size; fj++) {
                        // Calculate original image coordinates
                        int img_x = i + fi;
                        int img_y = j + fj;
                        
                        // Value to store
                        float value = 0.0f;
                        
                        // Check if within image bounds
                        if (img_x >= 0 && img_x < image_width && img_y >= 0 && img_y < image_height) {
                            int img_idx = img_x * image_height + img_y;
                            if (img_idx < image_vec.size) {
                                value = image_vec(img_idx);
                            }
                        }
                        
                        // Store directly in the matrix at row (fi*filter_size+fj), column (submatrix_idx)
                        if (submatrix_idx < total_submatrices && (fi * filter_size + fj) < result.row) {
                            result(fi * filter_size + fj, submatrix_idx) = value;
                        }
                    }
                }
                
                submatrix_idx++;
                if (submatrix_idx >= total_submatrices) {
                    break;  // Safety check
                }
            }
            if (submatrix_idx >= total_submatrices) {
                break;  // Safety check
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in prep_vec: " << e.what() << std::endl;
    }
    
    return result;
};





class conv_layer : public layer{
    private:
    int input_size_x;
    int input_size_y;
    int output_size_x;
    int output_size_y;
    int num_of_inputs;
    int filter_size;
    int num_of_filters;
    int stride;
    int padding;
    matrix<float> weights;
    vec<float> bias;
    vecs<float> input;
    matricies<float> input_matricies;
    vecs<float> output;
    vecs<float> preactivation;
    matricies<float> preactivation_matrix;
    vecs<float> delta;
    int call;
    act_func_vec act_func;
    int num_outputs;
    public:
    conv_layer(int input_size_x, int input_size_y, int output_size_x, int output_size_y, int num_of_inputs, int num_of_filters, int filter_size, int stride, int padding, std::string activation_func){
        this->input_size_x = input_size_x;
        this->input_size_y = input_size_y;
        this->output_size_x = output_size_x;
        this->output_size_y = output_size_y;
        this->num_of_inputs = num_of_inputs;
        this->num_of_filters = num_of_filters;
        this->filter_size = filter_size;
        this->stride = stride;
        this->padding = padding;
        this->num_outputs = num_of_filters * num_of_inputs;

        // example weight matrix: 2 2x2 filters:
        // f11 f12 f13 f14
        // f21 f22 f23 f24
        // this is equalivant to the filters:
        // f11 f12
        // f13 f13

        // and: 
        // f21 f22
        // f23 f24

        weights = matrix<float>(num_of_filters, filter_size*filter_size);
        init_mat(weights);

        bias = vec<float>(num_of_filters);
        init_vec(bias);

        

        int num_submatrices_x = (input_size_x + 2 * padding - filter_size) / stride + 1;
        int num_submatrices_y = (input_size_y + 2 * padding - filter_size) / stride + 1;
        int total_submatrices = num_submatrices_x * num_submatrices_y;
        input_matricies = matricies<float>(num_of_filters,filter_size*filter_size, total_submatrices);
        preactivation_matrix = matricies<float>(num_of_filters*num_of_inputs, num_of_filters ,weights.col);
        preactivation = vecs<float>(num_outputs, total_submatrices);
        output = vecs<float>(num_outputs, total_submatrices);
        delta = vecs<float>(num_outputs, total_submatrices);

        if(activation_func == "relu"){
            call = 0;
        }else if(activation_func == "leaky_relu"){
            call = 1;
        }else{
            std::cout << "activation function not supported: " << activation_func << std::endl;
        }

    }

    vecs<float> forward(const vecs<float>& in) override {
        // Create a local copy of the input
        vecs<float> local_input = in;
        
        // Process using CPU implementation
        for(int i = 0; i < local_input.num_of_vecs(); i++) {
            input_matricies(i) = prep_vec(local_input(i), input_size_x, input_size_y, filter_size, stride, padding);
            
            // Apply all filters to the current input
            preactivation_matrix(i) = weights * input_matricies(i);
            
            preactivation_matrix(i).add_per_col(bias);
            
            // Store results in the right order:
            // For input i, store filter j result at position (i*num_of_filters + j)
            for(int j = 0; j < num_of_filters; j++) {
                preactivation(i*num_of_filters + j) = preactivation_matrix(i).get_row(j);
            }
        }
        
        // Store the input for backpropagation
        input = in;
        
        // Apply activation function
        output = act_func(preactivation, call);
        
        return output;
    }

    //this function is cancer and needs to be optimized
    vecs<float> calc_gradient(const vecs<float>& prev_delta, const matricies<float>& prev_weight, std::string prev_layer_type) override {
        vecs<float> temp_delta(prev_delta.num_of_vecs(), prev_weight.size_row());
        
        if(prev_layer_type == "pool"){
            //delta is a vecs of size num_of_filters * num_of_inputs
            for(int i = 0; i < prev_delta.num_of_vecs(); i++){
                delta(i) = transpose_multiply_vec(prev_weight(i), prev_delta(i));
                delta(i) = element_mult(delta(i), act_func.deriv(preactivation(i), call));
                
                // Check for NaN values
                for(int j = 0; j < delta(i).size; j++) {
                    if(std::isnan(delta(i)(j))) {
                        delta(i)(j) = 0.0f; // Replace NaN with zero
                        std::cout << "NaN detected in gradient (pool layer) - replaced with 0" << std::endl;
                    }
                }
            }
        }
        else if(prev_layer_type == "conv"){
            int prev_num_of_filters = prev_weight.size();
            
            
            // Initialize delta to zero
            for(int i = 0; i < delta.num_of_vecs(); i++) {
                for(int j = 0; j < delta(i).size; j++) {
                    delta(i)(j) = 0.0f;
                }
            }
            
            // Process each filter weight matrix
            for(int i = 0; i < prev_num_of_filters; i++) {
        
                
                // For each delta
                for(int j = 0; j < num_outputs; j++) {
                    // Check matrix dimensions before multiplication
                    
                    // Calculate input gradient
                    temp_delta(j) = transpose_multiply_vec(prev_weight(i), prev_delta(j));
                    
                    // Apply activation derivative - use proper indexing
                    int input_idx = j / prev_num_of_filters;
                    if(input_idx < preactivation.num_of_vecs()) {
                        temp_delta(j) = element_mult(temp_delta(j), act_func.deriv(preactivation(input_idx), call));
                        
                        // Accumulate gradients
                        delta(input_idx) += temp_delta(j);
                    }
                }
            }
            
            // Check for NaN or infinity and apply gradient clipping
            float gradient_clip_threshold = 1.0f;
            for(int i = 0; i < delta.num_of_vecs(); i++) {
                for(int j = 0; j < delta(i).size; j++) {
                    if(std::isnan(delta(i)(j)) || std::isinf(delta(i)(j))) {
                        delta(i)(j) = 0.0f;
                        std::cout << "NaN/Inf detected in gradient (conv layer) - replaced with 0" << std::endl;
                    }
                    else if(delta(i)(j) > gradient_clip_threshold) {
                        delta(i)(j) = gradient_clip_threshold;
                    }
                    else if(delta(i)(j) < -gradient_clip_threshold) {
                        delta(i)(j) = -gradient_clip_threshold;
                    }
                }
            }
        }else{
            std::cout << "prev layer type not supported: " << prev_layer_type << std::endl;
        }        
        
        return delta;
    }
    
    void update_params(float learning_rate) override {
        // Initialize gradient matrix with same dimensions as weights
        matrix<float> grad(weights.col, weights.row);
        // For each input
        for(int i = 0; i < num_of_inputs; i++) {
            // For each filter
            for(int j = 0; j < num_of_filters; j++) {
                // Get the delta for this input-filter combination
                int delta_idx = i * num_of_filters + j;
                
                // Bounds check
                if(delta_idx >= delta.num_of_vecs()) {
                    std::cout << "Error: Delta index out of bounds in update_params" << std::endl;
                    continue;
                }
                
                vec<float> current_delta = delta(delta_idx);
                
                // Multiply delta with the corresponding input matrix
                matrix<float> delta_matrix(current_delta.size, num_of_filters);
                for(int k = 0; k < current_delta.size; k++) {
                    delta_matrix(k, j) = current_delta(k);
                }
                
                
                
                // Multiply input matrix with delta matrix and add to gradient
                matrix<float> temp_grad = input_matricies(i) * delta_matrix;
                grad = grad + temp_grad; 
            }
        }
        
        // Apply gradient clipping
        float clip_threshold = 1.0f;
        for(int i = 0; i < grad.row; i++) {
            for(int j = 0; j < grad.col; j++) {
                if(std::isnan(grad(i, j)) || std::isinf(grad(i, j))) {
                    grad(i, j) = 0.0f;
                    std::cout << "NaN/Inf detected in weight gradient - replaced with 0" << std::endl;
                }
                else if(grad(i, j) > clip_threshold) {
                    grad(i, j) = clip_threshold;
                }
                else if(grad(i, j) < -clip_threshold) {
                    grad(i, j) = -clip_threshold;
                }
            }
        }
        
        // Update weights
        weights = weights - learning_rate * transpose(grad);
        
        // Update biases with bounds checking and gradient clipping
        for(int i = 0; i < num_of_inputs; i++) {
            for(int j = 0; j < num_of_filters; j++) {
                int delta_idx = i * num_of_filters + j;
                
                // Bounds check
                if(delta_idx < delta.num_of_vecs()) {
                    float bias_grad = delta(delta_idx).sum();
                    
                    // Handle NaN/Inf and apply clipping
                    if(std::isnan(bias_grad) || std::isinf(bias_grad)) {
                        bias_grad = 0.0f;
                        std::cout << "NaN/Inf detected in bias gradient - replaced with 0" << std::endl;
                    }
                    else if(bias_grad > clip_threshold) {
                        bias_grad = clip_threshold;
                    }
                    else if(bias_grad < -clip_threshold) {
                        bias_grad = -clip_threshold;
                    }
                    
                    bias(j) -= learning_rate * bias_grad;
                }
            }
        }
    }

    vecs<float> calc_gradient_last(const vec<float>& actual, std::string loss_func) override {
        return vecs<float>();
    }

    int get_output_size() const override {
        return output_size_x * output_size_y;
    }
    
    matricies<float> get_weights() override {
        //need to generate a different weight matrix to pass back
        //need to go from:
        // w11 w12 w13 w14
        // w21 w22 w23 w24
        //with an input of x1 x2 x3 x4 x5 x6 x7 x8 x9
        //with the corresponding combo matrix
        // x1 x2 x4 x5
        // x2 x3 x5 x6
        // x4 x5 x7 x8
        // x5 x6 x8 x9
        // to w new 4x9 weight matrix
        // w11 w12 0   w13 w14 0   0   0   0
        // 0   w11 w12 0   w13 w14 0   0   0
        // 0   0   0   w11 w12 0   w13 w14 0
        // 0   0   0   0   w11 w12 0   w13 w14
        // and 
        // w21 w22 0   w23 w24 0   0   0   0
        // 0   w21 w22 0   w23 w24 0   0   0
        // 0   0   0   w21 w22 0   w23 w24 0
        // 0   0   0   0   w21 w22 0   w23 w24
        int num_submatrices_x = (input_size_x + 2 * padding - filter_size) / stride + 1;
        int num_submatrices_y = (input_size_y + 2 * padding - filter_size) / stride + 1;
        int total_submatrices = num_submatrices_x * num_submatrices_y;
        
        // Create result matrices - one for each filter
        matricies<float> result(num_of_filters, total_submatrices, input_size_x * input_size_y);
        
        // For each filter
        for (int f = 0; f < num_of_filters; f++) {
            // For each position in the output
            for (int i = 0; i < num_submatrices_x; i++) {
                for (int j = 0; j < num_submatrices_y; j++) {
                    int output_idx = i * num_submatrices_y + j;
                    
                    // For each position in the filter
                    for (int fi = 0; fi < filter_size; fi++) {
                        for (int fj = 0; fj < filter_size; fj++) {
                            // Calculate input position
                            int input_x = i * stride + fi - padding;
                            int input_y = j * stride + fj - padding;
                            
                            // Only set weight if input position is valid
                            if (input_x >= 0 && input_x < input_size_x && 
                                input_y >= 0 && input_y < input_size_y) {
                                int input_idx = input_x * input_size_y + input_y;
                                int filter_idx = fi * filter_size  + fj;
                                result(f)(output_idx, input_idx) = weights(f, filter_idx);
                            }
                        }
                    }
                }
            }
        }
        
        return result;
    }
    vec<float> get_bias() {
        return bias;
    }

    std::string get_layer_type() const override {
        return "conv";
    }
};


class pool_layer : public layer{
    public:
    int input_size_x;
    int input_size_y;
    int output_size_x;
    int output_size_y;
    int pool_size;
    int stride;
    vecs<float> input;
    matricies<float> input_matrix;
    vecs<float> output;
    vecs<float> delta;
    matricies<float> max_pool_weights;

    inline vec<float> max_pool(matrix<float> &data, int index){
        vec<float> result(data.col);
        // Create a weights matrix that transforms the original input vector to max pooled output
        // For a 3x3 input with 2x2 pooling and stride 1, we get 4 regions
        // The weights should be 4x9 (4 outputs x 9 input elements)
        
        
        // Find the maximum value in each column (each pooling region)
        for(int j = 0; j < data.col; j++) {
            float max_val = data(0, j);
            int max_idx = 0;
            
            // Check all values in this column (pooling region)
            for(int i = 1; i < data.row; i++) {
                if(data(i, j) > max_val) {
                    max_val = data(i, j);
                    max_idx = i;
                }
            }
            
            // Store the maximum value in the result
            result(j) = max_val;
            
            // Now we need to map the max_idx in the input_matrix back to its original position
            // in the input vector to place a 1 in the weights matrix
            // Each column j in data represents a pooling region
            // Calculate the starting point of this region in the  original input
            int region_row = (j / ((input_size_x - pool_size) / stride + 1)) * stride;
            int region_col = (j % ((input_size_x - pool_size) / stride + 1)) * stride;
            
            // Calculate which position in the pooling window had the max value
            int window_row = max_idx / pool_size;
            int window_col = max_idx % pool_size;
            
            // Calculate the original index in the input// Batch training method for multiple examples
            int orig_row = region_row + window_row;
            int orig_col = region_col + window_col;
            int orig_idx = orig_row * input_size_x + orig_col;
            
            // Place a 1 in the weight matrix where output j takes value from input orig_idx
            max_pool_weights(index)(j, orig_idx) = 1.0f;
        }
        
        // Store the weights matrix for backpropagation
        this->max_pool_weights = max_pool_weights;
        
        return result;
    }

    pool_layer(int input_size_x, int input_size_y, int output_size_x, int output_size_y, int pool_size, int stride, int num_inputs){
        this->input_size_x = input_size_x;
        this->input_size_y = input_size_y;
        this->output_size_x = output_size_x;
        this->output_size_y = output_size_y;
        this->pool_size = pool_size;
        this->stride = stride;

        int num_submatrices_x = (input_size_x - pool_size) / stride + 1;
        int num_submatrices_y = (input_size_y - pool_size) / stride + 1;
        int total_submatrices = num_submatrices_x * num_submatrices_y;
        input_matrix = matricies<float>(num_inputs, pool_size*pool_size, total_submatrices);
        //std::cout << "Input matrix: \n";
        //std::cout << "input matrix size: " << input_matrix.size() << std::endl;
        //std::cout << "input matrix size_col: " << input_matrix.size_col() << std::endl;
        //std::cout << "input matrix size_row: " << input_matrix.size_row() << std::endl;
        output = vecs<float>(num_inputs, total_submatrices);
        max_pool_weights = matricies<float>(num_inputs, total_submatrices, input_size_x * input_size_y);
    }

    vecs<float> forward(const vecs<float>& in) override {
        input = in;
        for(int i = 0; i < input.num_of_vecs(); i++){
            input_matrix(i) = prep_vec(input(i), input_size_x, input_size_y, pool_size, stride, 0); 
            output(i) = max_pool(input_matrix(i), i);
        }
        return output;
    }

    vecs<float> calc_gradient(const vecs<float>& prev_delta, const matricies<float>& prev_weight, std::string prev_layer_type) override {
        if(prev_layer_type == "dense"){
// Batch training method for multiple examples
            vec<float> temp = transpose_multiply_vec(prev_weight(0), prev_delta.vectorize());
            int num_deltas = input.num_of_vecs();
            int subdelta_size = max_pool_weights(0).row;  // Use column count of max_pool_weights
            delta = vecs<float>(num_deltas, subdelta_size);
            for(int i = 0; i < num_deltas; i++){
                for(int j = 0; j < subdelta_size; j++){
                    delta(i)(j) = (i*subdelta_size + j < temp.size) ? temp(i*subdelta_size + j) : 0.0f;
                }
            }
        }
        /*
        else if(prev_layer_type == "conv"){
            
        }else{//pool

        }*/ //only going to support  dense layers for now

        return delta;
    }

    vecs<float> calc_gradient_last(const vec<float>& actual, std::string loss_func) override {
        // Pooling layers don't have trainable parameters, so this is a no-op
        return vecs<float>();
    }
    
    void update_params(float learning_rate) override {// Batch training method for multiple examples
        // Pooling layers don't have trainable parameters, so this is a no-op
    }
    
    matricies<float> get_weights() override {
        return max_pool_weights;
    }
    

    int get_output_size() const override {
        return output_size_x * output_size_y * input.num_of_vecs();
    }

    std::string get_layer_type() const override {
        return "pool";
    }
};


class dense_layer : public layer{
    public:
    int input_size;
    int output_size;
    matrix<float> weights;
    vec<float> bias;
    vec<float> input;
    vec<float> output;
    vec<float> preactivation;
    vec<float> delta;
    int call;
    act_func_vec act_func;
    
    virtual ~dense_layer() noexcept = default;
    
    dense_layer(int input_size, int output_size, std::string activation_func){
        this->input_size = input_size;
        this->output_size = output_size;
        weights = matrix<float>(output_size, input_size);
        bias = vec<float>(output_size);
        input = vec<float>(input_size);
        output = vec<float>(output_size);
        preactivation = vec<float>(output_size);
        delta = vec<float>(output_size);
        if(activation_func == "relu"){
            call = 0;
        }
        else if(activation_func == "leaky_relu"){
            call = 1;
        }
        else if(activation_func == "sigmoid"){
            call = 2;
        }
        else if(activation_func == "softmax"){
            call = 3;
        }
        else if(activation_func == "linear"){
            call = 4;
        }
        else{
            throw std::invalid_argument("Invalid activation function");
        }
        init_mat(weights);
        init_vec(bias);
    }

    vecs<float> forward(const vecs<float>& in) override {
        
        input = in.vectorize();  // Store input for backprop
        
        // Matrix-vector multiplication for forward pass
        preactivation = weights*input + bias;
        
        // Apply activation function
        vec<float> preact_copy = preactivation; // Copy to avoid modifying preactivation
        output = act_func(preact_copy, call);
        vecs<float> output_vecs(1, output.size);// Batch training method for multiple examples
        output_vecs(0) = output;
        return output_vecs;
    }
    
    vecs<float> calc_gradient(const vecs<float>& prev_delta, const matricies<float>& prev_weight, std::string prev_layer_type) override {
        if(prev_layer_type == "dense"){
            delta = prev_delta.vectorize();
        }
        delta = transpose_multiply_vec(prev_weight(0), delta);
        delta = element_mult(delta, act_func.deriv(preactivation, call));
        vecs<float> delta_vecs(1, delta.size);
        delta_vecs(0) = delta;
        return delta_vecs;
    }
    
    vecs<float> calc_gradient_last(const vec<float>& actual, std::string loss_func) override {
        delta = output - actual;
        if(!(loss_func == "cross_entropy")){
            delta = element_mult(delta, act_func.deriv(preactivation, call));
        }
        vecs<float> delta_vecs(1, delta.size);
        delta_vecs(0) = delta;
        return delta_vecs;
    }
    
    void update_params(float learning_rate) override {
        // Compute weight updates using gradient descent
        matrix<float> weight_update = outer_product(delta, input);        
        weights = weights - learning_rate * weight_update;
        bias = bias - learning_rate * delta;
    }
    
    int get_output_size() const override {
        return output_size;
    }
    
    matricies<float> get_weights() override {
        matricies<float> result(1, weights.col, weights.row);
        result(0) = weights;
        return result;
    }

    
    std::string get_layer_type() const override {
        return "dense";
    }
};


class NeuralNetwork{
    private:
    std::vector<std::unique_ptr<layer>> layers;
    std::string loss_func;
    float learning_rate;
    int epochs;
    public:

    NeuralNetwork(std::string loss_func, float learning_rate, int epochs): loss_func(loss_func), learning_rate(learning_rate), epochs(epochs){
    }

    void add_layer(std::string layer_type, std::string activation_func, int input_size_x, int output_size_x, int input_size_y = 0, int output_size_y = 0, int input_size_z = 0, int output_size_z = 0){
        if(layer_type == "dense"){
            layers.push_back(std::make_unique<dense_layer>(input_size_x, output_size_x, activation_func));
        }
    }
    void add_pool_layer(int input_size_x, int input_size_y, int output_size_x, int output_size_y, int pool_size, int stride, int num_inputs){
        layers.push_back(std::make_unique<pool_layer>(input_size_x, input_size_y, output_size_x, output_size_y, pool_size, stride, num_inputs));
    }
    void add_conv_layer(int input_size_x, int input_size_y, int output_size_x, int output_size_y, int num_of_inputs, int num_of_filters, int filter_size, int stride, int padding, std::string activation_func){
        layers.push_back(std::make_unique<conv_layer>(input_size_x, input_size_y, output_size_x, output_size_y, num_of_inputs, num_of_filters, filter_size, stride, padding, activation_func));
    }
    
    vec<float> forward(const vec<float>& input){
        vecs<float> current_input(1, input.size);
        current_input(0) = input;
        for(size_t i = 0; i < layers.size(); ++i){
            current_input = layers[i]->forward(current_input);
        }
        return current_input.vectorize();
    }
    
    void backpropagation(const vec<float>& input, const vec<float>& actual, std::string loss_func, float learning_rate){
        // Forward pass to ensure all layers have calculated their outputs
        
        forward(input);
        
        // If no layers, do nothing
        if (layers.empty()) return;
        
        // Calculate gradient for the last layer
        
        vecs<float> delta = layers.back()->calc_gradient_last(actual, loss_func);
        std::string prev_layer_type = layers.back()->get_layer_type();

        // Backpropagate through the remaining layers
        
        for (int i = layers.size() - 2;i >= 0; i--) {
            // Get weights from the next layer using the new get_weights method
            matricies<float> next_layer_weights = layers[i + 1]->get_weights();
            
            // Backpropagate delta through previous layers
            delta = layers[i]->calc_gradient(delta, next_layer_weights, prev_layer_type);

            prev_layer_type = layers[i]->get_layer_type();
            
        }
        
        // Update parameters for all layers
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->update_params(learning_rate);
        }
    }
    
    // Simplified training method for single examples
    void train(const vec<float>& input, const vec<float>& target) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            std::cout << "Epoch " << epoch+1 << " of " << epochs << std::endl;
            backpropagation(input, target, loss_func, learning_rate);
        }
    }
    
    
    void train(const vecs<float>& inputs, const vecs<float>& targets) {
        int num_samples = inputs.num_of_vecs();
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float total_error = 0.0f;
            
            // Train on each example in the batch
            for (int i = 0; i < num_samples; i++) {
                vec<float> input = inputs(i);
                vec<float> target = targets(i);
                
                // Forward pass to calculate current error
                vec<float> prediction = forward(input);
                
                // Calculate error based on loss function
                float sample_error = 0.0f;
                if (loss_func == "cross_entropy") {
                    // Binary cross-entropy loss
                    for (int j = 0; j < prediction.size; j++) {
                        // Clip values to avoid log(0)
                        float pred_j = std::max(std::min(prediction(j), 0.9999f), 0.0001f);
                        sample_error -= (target(j) * log(pred_j) + (1.0f - target(j)) * log(1.0f - pred_j));
                    }
                } else {
                    // Mean squared error
                    vec<float> error_vec = target - prediction;
                    for (int j = 0; j < error_vec.size; j++) {
                        sample_error += error_vec(j) * error_vec(j);  // Sum of squared errors
                    }
                }
                total_error += sample_error;
                
                // Actual backpropagation
                backpropagation(input, target, loss_func, learning_rate);
                
                if(i+1 % 100 == 0){
                    std::cout << "Sample " << i+1 << " of " << num_samples << " Error: " << sample_error << std::endl;
                }
                
            }
            
            // Calculate average error
            float avg_error = total_error / num_samples;
            
            // Print progress every 1000 epochs
            
            std::cout << "Epoch " << epoch+1 << "/" << epochs << " - Error: " << avg_error << std::endl;
        }
    }

    // Function to evaluate the model performance
    void evaluate_model(const vecs<float>& inputs, const vecs<float>& targets) {
        int num_samples = inputs.num_of_vecs();
        if (num_samples != targets.num_of_vecs()) {
            std::cerr << "Error: Number of input samples (" << num_samples 
                      << ") does not match number of target samples (" << targets.num_of_vecs() 
                      << ")" << std::endl;
            return;
        }
        if (num_samples == 0) {
            std::cout << "Evaluation dataset is empty." << std::endl;
            return;
        }

        float total_loss = 0.0f;
        int correct_predictions = 0;

        for (int i = 0; i < num_samples; ++i) {
            vec<float> input = inputs(i);
            vec<float> target = targets(i);
            vec<float> prediction = forward(input);

            // Calculate loss for the sample
            float sample_loss = 0.0f;
            if (loss_func == "cross_entropy") {
                // Binary/Categorical cross-entropy loss
                for (int j = 0; j < prediction.size; ++j) {
                    // Clip values to avoid log(0) or log(1) issues
                    float pred_j = std::max(std::min(prediction(j), 1.0f - 1e-7f), 1e-7f);
                    // Check if target is binary (size 1) or one-hot encoded
                    if (target.size == 1) { 
                        sample_loss -= (target(0) * log(pred_j) + (1.0f - target(0)) * log(1.0f - pred_j));
                    } else { // Assuming one-hot encoded target
                         sample_loss -= target(j) * log(pred_j); // Simplified for one-hot
                    }
                }
                 if (target.size == 1 && prediction.size > 1) { // Handle binary case with multi-output sigmoid separately if needed
                     // This part might need adjustment based on specific binary classification setup
                     // Assuming first output neuron corresponds to the positive class probability for binary case
                     float pred_0 = std::max(std::min(prediction(0), 1.0f - 1e-7f), 1e-7f);
                     sample_loss = -(target(0) * log(pred_0) + (1.0f - target(0)) * log(1.0f - pred_0));
                 }

            } else { // Defaulting to Mean Squared Error
                vec<float> error_vec = target - prediction;
                for (int j = 0; j < error_vec.size; ++j) {
                    sample_loss += error_vec(j) * error_vec(j);
                }
                sample_loss /= error_vec.size; // Average squared error
            }
            total_loss += sample_loss;

            // Calculate accuracy (assuming classification)
            // Find index of max value in prediction and target (for one-hot encoding)
            int predicted_class = 0;
            float max_pred = prediction(0);
            for(int j = 1; j < prediction.size; ++j) {
                if (prediction(j) > max_pred) {
                    max_pred = prediction(j);
                    predicted_class = j;
                }
            }

            int target_class = 0;
             if (target.size == 1) { // Binary classification target (0 or 1)
                 target_class = static_cast<int>(round(target(0)));
                 // Adjust prediction logic for binary sigmoid output if needed
                 // Often comparing prediction(0) > 0.5 threshold
                 predicted_class = (prediction(0) > 0.5f) ? 1 : 0; 
             } else { // One-hot encoded target
                 float max_target = target(0);
                 for(int j = 1; j < target.size; ++j) {
                     if (target(j) > max_target) { // Find the index of the '1'
                         max_target = target(j);
                         target_class = j;
                     }
                 }
             }


            if (predicted_class == target_class) {
                correct_predictions++;
            }
        }

        float average_loss = total_loss / num_samples;
        float accuracy = static_cast<float>(correct_predictions) / num_samples;

        std::cout << "Evaluation Results:" << std::endl;
        std::cout << "  Average Loss: " << average_loss << std::endl;
        std::cout << "  Accuracy: " << accuracy * 100.0f << "% (" 
                  << correct_predictions << "/" << num_samples << ")" << std::endl;
    }

    // Test function to verify max_pool operation with a specific example
    void test_max_pool() {
        // Example input [1,2,3,4,5,6,7,8,9] representing a 3×3 matrix
        vec<float> input = {1,2,3,4,5,6,7,8,9};
        int input_size_x = 3;
        int input_size_y = 3;
        int pool_size = 2;
        int stride = 1;
        
        // Create a test pool layer
        int output_size_x = 2;
        int output_size_y = 2;
        pool_layer test_layer(input_size_x, input_size_y, output_size_x, output_size_y, pool_size, stride, 1);
        
        // Prepare the input for max pooling
        matrix<float> input_matrix = prep_vec(input, input_size_x, input_size_y, pool_size, stride, 0);
        
        // The input_matrix should now have 4 columns (one for each pooling region)
        // Each column should have 4 rows (one for each position in the 2x2 pooling window)
        
        // Expected pooling regions:
        // Region 0: [1,2,4,5] -> max is 5
        // Region 1: [2,3,5,6] -> max is 6
        // Region 2: [4,5,7,8] -> max is 8
        // Region 3: [5,6,8,9] -> max is 9
        
        // Perform max pooling
        vec<float> output = test_layer.max_pool(input_matrix,0);
        
        // Expected output: [5,6,8,9]
        std::cout << "Max pooling output: ";
        for(int i = 0; i < output.size; i++) {
            std::cout << output(i) << " ";
        }
        std::cout << std::endl;
        
        // Verify max_pool_weights matrix works
        // Convert input_matrix to a vector to match how max_pool_weights expects input
        
        // Test that max_pool_weights * flattened_regions produces expected output
        matrix<float> weights = test_layer.get_weights()(0);
        vec<float> check_output = weights * input;
        
        std::cout << "Weights matrix output for first region: " << check_output(0) << std::endl;
        
        // Print the max_pool_weights matrix
        std::cout << "Max pooling weights matrix:" << std::endl;
        for(int i = 0; i < weights.row; i++) {
            for(int j = 0; j < weights.col; j++) {
                std::cout << weights(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
};