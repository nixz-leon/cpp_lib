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
            result = element_mult(sigmoid(v), result-sigmoid(v));
            return result;
        }
        static vec<float> softmax(vec<float> &v){
            vec<float> result(v.size);
            float sum = 0.0f;
            for(int i = 0; i < v.size; i++){
                result(i) = exp(v(i));
                sum += result(i);
            }
            for(int i = 0; i < v.size; i++){
                result(i) = result(i)/sum;
            }
            return result;
        }
        static vec<float> softmax_deriv(vec<float> &v){
            vec<float> result(v.size, 1.0f);
            result = element_mult(softmax(v), result-softmax(v));
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
    public:
        vec<float> operator()(vec<float> &v, int call){
            return act_funcs[call](v);
        }
        vec<float> deriv(vec<float> &v, int call){
            return act_derivs[call](v);
        }
};

class layer{
    public:
    virtual ~layer() noexcept = default;
    virtual vec<float> forward(const vec<float>& input) = 0;
    virtual vec<float> calc_gradient(const vec<float>& prev_delta, const matrix<float>& prev_weight) = 0;
    virtual vec<float> calc_gradient_last(const vec<float>& actual, std::string loss_func) = 0;
    virtual void update_params(float learning_rate) = 0;
    virtual matrix<float> get_weights() const = 0;
    virtual int get_output_size() const = 0;
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
    
    // Create the result matrix directly - each submatrix becomes a column
    // Rows = filter_size * filter_size, Cols = total_submatrices
    matrix<float> result(filter_size * filter_size, total_submatrices);
    
    // Check if we should use GPU acceleration
    if (OpenCLAccelerator::isOpenCLAvailable() && 
        OpenCLAccelerator::shouldUseGPU(total_submatrices * filter_size * filter_size)) {
        
        // Allocate result data
        float* result_data = new float[total_submatrices * filter_size * filter_size];
        
        // Call the GPU implementation
        bool success = OpenCLAccelerator::prepVecGPU(
            image_vec.data, image_width, image_height, 
            result_data, filter_size, stride, padding
        );
        
        if (success) {
            // Copy data from result_data to the matrix
            // In the kernel, each submatrix is a contiguous block
            // Need to transpose the data arrangement to get columns
            for (int submatrix_idx = 0; submatrix_idx < total_submatrices; submatrix_idx++) {
                for (int i = 0; i < filter_size; i++) {
                    for (int j = 0; j < filter_size; j++) {
                        // Source index in result_data
                        int src_idx = submatrix_idx * (filter_size * filter_size) + i * filter_size + j;
                        // Target index in result matrix (column-oriented)
                        int tgt_idx = (i * filter_size + j) * total_submatrices + submatrix_idx;
                        result.data[tgt_idx] = result_data[src_idx];
                    }
                }
            }
            
            // Clean up the temporary array
            delete[] result_data;
            return result;
        }
        
        // Fall back to CPU if GPU implementation failed
        delete[] result_data;
    }
    
    // CPU implementation - directly filling the matrix
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
                        value = image_vec(img_x * image_height + img_y);
                    }
                    
                    // Store directly in the matrix at row (fi*filter_size+fj), column (submatrix_idx)
                    result(fi * filter_size + fj, submatrix_idx) = value;
                }
            }
            
            submatrix_idx++;
        }
    }
    return result;
}




/*
class conv_layer : public layer{
    private:
    int input_size_x;
    int input_size_y;
    int output_size_x;
    int output_size_y;
    int filter_size;
    int stride;
    int padding;
    matrix<float> weights;
    vec<float> bias;
    vec<float> input;
    matrix<float> input_matrix;
    vec<float> output;
    vec<float> preactivation;
    vec<float> delta;
    int call;
    act_func_vec act_func;

    conv_layer(int input_size_x, int input_size_y, int output_size_x, int output_size_y, int num_of_filters, int filter_size, int stride, int padding, std::string activation_func){
        this->input_size_x = input_size_x;
        this->input_size_y = input_size_y;
        this->output_size_x = output_size_x;
        this->output_size_y = output_size_y;
        this->num_of_filters = num_of_filters;
        this->filter_size = filter_size;
        this->stride = stride;
        this->padding = padding;

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
        bias = vec<float>(num_of_filters);

        int num_submatrices_x = (input_size_x + 2 * padding - filter_size) / stride + 1;
        int num_submatrices_y = (input_size_y + 2 * padding - filter_size) / stride + 1;
        int total_submatrices = num_submatrices_x * num_submatrices_y;
        input_matrix = matrix<float>(filter_size*filter_size, total_submatrices);
    }

    vec<float> forward(const vec<float>& in) override {
        input = in;
        input_matrix = prep_vec(input, input_size_x, input_size_y, filter_size, stride, padding);
        preactivation = weights*input_matrix;
        preactivation = per_column_add(preactivation, bias);
        output = act_func(preactivation, call);
        return output;
    }

    vec<float> calc_gradient(const vec<float>& prev_delta, const matrix<float>& prev_weight) override {
        delta = transpose(prev_weight)*prev_delta;
        delta = element_mult(delta, act_func.deriv(preactivation, call));
        return delta;
    }
    
    void update_params(float learning_rate) override {
        matrix<float> weight_update = transpose(input_matrix*delta);
        weights = weights - learning_rate * weight_update;
        bias = bias - learning_rate * delta;
    }

    int get_output_size() const override {
        return output_size_x * output_size_y;
    }
    
    matrix<float> get_weights() const override {
        return weights;
    }
};
*/

class pool_layer : public layer{
    public:
    int input_size_x;
    int input_size_y;
    int output_size_x;
    int output_size_y;
    int pool_size;
    int stride;
    vec<float> input;
    matrix<float> input_matrix;
    vec<float> output;
    vec<float> delta;
    matrix<float> max_pool_weights;

    vec<float> max_pool(matrix<float> &data){
        vec<float> result(data.col);
        // Create a weights matrix that transforms the original input vector to max pooled output
        // For a 3x3 input with 2x2 pooling and stride 1, we get 4 regions
        // The weights should be 4x9 (4 outputs x 9 input elements)
        matrix<float> max_pool_weights(data.col, input_size_x * input_size_y);
        
        // Initialize weights to zero
        for(int i = 0; i < max_pool_weights.row; i++) {
            for(int j = 0; j < max_pool_weights.col; j++) {
                max_pool_weights(i, j) = 0.0f;
            }
        }
        
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
            // Calculate the starting point of this region in the original input
            int region_row = (j / ((input_size_x - pool_size) / stride + 1)) * stride;
            int region_col = (j % ((input_size_x - pool_size) / stride + 1)) * stride;
            
            // Calculate which position in the pooling window had the max value
            int window_row = max_idx / pool_size;
            int window_col = max_idx % pool_size;
            
            // Calculate the original index in the input
            int orig_row = region_row + window_row;
            int orig_col = region_col + window_col;
            int orig_idx = orig_row * input_size_x + orig_col;
            
            // Place a 1 in the weight matrix where output j takes value from input orig_idx
            max_pool_weights(j, orig_idx) = 1.0f;
        }
        
        // Store the weights matrix for backpropagation
        this->max_pool_weights = max_pool_weights;
        
        return result;
    }

    pool_layer(int input_size_x, int input_size_y, int output_size_x, int output_size_y, int pool_size, int stride){
        this->input_size_x = input_size_x;
        this->input_size_y = input_size_y;
        this->output_size_x = output_size_x;
        this->output_size_y = output_size_y;
        this->pool_size = pool_size;
        this->stride = stride;

        int num_submatrices_x = (input_size_x - pool_size) / stride + 1;
        int num_submatrices_y = (input_size_y - pool_size) / stride + 1;
        int total_submatrices = num_submatrices_x * num_submatrices_y;
        input_matrix = matrix<float>(pool_size*pool_size, total_submatrices);
        output = vec<float>(total_submatrices);
    }

    vec<float> forward(const vec<float>& in) override {
        input = in;
        input_matrix = prep_vec(input, input_size_x, input_size_y, pool_size, stride, 0);
        output = max_pool(input_matrix);
        return output;
    }

    vec<float> calc_gradient(const vec<float>& prev_delta, const matrix<float>& prev_weight) override {
        delta = transpose(prev_weight)*prev_delta;
        return delta;
    }

    vec<float> calc_gradient_last(const vec<float>& actual, std::string loss_func) override {
        // Pooling layers don't have trainable parameters, so this is a no-op
    }
    
    void update_params(float learning_rate) override {
        // Pooling layers don't have trainable parameters, so this is a no-op
    }
    
    matrix<float> get_weights() const override {
        return max_pool_weights;
    }
    
    int get_output_size() const override {
        return output_size_x * output_size_y;
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

    vec<float> forward(const vec<float>& in) override {
        input = in;  // Store input for backprop
        
        // Matrix-vector multiplication for forward pass
        preactivation = weights*input + bias;
        
        // Apply activation function
        vec<float> preact_copy = preactivation; // Copy to avoid modifying preactivation
        output = act_func(preact_copy, call);
        
        return output;
    }
    
    vec<float> calc_gradient(const vec<float>& prev_delta, const matrix<float>& prev_weight) override {
        delta = transpose(prev_weight)*prev_delta;
        delta = element_mult(delta, act_func.deriv(preactivation, call));
        return delta;
    }
    
    vec<float> calc_gradient_last(const vec<float>& actual, std::string loss_func) override {
        delta = output - actual;
        if(!(loss_func == "cross_entropy")){
            delta = element_mult(delta, act_func.deriv(preactivation, call));
        }
        return delta;
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
    
    matrix<float> get_weights() const override {
        return weights;
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
    
    vec<float> forward(const vec<float>& input){
        vec<float> current_input = input;
        for(auto& layer : layers){
            current_input = layer->forward(current_input);
        }
        return current_input;
    }
    
    void backpropagation(const vec<float>& input, const vec<float>& actual, std::string loss_func, float learning_rate){
        // Forward pass to ensure all layers have calculated their outputs
        forward(input);
        
        // If no layers, do nothing
        if (layers.empty()) return;
        
        // Calculate gradient for the last layer
        vec<float> delta = layers.back()->calc_gradient_last(actual, loss_func);
        
        // Backpropagate through the remaining layers
        for (int i = layers.size() - 2; i >= 0; i--) {
            // Get weights from the next layer using the new get_weights method
            matrix<float> next_layer_weights = layers[i + 1]->get_weights();
            
            // Backpropagate delta through previous layers
            delta = layers[i]->calc_gradient(delta, next_layer_weights);
        }
        
        // Update parameters for all layers
        for (auto& layer : layers) {
            layer->update_params(learning_rate);
        }
    }
    
    // Simplified training method for single examples
    void train(const vec<float>& input, const vec<float>& target) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            backpropagation(input, target, loss_func, learning_rate);
        }
    }
    
    // Batch training method for multiple examples
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
            }
            
            // Calculate average error
            float avg_error = total_error / num_samples;
            
            // Print progress every 1000 epochs
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << "/" << epochs << " - Error: " << avg_error << std::endl;
            }
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
        pool_layer test_layer(input_size_x, input_size_y, output_size_x, output_size_y, pool_size, stride);
        
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
        vec<float> output = test_layer.max_pool(input_matrix);
        
        // Expected output: [5,6,8,9]
        std::cout << "Max pooling output: ";
        for(int i = 0; i < output.size; i++) {
            std::cout << output(i) << " ";
        }
        std::cout << std::endl;
        
        // Verify max_pool_weights matrix works
        // Convert input_matrix to a vector to match how max_pool_weights expects input
        
        // Test that max_pool_weights * flattened_regions produces expected output
        matrix<float> weights = test_layer.get_weights();
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