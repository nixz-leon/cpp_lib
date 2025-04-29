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
    virtual vec<float> update_params(float learning_rate) = 0;
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
    
    vec<float> update_params(float learning_rate) override {
        // Compute weight updates using gradient descent
        matrix<float> weight_update = outer_product(delta, input);
        
        // Apply updates
        for(int i = 0; i < weights.row; i++) {
            for(int j = 0; j < weights.col; j++) {
                weights(i, j) -= learning_rate * weight_update(i, j);
            }
        }
        
        // Update biases
        for(int i = 0; i < bias.size; i++) {
            bias(i) -= learning_rate * delta(i);
        }
        
        return vec<float>(input_size);
    }
    
    int get_output_size() const override {
        return output_size;
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
            // Get weights from the next layer
            dense_layer* next_layer = dynamic_cast<dense_layer*>(layers[i + 1].get());
            if (!next_layer) continue;
            
            // Backpropagate delta through previous layers
            delta = layers[i]->calc_gradient(delta, next_layer->weights);
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
};