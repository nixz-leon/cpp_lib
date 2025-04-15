#include "Lin_alg.hpp"
#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <cstdarg>
#include <iomanip>
#include <cmath>  // For isnan() and isinf()

class activation_functions{
    private:
        // Activation functions
        static double relu(double &x){return x>0? x:0;};
        static double relu_deriv(double &x){return x>0? 1:0;};
        static double leaky_relu(double &x){return x>0? x:0.01*x;};
        static double leaky_relu_deriv(double &x){return x>0? 1:0.01;};
        static double sigmoid(double &x){return 1/(1+exp(-x));};
        static double sigmoid_deriv(double &x){return sigmoid(x)*(1-sigmoid(x));};
        static double tanh(double &x){return (exp(x)-exp(-x))/(exp(x)+exp(-x));};
        static double tanh_deriv(double &x){return 1-tanh(x)*tanh(x);};
        double (*act_funcs[4])(double&)={relu, leaky_relu, sigmoid, tanh};
        double (*act_derivs[4])(double&)={relu_deriv, leaky_relu_deriv, sigmoid_deriv, tanh_deriv};
        int call;
    public:
        activation_functions(std::string name){
            if(name == "relu"){
                call = 0;
            }else if(name == "leaky_relu"){
                call = 1;
            }else if(name == "sigmoid"){
                call = 2;
            }else if(name == "tanh"){
                call = 3;
            }else{
                //defaults to relu;
                call =0;
            }
        }
        inline double operator()(double x){
            return act_funcs[call](x);
        };
        inline double deriv(double x){
            return act_derivs[call](x);
        };
        inline vec<double> operator()(vec<double> &v){
            vec<double> result(v.size);
            for(int i = 0; i < v.size; i++){
                result(i) = act_funcs[call](v(i));
            }
            return result;
        };
        inline vec<double> deriv(vec<double> &v){
            vec<double> result(v.size);
            for(int i = 0; i < v.size; i++){
                result(i) = act_derivs[call](v(i));
            }
            return result;
        };
        inline matrix<double> operator()(matrix<double> &m){
            matrix<double> result(m.row, m.col);
            for(int i = 0; i < m.row; i++){
                for(int j = 0; j < m.col; j++){
                    result(i,j) = act_funcs[call](m(i,j));
                }
            }
            return result;
        }
        inline matrix<double> deriv(matrix<double> &m){
            matrix<double> result(m.row, m.col);
            for(int i = 0; i < m.row; i++){
                for(int j = 0; j < m.col; j++){
                    result(i,j) = act_derivs[call](m(i,j));
                }
            }
            return result;
        }
};

class loss_function{
    private:
        static double mse(vec<double> &pred, vec<double> &actual){
            vec<double> error = pred - actual;
            return (error * error);
        };
        static vec<double> mse_deriv(vec<double> &pred, vec<double> &actual){
            vec<double> error = 2*(pred - actual);
            return error;
        }
        static double mae(vec<double> &pred, vec<double> &actual){
            vec<double> e = pred - actual;
            double loss = 0.0;
            for(int i =0; i < e.size; i++){
                loss += std::abs(e(i));
            }
            return loss/e.size;
        }
        static vec<double> mae_deriv(vec<double> &pred, vec<double> &actual){
            vec<double> e = pred - actual;
            vec<double> result(e.size);
            for(int i = 0; i < e.size; i++){
                result(i) = (e(i) > 0) ? 1 : -1;
            }
            return result;
        }
        static double cross_entropy(vec<double> &pred, vec<double> &actual){
            double loss = 0.0;
            for(int i = 0; i < pred.size; i++){
                if(actual(i) > 0) {
                    loss -= actual(i) * std::log(std::max(pred(i), 1e-15));
                }
            }
            return loss;
        }
        static vec<double> cross_entropy_deriv(vec<double> &pred, vec<double> &actual){
            vec<double> error(pred.size);
            for(int i = 0; i < pred.size; i++){
                if(actual(i) > 0) {
                    error(i) = -actual(i) / std::max(pred(i), 1e-15);
                }
            }
            return error;
        }
        //softmax
        static double softmax(vec<double> &pred, vec<double> &actual){
            double loss = 0.0;
            for(int i = 0; i < pred.size; i++){
                loss -= actual(i) * std::log(std::max(pred(i), 1e-15));
            }
            return loss;
        }
        static vec<double> softmax_deriv(vec<double> &pred, vec<double> &actual){
            vec<double> error(pred.size);
            for(int i = 0; i < pred.size; i++){
                error(i) = pred(i) - actual(i);
            }
            return error;
        };
        static double hinge(vec<double> &pred, vec<double> &actual){
            double loss = 0.0;
            for(int i = 0; i < pred.size; i++){
                loss += std::max(0.0, 1.0 - actual(i) * pred(i));
            }
            return loss/pred.size;
        };
        static vec<double> hinge_deriv(vec<double> &pred, vec<double> &actual){
            vec<double> error(pred.size);
            for(int i = 0; i < pred.size; i++){
                error(i) = (actual(i) * pred(i) < 1) ? -actual(i) : 0;
            }
            return error;
        };
        double (*loss_funcs[5])(vec<double>&, vec<double>&) = {mse, mae, cross_entropy, softmax, hinge};
        vec<double> (*loss_deriv[5])(vec<double>&, vec<double>&) = {mse_deriv, mae_deriv, cross_entropy_deriv, softmax_deriv, hinge_deriv};
        int call =0;
    public:
        loss_function(std::string name){
            if(name == "mse"){
                call = 0;
            }else if(name == "mae"){
                call = 1;
            }else if(name == "cross_entropy"){
                call = 2;
            }else if(name == "softmax"){
                call = 3;
            }else if(name == "hinge"){
                call = 4;
            }else{
                call = 0;
            }
        }
        inline double operator()(vec<double> &pred, vec<double> &actual){
            return loss_funcs[call](pred, actual);
        };
        inline vec<double> deriv(vec<double> &pred, vec<double> &actual){
            return loss_deriv[call](pred, actual);
        };
};


class nn_linear_double{
    private:
        matrix<double> weights;     // Weight matrix [output_size × input_size]
        vec<double> bias;          // Bias vector [output_size]
        vec<double> inputs;        // Last input received
        vec<double> outputs;       // Last output produced
        vec<double> gradients;     // Gradients for backprop
        int input_size;
        int output_size; 
        activation_functions fn;
        bool softmax = false;

        inline void init_mat(matrix<double> &mat, double scale){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dist(0.0, scale);
            for(int i =0; i < mat.row; i++){
                for(int j =0; j < mat.col; j++){
                    mat(i,j) = dist(gen);
                }
            }
        };
        
        void init_vec(vec<double> &vec, double scale){
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<double> dist(0.0, scale);
            for(int i=0; i<vec.size; i++){
                vec(i) = dist(gen);
            }
        };
        
        inline void sanitize(double &x) {
            if (std::isnan(x) || std::isinf(x)) {
                x = 0.0;
            }
            const double MAX_VALUE = 1e6;
            if (x > MAX_VALUE) x = MAX_VALUE;
            if (x < -MAX_VALUE) x = -MAX_VALUE;
        }
        
        inline void sanitize(vec<double> &v) {
            for(int i = 0; i < v.size; i++) {
                sanitize(v(i));
            }
        }

        inline void apply_softmax(vec<double> &v) {
            double max_val = v(0);
            for(int i = 1; i < v.size; i++) {
                if(v(i) > max_val) max_val = v(i);
            }
            
            double sum = 0.0;
            for(int i = 0; i < v.size; i++) {
                v(i) = exp(v(i) - max_val);
                sum += v(i);
            }
            
            for(int i = 0; i < v.size; i++) {
                v(i) /= sum;
                sanitize(v(i));
            }
        }

    public:
        // Constructor
        nn_linear_double(int input_dim, int output_dim, std::string act_func = "relu", bool use_softmax = false) 
            : input_size(input_dim), output_size(output_dim), fn(act_func), softmax(use_softmax) {
            
            weights = matrix<double>(output_size, input_size);
            bias = vec<double>(output_dim);
            gradients = vec<double>(output_size);
            
            // He initialization for weights
            double weight_scale = std::sqrt(2.0 / input_size);
            init_mat(weights, weight_scale);
            init_vec(bias, 0.01);
        }

        // Forward pass
        inline vec<double> forward(vec<double> &input) {
            sanitize(input);
            inputs = input;
            vec<double> pre_activation = weights * inputs + bias;
            sanitize(pre_activation);
            outputs = fn(pre_activation);
            return outputs;
        }

        // Backward pass - returns gradients for the input
        inline vec<double> backward(vec<double> &grad_output, double learning_rate) {
            if(softmax) {
                gradients = grad_output;  // For softmax, gradient is direct
            } else {
                gradients = element_mult(grad_output, fn.deriv(outputs));
            }
            
            // Compute gradients for weights and bias
            matrix<double> weight_gradients = outer_product(gradients, inputs);
            vec<double> bias_gradients = gradients;
            
            // Update weights and bias
            weights = weights - learning_rate * weight_gradients;
            bias = bias - learning_rate * bias_gradients;
            
            // Return gradients for the input (for backprop to previous layer)
            return transpose(weights) * gradients;
        }

        // Getters for dimensions
        inline int get_input_size(){ return input_size; }
        inline int get_output_size(){ return output_size; }
        
        // Getters for weights and bias
        inline  matrix<double>& get_weights() { return weights; }
        inline  vec<double>& get_bias() { return bias; }
        
        // Get last input and output (useful for debugging)
        inline vec<double>& get_last_input()  { return inputs; }
        inline vec<double>& get_last_output()  { return outputs; }
    
};

class NeuralNetwork {
private:
    std::vector<nn_linear_double> layers;
    int input_size;
    double learning_rate;
    int epochs;
    bool use_softmax;
    vec<double> last_output;
    vec<double> last_input;
    loss_function loss;  // Use the loss_function class
    inline void sanitize(double &x) {
        if (std::isnan(x) || std::isinf(x)) {
            x = 0.0;
        }
        const double MAX_VALUE = 1e6;
        if (x > MAX_VALUE) x = MAX_VALUE;
        if (x < -MAX_VALUE) x = -MAX_VALUE;
    }
    inline void apply_softmax(vec<double> &v) {
        double max_val = v(0);
        for(int i = 1; i < v.size; i++) {
            if(v(i) > max_val) max_val = v(i);
        }
        
        double sum = 0.0;
        for(int i = 0; i < v.size; i++) {
            v(i) = exp(v(i) - max_val);
            sum += v(i);
        }
        
        for(int i = 0; i < v.size; i++) {
            v(i) /= sum;
            sanitize(v(i));
        }
    }
    
public:
    // Constructor with layer sizes and training parameters
    NeuralNetwork(int nun_input, double lr = 0.01,int num_epochs = 100,std::string loss_func = "mse") 
        : input_size(nun_input), learning_rate(lr), epochs(num_epochs), loss(loss_func) {
            if(loss_func != "softmax" || loss_func != "cross_entropy"){
                use_softmax = false;
            }
            else{
                use_softmax = true;
            }
        };
        
    void add_layer(int num_nodes, std::string activation = "leaky_relu", bool is_last = true){
        if(layers.size() != 0){
            int prev_size = layers.back().get_output_size();
            layers.emplace_back(prev_size, num_nodes, activation, (is_last && use_softmax));
        }
        else{
            layers.emplace_back(input_size, num_nodes, activation, (is_last && use_softmax));
        }
    };
    
    // Forward pass through all layers
    vec<double> forward(vec<double>& input) {
        vec<double> current = input;
        for(auto& layer : layers) {
            current = layer.forward(current);
        }
        if(use_softmax){
            apply_softmax(current);
        }
        return current;
    }
    
    // Backward pass through all layers
    void backward(vec<double>& target) {
        // Start with the last layer
        vec<double> grad = loss.deriv(layers.back().get_last_output(), target);
        
        // Backpropagate through layers in reverse
        for(auto it = layers.rbegin(); it != layers.rend(); ++it) {
            grad = it->backward(grad, learning_rate);
        }
    }
    
    // Train the network on a dataset
    void train(vecs<double>& inputs, vecs<double>& targets, int num_epochs = -1) {
        if(num_epochs < 0) num_epochs = epochs;
        
        if(inputs.num_of_vecs() != targets.num_of_vecs()) {
            std::cerr << "Error: Number of inputs and targets must match" << std::endl;
            return;
        }
        
        int num_samples = inputs.num_of_vecs();
        double reg_lambda = 0.0001;  // L2 regularization factor
        
        for(int epoch = 0; epoch < num_epochs; epoch++) {
            double total_loss = 0.0;            
            for(int idx = 0; idx < num_samples; idx++) {
                // Forward pass
                vec<double> output = forward(inputs(idx));
                
                // Compute loss using the loss_function class
                total_loss += loss(output, targets(idx));
                // Backward pass
                backward(targets(idx));
                // Apply L2 regularization
                for(auto& layer : layers) {
                    layer.get_weights() = layer.get_weights() * (1.0 - learning_rate * reg_lambda);
                }
            }
        }
    }
};


/*
class conv2d_layer {
private:
    int input_dim_x;
    int input_dim_y;
    int output_dim_y;
    int output_dim_x;
    int num_filters;
    int filter_dim;
    activation_functions fn;
    bool use_padding = false;   // Whether to use padding
    int padding_y = 0;          // Padding in y dimension
    int padding_x = 0;          // Padding in x dimension
    double learning_rate = 0.01; // Learning rate for updates
    
    int stride = 1;
    vec<double> biases;         // [num_filters]
    matrix<double> last_input;  // Last input received
    matrix<double> last_output; // Last output produced
    matrix<double> gradients;   // Gradients for backprop
    matricies<double> filters;  // [num_filters, filter_dim, filter_dim]
    matricies<double> output;   // [num_filters, output_dim_x, output_dim_y]
    

    void init_filters(double scale){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, scale);
        for(int i = 0; i < num_filters; i++){
            for(int j = 0; j < filter_dim; j++){
                for(int k = 0; k < filter_dim; k++){
                    filters(i)(j,k) = dist(gen);
                }
            }
        }
    }

public:
    conv2d_layer(int input_x, int input_y, int filters_count, int filter_dim, std::string activation = "leaky_relu", bool use_padding = false, int stride = 1)
    : input_dim_x(input_x), input_dim_y(input_y), num_filters(filters_count), filter_dim(filter_dim), fn(activation), use_padding(use_padding), stride(stride) 
    {
        output_dim_x = (input_dim_x - filter_dim) / (stride+1);
        output_dim_y = (input_dim_y - filter_dim) / (stride+1);
        filters = matricies<double>(num_filters, filter_dim, filter_dim);
        biases = vec<double>(num_filters);
        init_filters(0.01);
    };

    matricies<double> convolve(matrix<double> &input){
        // Store input for backpropagation
        last_input = input;
        
        // Calculate output dimensions based on stride and padding
        if(use_padding){    
            int padded_input_x = input_dim_x + 2 * padding_x;
            int padded_input_y = input_dim_y + 2 * padding_y;
        }
        else{
            int padded_input_x = input_dim_x;
            int padded_input_y = input_dim_y;
        }
        
        // Initialize output with proper dimensions
        matricies<double> output(num_filters, output_dim_x, output_dim_y);
        
        // For each filter
        for(int f = 0; f < num_filters; f++){
            // For each output position
            for(int y = 0; y < output_dim_y; y++){
                for(int x = 0; x < output_dim_x; x++){
                    double sum = 0.0;
                    // For each position in the filter
                    for(int fy = 0; fy < filter_dim; fy++){
                        for(int fx = 0; fx < filter_dim; fx++){
                            // Calculate input position with stride and padding
                            int in_y = y * stride + fy - padding_y;
                            int in_x = x * stride + fx - padding_x;
                            
                            // Skip if out of bounds
                            if(in_y < 0 || in_y >= input_dim_y || in_x < 0 || in_x >= input_dim_x){
                                continue;
                            }
                            // Accumulate the convolution sum
                            sum += input(in_y, in_x) * filters(f)(fy, fx);
                        }
                    }
                    
                    // Add bias and apply activation
                    sum += biases(f);
                    output(f)(y, x) = fn(sum);
                }
            }
        }
        
        // Store output for backpropagation
        last_output = output(0);  // Store first filter's output as last_output
        return output;
    };
    
 }; 
 */