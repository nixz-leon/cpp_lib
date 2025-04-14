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

//once I get this done, I will transition to working on the more complex NN class with multishaped neural networks


class nn_double{
    private:
        matrix<double> in_weights;     // Input layer weights
        matrix<double> out_weights;    // Output layer weights
        matricies<double> weights;     // Hidden layer weights
        vecs<double> biases;          // Hidden layer biases
        vec<double> out_bias;         // Output layer bias
        vecs<double> layer_outputs;   // Hidden layer outputs
        vec<double> in_layer_outputs; // Input layer outputs
        vec<double> out_layer_outputs;// Output layer outputs
        int input_size;
        int output_size; 
        int hidden_layers;
        int per_layer;
        int epochs;
        double learning_rate;
        bool softmax = true;
        
<<<<<<< HEAD
        // Stable leaky ReLU with alpha=0.01
        double act_func(double x) {
            return x > 0 ? x : 0.1 * x;
        }
        
        double act_deriv(double x) {
            return x > 0 ? 1.0 : 0.1;
        }
        
        // Ensure values are finite and within reasonable bounds
        double sanitize(double x) {
            // Check for NaN or infinity
            if (std::isnan(x) || std::isinf(x)) {
                return 0.0;
            }
            // Clip extremely large values
            const double MAX_VALUE = 1e6;
            if (x > MAX_VALUE) return MAX_VALUE;
            if (x < -MAX_VALUE) return -MAX_VALUE;
            return x;
=======
        // RNG for weight initialization
        std::random_device rd;
        std::mt19937 gen;
        
        // Pre-allocated memory for calculations
        vec<double> final_error;
        vecs<double> layer_errors;
        vec<double> error_temp;
        matrix<double> hidden_updates;
        vec<double> pre_activation;
        
        // Stable leaky ReLU with alpha=0.01
        //double act_func(double x) {return x > 0 ? x : 0.01 * x;}
        //double act_deriv(double x) {return x > 0 ? 1.0 : 0.01;}
        inline void act_func(double &x) {x =1.0/(1.0+exp(-x));}
        inline void act_deriv(double &x) {x= x*(1.0-x);}
        
        // Ensure values are finite and within reasonable bounds
        inline void sanitize(double &x) {
            // Check for NaN or infinity and replace with 0
            bool invalid = std::isnan(x) || std::isinf(x);
            x = invalid ? 0.0 : x;
            
            // Clip to MAX_VALUE
            const double MAX_VALUE = 1e6;
            x = (x > MAX_VALUE) ? MAX_VALUE : x;
            x = (x < -MAX_VALUE) ? -MAX_VALUE : x;
>>>>>>> 089e32ab5a9e6068432af90a86f456738c9279da
        }
        inline void sanitize(vec<double> &v){
            for(int i = 0; i < v.size; i++){
                sanitize(v(i));
            }
        }
        inline void sanitize(matrix<double> &m){
            for(int i = 0; i < m.row; i++){
                for(int j = 0; j < m.col; j++){
                    sanitize(m(i,j));
                }
            }
        }
        inline void apply_act(vec<double> &v) {
            int i = 0;
            // Unroll the loop in chunks of 4
            for(; i <= v.size - 4; i += 4) {
                act_func(v(i));
                act_func(v(i+1));
                act_func(v(i+2));
                act_func(v(i+3));
                sanitize(v(i));
                sanitize(v(i+1));
                sanitize(v(i+2));
                sanitize(v(i+3));
            }
            // Handle remaining elements
            for(; i < v.size; i++) {
                act_func(v(i));
                sanitize(v(i));
            }
        }   
        
        inline void apply_deriv(vec<double> &v) {
            int i = 0;
            // Unroll the loop in chunks of 4
            for(; i <= v.size - 4; i += 4) {
                act_deriv(v(i));
                act_deriv(v(i+1));
                act_deriv(v(i+2));
                act_deriv(v(i+3));
                sanitize(v(i));
                sanitize(v(i+1));
                sanitize(v(i+2));
                sanitize(v(i+3));
            }
            // Handle remaining elements
            for(; i < v.size; i++) {
                act_deriv(v(i));
                sanitize(v(i));
            }
        }

    public:
        void init_mat(matrix<double> &mat, double scale){
            std::normal_distribution<double> dist(0.0, scale);
            for(int i = 0; i < mat.row; i++){
                for(int j = 0; j < mat.col; j++){
                    mat(i,j) = dist(gen);
                }
            }
        };
        void init_vec(vec<double> &vec, double scale){
            std::normal_distribution<double> dist(0.0, scale);
            for(int i = 0; i < vec.size; i++){
                vec(i) = dist(gen);
            }
        }
        nn_double(int input_dim, int output_dim, int hidden_layers_count, 
                  int nodes_per_layer, int num_epochs = 100, double lr = 0.01, 
                  bool use_softmax = true) : gen(rd()) {
            input_size = input_dim;
            output_size = output_dim;
            hidden_layers = hidden_layers_count;
            per_layer = nodes_per_layer;
            epochs = num_epochs;
            learning_rate = lr;
            softmax = use_softmax;  // Fixed parameter shadowing

            in_weights = matrix<double>(per_layer, input_size);
            out_weights = matrix<double>(output_size, per_layer); 
            weights = matricies<double>(hidden_layers-1, per_layer, per_layer);
            biases = vecs<double>(hidden_layers,per_layer);
            out_bias = vec<double>(output_dim);
            layer_outputs = vecs<double>(hidden_layers, per_layer);
            in_layer_outputs = vec<double>(input_size);
            out_layer_outputs = vec<double>(output_dim);

            // Initialize pre-allocated memory
            final_error = vec<double>(output_size);
            layer_errors = vecs<double>(hidden_layers, per_layer);
            error_temp = vec<double>(per_layer);
            hidden_updates = matrix<double>(per_layer, per_layer);
            pre_activation = vec<double>(std::max(per_layer, output_size));

            // He initialization for ReLU variants
            double input_scale = std::sqrt(2.0 / input_size);
            double hidden_scale = std::sqrt(2.0 / per_layer);
            double output_scale = std::sqrt(2.0 / per_layer);

            init_mat(in_weights, input_scale);
            init_mat(out_weights, output_scale);
           
            
            
            if (hidden_layers > 0) {
                for(int i = 0; i < hidden_layers-1; i++){
                    init_mat(weights(i), hidden_scale);
                }
                
                for(int i = 0; i < hidden_layers; i++){
                    init_vec(biases(i), 0.01);
                }
            }
            
            
            init_vec(out_bias,0.01);
        };
        inline double mse(vec<double> &pred, vec<double> &actual){
            double sum = 0.0;
            for(int i = 0; i < pred.size; i++) {
                double diff = pred(i) - actual(i);
                sum += diff * diff;
            }
            return sum / pred.size;
        };
        inline void mse_deriv(vec<double> &pred, vec<double> &actual, vec<double> &result){
            for(int i = 0; i < pred.size; i++) {
                result(i) = 2.0 * (pred(i) - actual(i));
            }
        }


        
        inline vec<double> forward_prop(vec<double> &input) {
            // Store input for backpropagation
            in_layer_outputs = input;
            
            // Only sanitize input once
            sanitize(in_layer_outputs);
            
            if(hidden_layers > 0) {
                // First hidden layer
                pre_activation = in_weights * in_layer_outputs + biases(0);
                layer_outputs(0) = pre_activation;
                apply_act(layer_outputs(0));
                
                // Subsequent hidden layers
                for(int i = 1; i < hidden_layers; i++) {
                    pre_activation = weights(i-1) * layer_outputs(i-1) + biases(i);
                    layer_outputs(i) = pre_activation;
                    apply_act(layer_outputs(i));
                }
                
                // Output layer
                pre_activation = out_weights * layer_outputs.back() + out_bias;
                out_layer_outputs = pre_activation;
                apply_act(out_layer_outputs);
            } else {
                // Single layer network
                pre_activation = in_weights * in_layer_outputs + out_bias;
                out_layer_outputs = pre_activation;
                apply_act(out_layer_outputs);
            }
            
            return out_layer_outputs;
        };
        
        inline void back_prop(vec<double> &actual) {
            if(hidden_layers > 0){
                apply_deriv(out_layer_outputs);
                
                // Calculate error derivative
                mse_deriv(out_layer_outputs, actual, final_error);
                
                // Apply activation derivative
                for(int i = 0; i < final_error.size; i++) {
                    final_error(i) *= out_layer_outputs(i);
                }
                
                // Rest of the function
                out_weights = out_weights - learning_rate * outer_product(final_error, layer_outputs.back());
                out_bias = out_bias - learning_rate * final_error;
                apply_deriv(layer_outputs(hidden_layers-1));
                layer_errors(hidden_layers-1) = element_mult(transpose(out_weights) * final_error, layer_outputs(hidden_layers-1));
                if (hidden_layers > 1) {
                    hidden_updates = outer_product(layer_errors(hidden_layers-1), layer_outputs(hidden_layers-2));
                    
                    // Apply and sanitize weight updates
                    for(int i = 0; i < weights(hidden_layers-2).row; i++) {
                        for(int j = 0; j < weights(hidden_layers-2).col; j++) {
                            double update = learning_rate * hidden_updates(i, j);
                            sanitize(update);
                            weights(hidden_layers-2)(i, j) -= update;
                        }
                    }
                }
                
                // Update biases for the last hidden layer
                for(int i = 0; i < biases(hidden_layers-1).size; i++) {
                    double update = learning_rate * layer_errors(hidden_layers-1)(i);
                    sanitize(update);
                    biases(hidden_layers-1)(i) -= update;
                }
                
                // Backpropagate through remaining hidden layers
                for(int i = hidden_layers-2; i >= 0; i--) {
                    apply_deriv(layer_outputs(i));
                    layer_errors(i) = element_mult(transpose(weights(i)) * layer_errors(i+1), layer_outputs(i));
                    if(i > 0){
                        hidden_updates = outer_product(layer_errors(i), layer_outputs(i-1));
                        for(int r = 0; r < weights(i-1).row; r++) {
                            for(int c = 0; c < weights(i-1).col; c++) {
                                double update = learning_rate * hidden_updates(r, c);
                                sanitize(update);
                                weights(i-1)(r, c) -= update;
                            }
                        }
                    } else {
                        hidden_updates = outer_product(layer_errors(0), in_layer_outputs);
                        for(int r = 0; r < in_weights.row; r++) {
                            for(int c = 0; c < in_weights.col; c++) {
                                double update = learning_rate * hidden_updates(r, c);
                                sanitize(update);
                                in_weights(r, c) -= update;
                            }
                        }
                    }
                    biases(i) = biases(i) - learning_rate * layer_errors(i);
                }
            } else {
                apply_deriv(out_layer_outputs);
                
                // Calculate error derivative
                mse_deriv(out_layer_outputs, actual, final_error);
                
                // Apply activation derivative
                for(int i = 0; i < final_error.size; i++) {
                    final_error(i) *= out_layer_outputs(i);
                }
                
                hidden_updates = outer_product(final_error, in_layer_outputs);
                for(int r = 0; r < in_weights.row; r++) {
                    for(int c = 0; c < in_weights.col; c++) {
                        double update = learning_rate * hidden_updates(r, c);
                        sanitize(update);
                        in_weights(r, c) -= update;
                    }
                }
                out_bias = out_bias - learning_rate * final_error;
            }
        };
        
        // Training function that processes the entire dataset for multiple epochs
        inline void train(vecs<double> &inputs, vecs<double> &targets, int num_epochs = -1) {
            if (num_epochs < 0) {
                num_epochs = epochs; // Use the default number of epochs
            }
            
            int num_samples = inputs.num_of_vecs();
            
            if (inputs.num_of_vecs() != targets.num_of_vecs()) {
                std::cout << "Error: Number of inputs and targets must match" << std::endl;
                return;
            }
            
            double reg_lambda = 0.0001;
            double weight_decay = 1.0 - learning_rate * reg_lambda;
            
            for (int epoch = 0; epoch < num_epochs; epoch++) {
                double total_error = 0.0;
                
                for (int i = 0; i < num_samples; i++) {
                    // Forward propagation reuses pre-allocated memory
                    forward_prop(inputs(i));
                    total_error += mse(out_layer_outputs, targets(i));
                    
                    // Back propagation reuses pre-allocated memory
                    back_prop(targets(i));
                    
                    // Apply weight decay (L2 regularization)
                    if (hidden_layers > 0) {
                        // Use direct multiplication for efficiency
                        for(int r = 0; r < in_weights.row; r++) {
                            for(int c = 0; c < in_weights.col; c++) {
                                in_weights(r, c) *= weight_decay;
                            }
                        }
                        
                        for (int layer = 0; layer < hidden_layers-1; layer++) {
                            for(int r = 0; r < weights(layer).row; r++) {
                                for(int c = 0; c < weights(layer).col; c++) {
                                    weights(layer)(r, c) *= weight_decay;
                                }
                            }
                        }
                        
                        for(int r = 0; r < out_weights.row; r++) {
                            for(int c = 0; c < out_weights.col; c++) {
                                out_weights(r, c) *= weight_decay;
                            }
                        }
                    }
                }
            }
        }
        
};

class activation_functions{
    private:
        // Activation functions
        static double relu(double x){return x>0? x:0;};
        static double relu_deriv(double x){return x>0? 1:0;};
        static double leaky_relu(double x){return x>0? x:0.01*x;};
        static double leaky_relu_deriv(double x){return x>0? 1:0.01;};
        static double sigmoid(double x){return 1/(1+exp(-x));};
        static double sigmoid_deriv(double x){return sigmoid(x)*(1-sigmoid(x));};
        static double tanh(double x){return (exp(x)-exp(-x))/(exp(x)+exp(-x));};
        static double tanh_deriv(double x){return 1-tanh(x)*tanh(x);};
        double (*act_funcs[4])(double)={relu, leaky_relu, sigmoid, tanh};
        double (*act_derivs[4])(double)={relu_deriv, leaky_relu_deriv, sigmoid_deriv, tanh_deriv};
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
        ~activation_functions(){
            delete[] act_funcs;
            delete[] act_derivs;
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
            if(softmax) {
                apply_softmax(outputs);
            }
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
        
        // Get gradients (useful for debugging)
        inline vec<double>& get_gradients()  { return gradients; }
};

class NeuralNetwork {
private:
    std::vector<nn_linear_double> layers;
    double learning_rate;
    int epochs;
    bool use_softmax;
    
    // Helper function to compute MSE loss
    inline double mse(vec<double>& pred, vec<double>& actual) {
        vec<double> error = pred - actual;
        return (error * error) / error.size;
    }
    
    // Helper function to compute cross entropy loss (for classification)
    inline double cross_entropy(vec<double>& pred, vec<double>& actual) {
        double loss = 0.0;
        for(int i = 0; i < pred.size; i++) {
            if(actual(i) > 0) {  // Only compute for true labels
                loss -= actual(i) * std::log(std::max(pred(i), 1e-15));
            }
        }
        return loss;
    }

public:
    // Constructor with layer sizes and training parameters
    NeuralNetwork(const std::vector<int>& layer_sizes, 
                 double lr = 0.01, 
                 int num_epochs = 100, 
                 bool use_softmax_output = true) 
        : learning_rate(lr), epochs(num_epochs), use_softmax(use_softmax_output) {
        
        // Create layers
        for(size_t i = 0; i < layer_sizes.size() - 1; i++) {
            bool is_last_layer = (i == layer_sizes.size() - 2);
            std::string activation = is_last_layer ? "softmax" : "relu";
            bool layer_softmax = is_last_layer && use_softmax;
            
            layers.emplace_back(
                layer_sizes[i],     // input size
                layer_sizes[i + 1], // output size
                activation,         // activation function
                layer_softmax      // use softmax on last layer
            );
        }
    }
    
    // Forward pass through all layers
    vec<double> forward(vec<double>& input) {
        vec<double> current = input;
        for(auto& layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }
    
    // Backward pass through all layers
    void backward(vec<double>& target) {
        // Start with the last layer
        vec<double> grad;
        if(use_softmax) {
            // For softmax output, gradient is (pred - target)
            grad = layers.back().get_last_output() - target;
        } else {
            // For non-softmax output, use MSE gradient
            grad = 2.0 * (layers.back().get_last_output() - target) / target.size;
        }
        
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
            for(int idx : indices) {
                // Forward pass
                vec<double> output = forward(inputs(idx));
                // Compute loss
                if(use_softmax) {
                    total_loss += cross_entropy(output, targets(idx));
                } else {
                    total_loss += mse(output, targets(idx));
                }
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