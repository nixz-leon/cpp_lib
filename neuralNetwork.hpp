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
        
        // Stable leaky ReLU with alpha=0.01
        double act_func(double x) {
            return x > 0 ? x : 0.01 * x;
        }
        
        double act_deriv(double x) {
            return x > 0 ? 1.0 : 0.01;
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
        }
        
        vec<double> apply_act(vec<double> v) {
            for(int i = 0; i < v.size; i++) {
                // Apply activation and sanitize result
                v(i) = sanitize(act_func(v(i)));
            }
            return v;
        }
        
        vec<double> apply_deriv(vec<double> v) {
            for(int i = 0; i < v.size; i++) {
                // Apply derivative and sanitize result
                v(i) = sanitize(act_deriv(v(i)));
            }
            return v;
        }

    public:
        void init_mat(matrix<double> &mat, double scale){
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
        }
        nn_double(int input_dim, int output_dim, int hidden_layers_count, 
                  int nodes_per_layer, int num_epochs = 100, double lr = 0.01, 
                  bool use_softmax = true) {
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
            vec<double> error = pred - actual;
            return (error * error);
        };
        inline vec<double> mse_deriv(vec<double> &pred, vec<double> &actual){
            vec<double> error = 2*(pred - actual);
            return error;
        }


        
        inline vec<double> forward_prop(vec<double> &input) {
            // Store input for backpropagation
            in_layer_outputs = input;
            
            // Sanitize input to prevent NaN propagation
            for(int i = 0; i < in_layer_outputs.size; i++) {
                in_layer_outputs(i) = sanitize(in_layer_outputs(i));
            }
            
            if(hidden_layers > 0) {
                // First hidden layer
                vec<double> pre_activation = in_weights * in_layer_outputs + biases(0);
                // Sanitize pre-activation values
                for(int i = 0; i < pre_activation.size; i++) {
                    pre_activation(i) = sanitize(pre_activation(i));
                }
                layer_outputs(0) = apply_act(pre_activation);
                
                // Subsequent hidden layers
                for(int i = 1; i < hidden_layers; i++) {
                    pre_activation = weights(i-1) * layer_outputs(i-1) + biases(i);
                    // Sanitize pre-activation values
                    for(int j = 0; j < pre_activation.size; j++) {
                        pre_activation(j) = sanitize(pre_activation(j));
                    }
                    layer_outputs(i) = apply_act(pre_activation);
                }
                
                // Output layer
                vec<double> out_pre_activation = out_weights * layer_outputs.back() + out_bias;
                // Sanitize pre-activation values
                for(int i = 0; i < out_pre_activation.size; i++) {
                    out_pre_activation(i) = sanitize(out_pre_activation(i));
                }
                out_layer_outputs = apply_act(out_pre_activation);
            } else {
                // Single layer network
                vec<double> pre_activation = in_weights * in_layer_outputs + out_bias;
                // Sanitize pre-activation values
                for(int i = 0; i < pre_activation.size; i++) {
                    pre_activation(i) = sanitize(pre_activation(i));
                }
                out_layer_outputs = apply_act(pre_activation);
            }
            
            // Final output check
            for(int i = 0; i < out_layer_outputs.size; i++) {
                if(std::isnan(out_layer_outputs(i)) || std::isinf(out_layer_outputs(i))) {
                    std::cerr << "Warning: NaN or Inf detected in output. Replacing with 0." << std::endl;
                    out_layer_outputs(i) = 0.0;
                }
            }
            
            return out_layer_outputs;
        };
        
        inline void back_prop(vec<double> &actual) {
            vec<double> final_error;
            vecs<double> layer_errors(hidden_layers, per_layer);
            
            if(hidden_layers > 0){
                final_error = element_mult(mse_deriv(out_layer_outputs, actual), apply_deriv(out_layer_outputs));
                out_weights = out_weights - learning_rate * outer_product(final_error, layer_outputs.back());
                out_bias = out_bias - learning_rate * final_error;
                layer_errors(hidden_layers-1) = element_mult(transpose(out_weights) * final_error, apply_deriv(layer_outputs(hidden_layers-1)));
                if (hidden_layers > 1) {
                    matrix<double> hidden_updates = outer_product(layer_errors(hidden_layers-1), layer_outputs(hidden_layers-2));
                    
                    // Apply and sanitize weight updates
                    for(int i = 0; i < weights(hidden_layers-2).row; i++) {
                        for(int j = 0; j < weights(hidden_layers-2).col; j++) {
                            double update = sanitize(learning_rate * hidden_updates(i, j));
                            weights(hidden_layers-2)(i, j) -= update;
                            weights(hidden_layers-2)(i, j) = sanitize(weights(hidden_layers-2)(i, j));
                        }
                    }
                }
                
                // Update biases for the last hidden layer
                for(int i = 0; i < biases(hidden_layers-1).size; i++) {
                    double update = sanitize(learning_rate * layer_errors(hidden_layers-1)(i));
                    biases(hidden_layers-1)(i) -= update;
                    biases(hidden_layers-1)(i) = sanitize(biases(hidden_layers-1)(i));
                }
                
                // Backpropagate through remaining hidden layers
                for(int i = hidden_layers-2; i >= 0; i--) {
                    layer_errors(i) = element_mult(transpose(weights(i)) * layer_errors(i+1), apply_deriv(layer_outputs(i)));
                    if(i > 0){
                        weights(i-1) = weights(i-1) - learning_rate * outer_product(layer_errors(i), layer_outputs(i-1));
                    } else {
                        in_weights = in_weights - learning_rate * outer_product(layer_errors(0), in_layer_outputs);
                    }
                    biases(i) = biases(i) - learning_rate * layer_errors(i);
                }
            } else {
                final_error = element_mult(mse_deriv(out_layer_outputs, actual), apply_deriv(out_layer_outputs));
                in_weights = in_weights - learning_rate * outer_product(final_error, in_layer_outputs);
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
            for (int epoch = 0; epoch < num_epochs; epoch++) {
                double total_error = 0.0;
                for (int i = 0; i < num_samples; i++) {
                    vec<double> output = forward_prop(inputs(i));
                    total_error += mse(output, targets(i));
                    back_prop(targets(i));
                    if (hidden_layers > 0) {
                        in_weights = in_weights * (1.0 - learning_rate * reg_lambda);
                        for (int layer = 0; layer < hidden_layers-1; layer++) {
                            weights(layer) = weights(layer) * (1.0 - learning_rate * reg_lambda);
                        }
                        out_weights = out_weights * (1.0 - learning_rate * reg_lambda); 
                    }
                }

            }
        }
        
};