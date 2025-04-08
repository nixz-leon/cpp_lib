#include "Lin_alg.hpp"
#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <cstdarg>
#include <iomanip>

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
        bool softmax =true;
        double act_func(double x){return x >0 ? x : 0;};
        double act_deriv(double x){return x > 0 ? 1.0 : 0.0;};
        //double act_func(double x){return 1.0 / (1.0 + exp(-x));};
        //double act_deriv(double x){return x * (1.0 - x);};
        vec<double> apply_act(vec<double> v){
            vec<double> out(v.size);
            for(int i =0; i < v.size; i++){
                out(i) = act_func(v(i));
            }
            return out;
        }
        vec<double> apply_deriv(vec<double> v){
            vec<double> out(v.size);
            for(int i =0; i < v.size; i++){
                out(i) = act_deriv(v(i));
            }
            return out;
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
        nn_double(int input_dim, int output_dim, int hidden_layers_count,int nodes_per_layer, int num_epochs = 100, double lr = 0.01, bool softmax = true){
            input_size = input_dim;
            output_size = output_dim;
            hidden_layers = hidden_layers_count;
            per_layer = nodes_per_layer;
            epochs = num_epochs;
            learning_rate = lr;
            softmax = softmax;

            in_weights = matrix<double>(per_layer, input_size);
            out_weights = matrix<double>(output_size, per_layer); 
            weights = matricies<double>(hidden_layers-1, per_layer, per_layer);
            biases = vecs<double>(hidden_layers,per_layer);
            out_bias = vec<double>(output_dim);
            layer_outputs = vecs<double>(hidden_layers, per_layer);
            in_layer_outputs = vec<double>(input_size);
            out_layer_outputs = vec<double>(output_dim);

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


        
        inline vec<double> forward_prop(vec<double> &input){
            // Store input for backpropagation
            in_layer_outputs = input;
            
            if(hidden_layers>0){
                layer_outputs(0) = apply_act((in_weights*input)+biases(0));
                for(int i =1; i < hidden_layers; i++){
                    layer_outputs(i) = apply_act(weights(i-1)*layer_outputs(i-1) +biases(i));
                }
                out_layer_outputs = apply_act(out_weights*layer_outputs.back()+ out_bias);
            }else{
                out_layer_outputs = apply_act(in_weights*input + out_bias);
            }
            return out_layer_outputs;
        };
        
        inline void back_prop(vec<double> &actual){
            vec<double> final_error;
            vecs<double> layer_errors(hidden_layers, per_layer);
            
            if(hidden_layers > 0){
                final_error = element_mult(mse_deriv(out_layer_outputs, actual), apply_deriv(out_layer_outputs));
                out_weights = out_weights - learning_rate * outer_product(final_error, layer_outputs.back());
                out_bias = out_bias - learning_rate * final_error;
                layer_errors(hidden_layers-1) = element_mult(transpose(out_weights) * final_error, apply_deriv(layer_outputs(hidden_layers-1)));
                if (hidden_layers > 1) {
                    weights(hidden_layers-2) = weights(hidden_layers-2) - learning_rate * outer_product(layer_errors(hidden_layers-1), layer_outputs(hidden_layers-2));
                }
                biases(hidden_layers-1) = biases(hidden_layers-1) - learning_rate * layer_errors(hidden_layers-1);
                for(int i = hidden_layers-2; i >= 0; i--){
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
            
            // Training loop
            for (int epoch = 0; epoch < num_epochs; epoch++) {
                double total_error = 0.0;
                
                // Process each sample
                for (int i = 0; i < num_samples; i++) {
                    vec<double> output = forward_prop(inputs(i));
                    total_error += mse(output, targets(i));
                    back_prop(targets(i));
                }
                
                // Calculate average error for the epoch
                double avg_error = total_error / num_samples;
                
                // Print progress every 10 epochs
                /*
                if (epoch % 10 == 0) {
                    std::cout << "Epoch " << epoch << "/" << num_epochs << ", Error: " << avg_error << std::endl;
                }*/
            }
        }
};