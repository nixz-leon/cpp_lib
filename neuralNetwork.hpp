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


inline void sanitize_vec(vec<float> &v) {
    for(int i = 0; i < v.size; i++) {
        if (std::isnan(v(i)) || std::isinf(v(i))) {
            v(i) = 0.0f;
        }
        const float MAX_VALUE = 1e6f;
        if (v(i) > MAX_VALUE) v(i) = MAX_VALUE;
        if (v(i) < -MAX_VALUE) v(i) = -MAX_VALUE;
    }
}
inline void sanitize(float &x){
    if (std::isnan(x) || std::isinf(x)) {
        x = 0.0f;
    }
    const float MAX_VALUE = 1e6f;
    if (x > MAX_VALUE) x = MAX_VALUE;
    if (x < -MAX_VALUE) x = -MAX_VALUE;
}

class activation_functions{
    private:
        // Activation functions
        static float relu(float &x){return x>0? x:0;};
        static float relu_deriv(float &x){return x>0? 1:0;};
        static float leaky_relu(float &x){return x>0? x:0.01f*x;};
        static float leaky_relu_deriv(float &x){return x>0? 1:0.01f;};
        static float sigmoid(float &x){return 1/(1+exp(-x));};
        static float sigmoid_deriv(float &x){return sigmoid(x)*(1-sigmoid(x));};
        static float tanh(float &x){return (exp(x)-exp(-x))/(exp(x)+exp(-x));};
        static float tanh_deriv(float &x){return 1-tanh(x)*tanh(x);};
        float (*act_funcs[4])(float&)={relu, leaky_relu, sigmoid, tanh};
        float (*act_derivs[4])(float&)={relu_deriv, leaky_relu_deriv, sigmoid_deriv, tanh_deriv};
    public:
        inline float operator()(float x, int call){
            return act_funcs[call](x);
        };
        inline float act(float x, int call){
            return act_funcs[call](x);
        }
        inline float deriv(float x, int call){
            return act_derivs[call](x);
        };
        inline vec<float> operator()(vec<float> &v, int call){
            vec<float> result(v.size);
            for(int i = 0; i < v.size; i++){
                result(i) = act_funcs[call](v(i));
            }
            return result;
        };
        inline vec<float> act(vec<float> &v, int call){
            vec<float> result(v.size);
            for(int i = 0; i < v.size; i++){
                result(i) = act_funcs[call](v(i));
            }
            return result;
        }
        inline vec<float> deriv(vec<float> &v, int call){
            vec<float> result(v.size);
            for(int i = 0; i < v.size; i++){
                result(i) = act_derivs[call](v(i));
            }
            return result;
        };
        inline float(*get_act_func(int call))(float&){return act_funcs[call];};
        inline float(*get_act_deriv(int call))(float&){return act_derivs[call];};
};

class loss_function{
    private:
        static float mse(vec<float> &pred, vec<float> &actual){
            vec<float> error = pred - actual;
            return (error * error);
        };
        static vec<float> mse_deriv(vec<float> &pred, vec<float> &actual){
            vec<float> error = 2*(pred - actual);
            return error;
        }
        static float mae(vec<float> &pred, vec<float> &actual){
            vec<float> e = pred - actual;
            float loss = 0.0f;
            for(int i =0; i < e.size; i++){
                loss += std::abs(e(i));
            }
            return loss/e.size;
        }
        static vec<float> mae_deriv(vec<float> &pred, vec<float> &actual){
            vec<float> e = pred - actual;
            vec<float> result(e.size);
            for(int i = 0; i < e.size; i++){
                result(i) = (e(i) > 0) ? 1 : -1;
            }
            return result;
        }
        static float cross_entropy(vec<float> &pred, vec<float> &actual){
            float loss = 0.0f;
            for(int i = 0; i < pred.size; i++){
                // Add small epsilon to avoid log(0)
                float p = std::max(std::min(pred(i), 1.0f - 1e-15f), 1e-15f);
                loss -= actual(i) * std::log(p) + (1.0f - actual(i)) * std::log(1.0f - p);
            }
            return loss;
        }
        static vec<float> cross_entropy_deriv(vec<float> &pred, vec<float> &actual){
            vec<float> error(pred.size);
            for(int i = 0; i < pred.size; i++){
                // Add small epsilon to avoid division by zero
                float p = std::max(std::min(pred(i), 1.0f - 1e-15f), 1e-15f);
                error(i) = (p - actual(i)) / (p * (1.0f - p));
            }
            return error;
        }
        //softmax
        static float softmax(vec<float> &pred, vec<float> &actual){
            float loss = 0.0f;
            for(int i = 0; i < pred.size; i++){
                loss -= actual(i) * std::log(std::max(pred(i), 1e-15f));
            }
            return loss;
        }
        static vec<float> softmax_deriv(vec<float> &pred, vec<float> &actual){
            vec<float> error(pred.size);
            for(int i = 0; i < pred.size; i++){
                error(i) = pred(i) - actual(i);
            }
            return error;
        };
        static float hinge(vec<float> &pred, vec<float> &actual){
            float loss = 0.0f;
            for(int i = 0; i < pred.size; i++){
                loss += std::max(0.0f, 1.0f - actual(i) * pred(i));
            }
            return loss/pred.size;
        };
        static vec<float> hinge_deriv(vec<float> &pred, vec<float> &actual){
            vec<float> error(pred.size);
            for(int i = 0; i < pred.size; i++){
                error(i) = (actual(i) * pred(i) < 1) ? -actual(i) : 0;
            }
            return error;
        };
        float (*loss_funcs[5])(vec<float>&, vec<float>&) = {mse, mae, cross_entropy, softmax, hinge};
        vec<float> (*loss_deriv[5])(vec<float>&, vec<float>&) = {mse_deriv, mae_deriv, cross_entropy_deriv, softmax_deriv, hinge_deriv};
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
        inline float operator()(vec<float> &pred, vec<float> &actual){
            return loss_funcs[call](pred, actual);
        };
        inline vec<float> deriv(vec<float> &pred, vec<float> &actual){
            return loss_deriv[call](pred, actual);
        };
        std::string get_name() {
            switch (call) {
                case 0: return "mse";
                case 1: return "mae";
                case 2: return "cross_entropy";
                case 3: return "softmax";
                case 4: return "hinge";
                default: return "unknown";
            }
        }
};

//changing nn_linear_double to just store weights, bias, and an in to call act functions

//I need to change this to be a fully matrix based approach, which is going to a pita
//this means that outputs are going to be in the for std::vector<matrix<float>> rather than vectors
//also need to update the back propagation and seriously work on making sure the math matches
//This might be generally faster than the way I doing it currently
class NeuralNetwork {
private:
    std::vector<vec<float>> outputs;
    std::vector<vec<float>> preactivation;
    std::vector<matrix<float>> weights;
    std::vector<vec<float>> biases;
    std::vector<int> calls;
    int layer_count = 0;
    int input_size;
    int output_size;
    float learning_rate;
    int epochs;
    bool use_softmax;
    float last_error = 0;
    activation_functions fn;
    loss_function loss;  // Use the loss_function class
    inline void sanitize(float &x) {
        if (std::isnan(x) || std::isinf(x)) {
            x = 0.0f;
        }
        const float MAX_VALUE = 1e6f;
        if (x > MAX_VALUE) x = MAX_VALUE;
        if (x < -MAX_VALUE) x = -MAX_VALUE;
    }
    inline void apply_softmax(vec<float> &v) {
        float max_val = v(0);
        for(int i = 1; i < v.size; i++) {
            if(v(i) > max_val) max_val = v(i);
        }
        
        float sum = 0.0f;
        for(int i = 0; i < v.size; i++) {
            v(i) = exp(v(i) - max_val);
            sum += v(i);
        }
        
        for(int i = 0; i < v.size; i++) {
            v(i) /= sum;
            sanitize(v(i));
        }
    }
    

    inline void init_mat(matrix<float> &mat, float scale){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        for(int i =0; i < mat.row; i++){
            for(int j =0; j < mat.col; j++){
                mat(i,j) = dist(gen);
            }
        }
    };
    
    void init_vec(vec<float> &vec, float scale){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        for(int i=0; i<vec.size; i++){
            vec(i) = dist(gen);
        }
    };

public:
    // Constructor with layer sizes and training parameters
    NeuralNetwork(int nun_input, float lr = 0.01f, int num_epochs = 100, std::string loss_func = "mse") 
        : input_size(nun_input), learning_rate(lr), epochs(num_epochs), loss(loss_func) {
            use_softmax = (loss_func == "softmax" || loss_func == "cross_entropy");
        };
        
    void add_layer(int num_nodes, std::string activation = "leaky_relu", bool is_last = true) {
        if (num_nodes <= 0) {
            std::cerr << "Error: Number of nodes must be positive" << std::endl;
            return;
        }

        // For the first layer, use input_size as the input dimension
        int input_dim = (layer_count == 0) ? input_size : output_size;
        
        // Create weight matrix: (num_nodes x input_dim)
        matrix<float> tempm(num_nodes, input_dim);
        float weight_scale = std::sqrt(2.0f / input_dim);
        init_mat(tempm, weight_scale);
        weights.push_back(tempm);
        
        // Create bias vector: (num_nodes)
        vec<float> tempv(num_nodes);
        init_vec(tempv, 0.01f);
        biases.push_back(tempv);
        
        // For the first layer, we need to add the input vector to outputs and preactivation
        if (layer_count == 0) {
            outputs.push_back(vec<float>(input_size));
            preactivation.push_back(vec<float>(input_size));
        }
        
        // Add the output and preactivation vectors for this layer
        outputs.push_back(vec<float>(num_nodes));
        preactivation.push_back(vec<float>(num_nodes));
        
        // Set activation function
        if (is_last && (loss.get_name() == "softmax" || loss.get_name() == "cross_entropy")) {
            calls.push_back(2);  // Use sigmoid for last layer in classification
        } else if(activation == "leaky_relu") {
            calls.push_back(1);
        } else if(activation == "sigmoid") {
            calls.push_back(2);
        } else if(activation == "tanh") {
            calls.push_back(3);
        } else {
            calls.push_back(0);  // Default to ReLU
        }
        
        // Update dimensions for next layer
        output_size = num_nodes;
        layer_count++;
    }
    
    // Forward pass through all layers
    vec<float> forward(vec<float>& input) {
        if (input.size != input_size) {
            std::cerr << "Error: Input size mismatch. Expected " << input_size << ", got " << input.size << std::endl;
            return vec<float>(0);
        }
        outputs[0] = input;
        for(int i = 0; i < layer_count; i++) {
            preactivation[i] = weights[i] * outputs[i] + biases[i];
            sanitize_vec(preactivation[i]);
            outputs[i+1] = fn(preactivation[i], calls[i]);
            sanitize_vec(outputs[i+1]);
        }
        if(use_softmax) {
            apply_softmax(outputs.back());
        }
        //std::cout << "during forward\n";
        //outputs.back().printout();
        return outputs.back();
    }
    
    // Backward pass through all layers
    void backward(vec<float>& target) {
        vec<float> delta = element_mult(loss.deriv(outputs.back(), target), fn.deriv(outputs[layer_count], calls.back()));
        matrix<float> partial = outer_product(delta, outputs[layer_count-1]);
        //std::cout << "delta \n";
        //delta.printout();
        //std::cout << "partial \n";
        //partial.printout();
        weights[layer_count-1] = weights[layer_count-1] - learning_rate*(partial);
        biases[layer_count-1] = biases[layer_count-1] - learning_rate*(delta);

        for(int i = layer_count-2; i >= 0; i--) {
            //std::cout << "layer pass " << i << '\n';
            delta = element_mult(transpose(weights[i+1])*delta,fn.deriv(preactivation[i],calls[i]));
            partial = outer_product(delta, outputs[i]);
            //std::cout << "delta \n";
            //delta.printout();
            //std::cout << "partial \n";
            //partial.printout();
            weights[i] = weights[i] - learning_rate *(partial);
            biases[i] = biases[i] - learning_rate*(delta);
        }
    }
    
    // Train the network on a dataset
    void train(vecs<float>& inputs, vecs<float>& targets, int num_epochs = -1) {
        if(num_epochs < 0) num_epochs = epochs;
        if(inputs.num_of_vecs() != targets.num_of_vecs()) {
            std::cerr << "Error: Number of inputs and targets must match" << std::endl;
            return;
        }
        
        int num_samples = inputs.num_of_vecs();
        float reg_lambda = 0.0001f;
        
        for(int epoch = 0; epoch < num_epochs; epoch++) {
            float total_loss = 0.0f;            
            for(int idx = 0; idx < num_samples; idx++) {
                vec<float> output = forward(inputs(idx));
                //std::cout << "here 1\n";
                last_error = loss(output, targets(idx));
                total_loss += last_error;
                backward(targets(idx));
                //std::cout << "here 2\n";
                for(int i = 0; i < layer_count; i++) {
                    weights[i] = weights[i] * (1.0f - learning_rate * reg_lambda);
                }
            }
        }
    }
    
};

/*
class conv2d_layer{
    private:
        int num_filters;
        int filter_size;
        int input_x;
        int input_y;
        int stride;
        int padding = 0;
    public:
        matrix<float> discrete_colvolution(matrix<float> &input){
            matrix<float> output();
}
}
*/