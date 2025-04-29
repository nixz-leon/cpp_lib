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
inline void sanitize_mat(matrix<float> &x){
    for(int i = 0; i < x.row; i++){
        for(int j = 0; j < x.col; j++){
            sanitize(x(i,j));
        }
    }
}

/*
class activation_functions{
    private:
        // Activation functions
        static float relu(float &x){return x>0? x:0;};
        static float relu_deriv(float &x){return x>0? 1:0;};
        static float leaky_relu(float &x){return x>0? x:0.01f*x;};
        static float leaky_relu_deriv(float &x){return x>0? 1:0.01f;};
        static float sigmoid(float &x){return 1/(1+exp(-x));};
        static float sigmoid_deriv(float &x){return sigmoid(x)*(1-sigmoid(x));};
        static float softmax(float &x){return exp(x)/sum(exp(x));};
        static float softmax_deriv(float &x){return softmax(x)*(1-softmax(x));};
        float (*act_funcs[4])(float&)={relu, leaky_relu, sigmoid, softmax};
        float (*act_derivs[4])(float&)={relu_deriv, leaky_relu_deriv, sigmoid_deriv, softmax_deriv};
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
        inline matrix<float> operator()(matrix<float> &m, int call){
            matrix<float> result(m.row, m.col);
            for(int i = 0; i < m.row; i++){
                for(int j = 0; j < m.col; j++){
                    result(i,j) = act_funcs[call](m(i,j));
                }
            }
            return result;
        }
        inline matrix<float> deriv(matrix<float> &m, int call){
            matrix<float> result(m.row, m.col);
            for(int i = 0; i < m.row; i++){
                for(int j = 0; j < m.col; j++){
                    result(i,j) = act_derivs[call](m(i,j));
                }
            }
            return result;
        }
        inline float(*get_act_func(int call))(float&){return act_funcs[call];};
        inline float(*get_act_deriv(int call))(float&){return act_derivs[call];};
};

class loss_function{
    private:
        static float mse(const vec<float> &pred, const vec<float> &actual){
            vec<float> error = pred - actual;
            return (error * error);
        };
        static vec<float> mse_deriv(const vec<float> &pred, const vec<float> &actual){
            vec<float> error = 2*(pred - actual);
            return error;
        }
        static float mae(const vec<float> &pred, const vec<float> &actual){
            vec<float> e = pred - actual;
            float loss = 0.0f;
            for(int i =0; i < e.size; i++){
                loss += std::abs(e(i));
            }
            return loss/e.size;
        }
        static vec<float> mae_deriv(const vec<float> &pred, const vec<float> &actual){
            vec<float> e = pred - actual;
            vec<float> result(e.size);
            for(int i = 0; i < e.size; i++){
                result(i) = (e(i) > 0) ? 1 : -1;
            }
            return result;
        }
        static float cross_entropy(const vec<float> &pred, const vec<float> &actual){
            float loss = 0.0f;
            for(int i = 0; i < pred.size; i++){
                // Add small epsilon to avoid log(0)
                float p = std::max(std::min(pred(i), 1.0f - 1e-15f), 1e-15f);
                loss -= actual(i) * std::log(p) + (1.0f - actual(i)) * std::log(1.0f - p);
            }
            return loss;
        }
        static vec<float> cross_entropy_deriv(const vec<float> &pred, const vec<float> &actual){
            vec<float> error(pred.size);
            for(int i = 0; i < pred.size; i++){
                // Add small epsilon to avoid division by zero
                float p = std::max(std::min(pred(i), 1.0f - 1e-15f), 1e-15f);
                error(i) = (p - actual(i)) / (p * (1.0f - p));
            }
            return error;
        }
        //softmax
        static float softmax(const vec<float> &pred, const vec<float> &actual){
            float loss = 0.0f;
            for(int i = 0; i < pred.size; i++){
                loss -= actual(i) * std::log(std::max(pred(i), 1e-15f));
            }
            return loss;
        }
        static vec<float> softmax_deriv(const vec<float> &pred, const vec<float> &actual){
            vec<float> error(pred.size);
            for(int i = 0; i < pred.size; i++){
                error(i) = pred(i) - actual(i);
            }
            return error;
        };
        static float hinge(const vec<float> &pred, const vec<float> &actual){
            float loss = 0.0f;
            for(int i = 0; i < pred.size; i++){
                loss += std::max(0.0f, 1.0f - actual(i) * pred(i));
            }
            return loss/pred.size;
        };
        static vec<float> hinge_deriv(const vec<float> &pred, const vec<float> &actual){
            vec<float> error(pred.size);
            for(int i = 0; i < pred.size; i++){
                error(i) = (actual(i) * pred(i) < 1) ? -actual(i) : 0;
            }
            return error;
        };
        float (*loss_funcs[5])(const vec<float>&, const vec<float>&) = {mse, mae, cross_entropy, softmax, hinge};
        vec<float> (*loss_deriv[5])(const vec<float>&, const vec<float>&) = {mse_deriv, mae_deriv, cross_entropy_deriv, softmax_deriv, hinge_deriv};
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
        inline vec<float> operator()(matrix<float> &pred, matrix<float> &actual){
            vec<float> result(pred.row);
            for(int i = 0; i < pred.row; i++){
                result(i) = loss_funcs[call](pred.get_row(i), actual.get_row(i));
            }
            return result;
        };
        inline matrix<float> deriv(matrix<float> &pred, matrix<float> &actual){
            matrix<float> result(pred.row, pred.col);
            for(int i = 0; i < pred.row; i++){
                result.set_row(i, loss_deriv[call](pred.get_row(i), actual.get_row(i)));
            }
            return result;
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
*/

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





class loss_func_deriv{
    private:
        static vec<float> mse_deriv(vec<float> &pred, vec<float> &actual){
            vec<float> error = 2*(pred - actual);
            return error;
        }
        static vec<float> cross_entropy_deriv(vec<float> &pred, vec<float> &actual){
            vec<float> deriv(pred.size);
            deriv = pred - actual;
            for(int i = 0; i < pred.size; i++){
                deriv(i) = deriv(i) / (pred(i) * (1 - pred(i)) + 1e-15f);
            }
            return deriv;
        }
        vec<float> (*loss_derivs[2])(vec<float>&, vec<float>&)={mse_deriv, cross_entropy_deriv};
        int call = 0;
        public:
            loss_func_deriv(std::string name){
                if(name == "mse"){
                    call = 0;
                }else if(name == "cross_entropy"){
                    call = 1;
                }
            }
            vec<float> operator()(vec<float> &pred, vec<float> &actual){
                return loss_derivs[call](pred, actual);
            }
            vec<float> deriv(vec<float> &pred, vec<float> &actual){
                return loss_derivs[call](pred, actual);
            }
            std::string get_name(){
                switch(call){
                    case 0: return "mse";
                    case 1: return "cross_entropy";
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
    act_func_vec fn;
    loss_func_deriv loss;  // Use the loss_function class
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
        if (is_last && (loss.get_name() == "cross_entropy")) {
            calls.push_back(2);  
        } else if(activation == "leaky_relu") {
            calls.push_back(1);
        } else if(activation == "sigmoid") {
            calls.push_back(2);
        } else {
            calls.push_back(0);  // Default to ReLU
        }
        
        // Update dimensions for next layer
        output_size = num_nodes;
        layer_count++;
        std::cout << "Debug: Updated output size: " << output_size << std::endl;
        std::cout << "Debug: Updated layer count: " << layer_count << std::endl;
    }
    //I need to write a version of foward such that, it supports batch sizes, I need to figure out on whether I want to store preactivation
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
        return outputs.back();
    }
    // Backward pass through all layers
    void backward(vec<float>& target) {
        if (target.size != outputs.back().size) {
            std::cerr << "Error: Target size mismatch. Expected " << outputs.back().size 
                      << ", got " << target.size << std::endl;
            return;
        }
        
        // Calculate loss derivative
        vec<float> loss_deriv = loss.deriv(outputs.back(), target);
        vec<float> act_deriv = fn.deriv(outputs[layer_count], calls.back());
        vec<float> delta = element_mult(loss_deriv, act_deriv);
        // Calculate partial derivative matrix
        matrix<float> partial = outer_product(delta, outputs[layer_count-1]);
        
        // Update weights and biases for the last layer
        weights[layer_count-1] = weights[layer_count-1] - learning_rate*(partial);
        biases[layer_count-1] = biases[layer_count-1] - learning_rate*(delta);
        
        // Backpropagate through remaining layers
        for(int i = layer_count-2; i >= 0; i--) {
            // Calculate new delta with proper size checks
            vec<float> weight_delta = transpose(weights[i+1])*delta;
            vec<float> act_deriv_i = fn.deriv(preactivation[i], calls[i]);
            delta = element_mult(weight_delta, act_deriv_i);
            // Calculate partial derivative matrix
            partial = outer_product(delta, outputs[i]);
            
            // Update weights and biases
            weights[i] = weights[i] - learning_rate*(partial);
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
            for(int idx = 0; idx < num_samples; idx++) {
                forward(inputs(idx));
                backward(targets(idx));
                for(int i = 0; i < layer_count; i++) {
                    weights[i] = weights[i] * (1.0f - learning_rate * reg_lambda);
                }
            }
        }
    }

    //start the template for adam.
    void backward_adam(vec<float>& target, std::vector<vec<float>>& m_biases, std::vector<matrix<float>>& m_weights, 
                       std::vector<vec<float>>& v_biases, std::vector<matrix<float>>& v_weights, 
                       int t, float beta1, float beta2, float epsilon) {
        
        // Calculate initial delta for output layer
        vec<float> delta = element_mult(loss.deriv(outputs.back(), target), fn.deriv(preactivation.back(), calls.back()));
        
        // Update for the output layer
        matrix<float> weight_grad = outer_product(delta, outputs[layer_count-1]);
        
        // Update moment estimates for the last layer
        m_weights[layer_count-1] = beta1 * m_weights[layer_count-1] + (1.0f - beta1) * weight_grad;
        m_biases[layer_count-1] = beta1 * m_biases[layer_count-1] + (1.0f - beta1) * delta;
        
        v_weights[layer_count-1] = beta2 * v_weights[layer_count-1] + (1.0f - beta2) * hadamard(weight_grad, weight_grad);
        v_biases[layer_count-1] = beta2 * v_biases[layer_count-1] + (1.0f - beta2) * element_mult(delta, delta);
        
        // Bias correction
        matrix<float> m_weights_corrected = m_weights[layer_count-1] * (1/(1.0f - std::pow(beta1, t)));
        vec<float> m_biases_corrected = m_biases[layer_count-1] * (1/(1.0f - std::pow(beta1, t)));
        
        matrix<float> v_weights_corrected = v_weights[layer_count-1] * (1/(1.0f - std::pow(beta2, t)));
        vec<float> v_biases_corrected = v_biases[layer_count-1] * (1/(1.0f - std::pow(beta2, t)));
        
        // Apply Adam update
        for(int i = 0; i < weights[layer_count-1].row; i++) {
            for(int j = 0; j < weights[layer_count-1].col; j++) {
                weights[layer_count-1](i,j) -= learning_rate * m_weights_corrected(i,j) / (std::sqrt(v_weights_corrected(i,j)) + epsilon);
            }
            biases[layer_count-1](i) -= learning_rate * m_biases_corrected(i) / (std::sqrt(v_biases_corrected(i)) + epsilon);
        }
        
        // Backpropagate through remaining layers
        for(int i = layer_count-2; i >= 0; i--) {
            // Calculate new delta
            vec<float> weight_delta = transpose(weights[i+1]) * delta;
            vec<float> act_deriv_i = fn.deriv(preactivation[i], calls[i]);
            delta = element_mult(weight_delta, act_deriv_i);
            
            // Calculate gradients
            matrix<float> weight_grad = outer_product(delta, outputs[i]);
            
            // Update moment estimates
            m_weights[i] = beta1 * m_weights[i] + (1.0f - beta1) * weight_grad;
            m_biases[i] = beta1 * m_biases[i] + (1.0f - beta1) * delta;
            
            v_weights[i] = beta2 * v_weights[i] + (1.0f - beta2) * hadamard(weight_grad, weight_grad);
            v_biases[i] = beta2 * v_biases[i] + (1.0f - beta2) * element_mult(delta, delta);
            
            // Bias correction
            matrix<float> m_weights_corrected = m_weights[i] * (1/(1.0f - std::pow(beta1, t)));
            vec<float> m_biases_corrected = m_biases[i] * (1/(1.0f - std::pow(beta1, t)));
            
            matrix<float> v_weights_corrected = v_weights[i] * (1/(1.0f - std::pow(beta2, t)));
            vec<float> v_biases_corrected = v_biases[i] * (1/(1.0f - std::pow(beta2, t)));
            
            // Apply Adam update
            for(int j = 0; j < weights[i].row; j++) {
                for(int k = 0; k < weights[i].col; k++) {
                    weights[i](j,k) -= learning_rate * m_weights_corrected(j,k) / (std::sqrt(v_weights_corrected(j,k)) + epsilon);
                }
                biases[i](j) -= learning_rate * m_biases_corrected(j) / (std::sqrt(v_biases_corrected(j)) + epsilon);
            }
        }
    }

    void train_adam(vecs<float>& inputs, vecs<float>& targets, float beta1 = 0.9f, float beta2 = 0.999f, int num_epochs = -1) {
        if(num_epochs < 0) num_epochs = epochs;
        if(inputs.num_of_vecs() != targets.num_of_vecs()) {
            std::cerr << "Error: Number of inputs and targets must match" << std::endl;
            return;
        }
        
        int num_samples = inputs.num_of_vecs();
        const float epsilon = 1e-8f;
        
        // Initialize momentum and velocity vectors for weights and biases
        std::vector<matrix<float>> m_weights(layer_count);
        std::vector<vec<float>> m_biases(layer_count);
        std::vector<matrix<float>> v_weights(layer_count);
        std::vector<vec<float>> v_biases(layer_count);
        
        // Initialize moment estimates to zeros
        for(int i = 0; i < layer_count; i++) {
            m_weights[i] = matrix<float>(weights[i].row, weights[i].col);
            m_biases[i] = vec<float>(biases[i].size);
            v_weights[i] = matrix<float>(weights[i].row, weights[i].col);
            v_biases[i] = vec<float>(biases[i].size);
        }
        
        // Training loop
        for(int epoch = 0; epoch < num_epochs; epoch++) {
            //float total_loss = 0.0f;
            
            for(int idx = 0; idx < num_samples; idx++) {
                // Forward pass
                forward(inputs(idx));
                
                // Compute loss if needed for reporting
                // total_loss += loss(outputs.back(), targets(idx));
                
                // Apply Adam optimization
                int t = epoch * num_samples + idx + 1; // timestep (1-indexed for proper bias correction)
                backward_adam(targets(idx), m_biases, m_weights, v_biases, v_weights, t, beta1, beta2, epsilon);
            }
            
            // Optional: Print progress every few epochs
            // if((epoch + 1) % 100 == 0) {
            //     std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
            //               << ", Loss: " << (total_loss / num_samples) << std::endl;
            // }
        }
    }

    void evaluate_model(vecs<float>& inputs, vecs<float>& targets){
        if(inputs.num_of_vecs() != targets.num_of_vecs()) {
            std::cerr << "Error: Number of inputs and targets must match" << std::endl;
            return;
        }
        
        int num_samples = inputs.num_of_vecs();
        int num_classes = targets(0).size;
        
        // Vectors to store actual class indices and predicted class indices
        vec<float> actual_classes(num_samples);
        vec<float> predicted_classes(num_samples);
        
        // For each sample
        for(int i = 0; i < num_samples; i++) {
            // Forward pass
            vec<float> prediction = forward(inputs(i));
            
            // Find the actual class (index with 1 in the one-hot encoded target)
            int actual_class = 0;
            for(int j = 0; j < targets(i).size; j++) {
                if(targets(i)(j) > 0.5) {
                    actual_class = j;
                    break;
                }
            }
            
            // Find the predicted class (index with highest probability)
            int predicted_class = 0;
            float max_prob = prediction(0);
            for(int j = 1; j < prediction.size; j++) {
                if(prediction(j) > max_prob) {
                    max_prob = prediction(j);
                    predicted_class = j;
                }
            }
            
            // Store the class indices
            actual_classes(i) = actual_class;
            predicted_classes(i) = predicted_class;
        }
        
        // Create confusion matrix and display metrics
        sm::ConfusionMatrix<float> confusion_matrix(actual_classes, predicted_classes);
        
        // Display metrics
        std::cout << "\nNeural Network Evaluation Results:" << std::endl;
        std::cout << "--------------------------------" << std::endl;
        std::cout << "Accuracy:  " << confusion_matrix.get_accuracy() * 100 << "%" << std::endl;
        std::cout << "Precision: " << confusion_matrix.get_precision() * 100 << "%" << std::endl;
        std::cout << "Recall:    " << confusion_matrix.get_recall() * 100 << "%" << std::endl;
        std::cout << "F1 Score:  " << confusion_matrix.get_f1_score() * 100 << "%" << std::endl;
    }
    
};


/*
class BatchNeuralNetwork {
private:
    std::vector<matrix<float>> outputs;
    std::vector<matrix<float>> preactivation;
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
    loss_function loss;
    int batch_size;

    inline void init_mat(matrix<float> &mat, float scale) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        for(int i = 0; i < mat.row; i++) {
            for(int j = 0; j < mat.col; j++) {
                mat(i,j) = dist(gen);
            }
        }
    }

    void init_vec(vec<float> &vec, float scale) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        for(int i = 0; i < vec.size; i++) {
            vec(i) = dist(gen);
        }
    }

    inline void apply_softmax(matrix<float> &m, bool row_major = false) {
        if (row_major) {
            // Row-wise softmax
            for(int i = 0; i < m.row; i++) {
                float max_val = m(i,0);
                for(int j = 1; j < m.col; j++) {
                    if(m(i,j) > max_val) max_val = m(i,j);
                }
                
                float sum = 0.0f;
                for(int j = 0; j < m.col; j++) {
                    m(i,j) = exp(m(i,j) - max_val);
                    sum += m(i,j);
                }
                
                for(int j = 0; j < m.col; j++) {
                    m(i,j) /= sum;
                    sanitize(m(i,j));
                }
            }
        } else {
            // Column-wise softmax
            for(int j = 0; j < m.col; j++) {
                float max_val = m(0,j);
                for(int i = 1; i < m.row; i++) {
                    if(m(i,j) > max_val) max_val = m(i,j);
                }
                
                float sum = 0.0f;
                for(int i = 0; i < m.row; i++) {
                    m(i,j) = exp(m(i,j) - max_val);
                    sum += m(i,j);
                }
                
                for(int i = 0; i < m.row; i++) {
                    m(i,j) /= sum;
                    sanitize(m(i,j));
                }
            }
        }
    }
    void init_outputs_preactivation_mats(int size){
        outputs.clear();
        preactivation.clear();
        outputs.push_back(matrix<float>(size, input_size));
        preactivation.push_back(matrix<float>(size, input_size));
        for(int i = 0; i < layer_count; i++){
            outputs.push_back(matrix<float>(size, weights[i].row));
            preactivation.push_back(matrix<float>(size, weights[i].row));
        }
    }
    //this function partitions the inputs into batches, sinces matricies assume all the matricies have the same size, samples.num_of_vecs() % batch_size != 0, will return a matrix of the last batch
    matrix<float> partition_inputs(vecs<float> &inputs, int batch_size, matricies<float> &batches){
        int num_samples = inputs.num_of_vecs();
        int num_full_batches = num_samples / batch_size;
        int last_batch_size = num_samples % batch_size;
        
        // Create a new matricies object with the correct number of batches

        matricies<float> new_batches(num_full_batches, batch_size, inputs.size());
        
        // Process full batches
        for(int i = 0; i < num_full_batches; i++) {
            int start_idx = i * batch_size;
            int end_idx = start_idx + batch_size - 1;  // end_idx is inclusive
            vecs<float> batch_vecs = inputs.subset(start_idx, end_idx);
            matrix<float> batch_mat(batch_vecs, true);
            new_batches(i) = batch_mat;
        }
        
        // Handle the last partial batch if it exists
        if(last_batch_size > 0) {
            int start_idx = num_full_batches * batch_size;
            int end_idx = num_samples - 1;  // end_idx is inclusive
            vecs<float> last_batch_vecs = inputs.subset(start_idx, end_idx);
            matrix<float> last_batch_mat(last_batch_vecs, true);
            batches = new_batches;
            return last_batch_mat;
        }
        
        // If no partial batch, return empty matrix
        batches = new_batches;
        return matrix<float>(0, 0);
    }
public:
    BatchNeuralNetwork(int num_input, float lr = 0.01f, int num_epochs = 100, 
                      std::string loss_func = "mse", int batch = 32) 
        : input_size(num_input), learning_rate(lr), epochs(num_epochs), 
          loss(loss_func), batch_size(batch) {
        use_softmax = (loss_func == "softmax" || loss_func == "cross_entropy");
    }

    void add_layer(int num_nodes, std::string activation = "leaky_relu", bool is_last = true) {
        if (num_nodes <= 0) {
            std::cerr << "Error: Number of nodes must be positive" << std::endl;
            return;
        }

        int input_dim = (layer_count == 0) ? input_size : output_size;
        
        matrix<float> tempm(num_nodes, input_dim);
        float weight_scale = std::sqrt(2.0f / input_dim);
        init_mat(tempm, weight_scale);
        weights.push_back(tempm);
        
        vec<float> tempv(num_nodes);
        init_vec(tempv, 0.01f);
        biases.push_back(tempv);
        
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
        
        output_size = num_nodes;
        layer_count++;
    }
            
    matrix<float> forward(matrix<float>& input) {
        if (input.row != input_size) {
            std::cerr << "Error: Input matrix rows must match input_size. Expected " 
                      << input_size << ", got " << input.row << std::endl;
            return matrix<float>(0, 0);
        }
        outputs[0] = input;
        for(int i = 0; i < layer_count; i++) {
            preactivation[i] = weights[i] * outputs[i];
            preactivation[i].add_per_col(biases[i]);
            sanitize_mat(preactivation[i]);
            outputs[i+1] = fn(preactivation[i], calls[i]);
            sanitize_mat(outputs[i+1]);
        }
        if(use_softmax) {
            apply_softmax(outputs.back(), false);
        }
        return outputs.back();
    };

    void backward(matrix<float>& target) {
        matrix<float> delta = hadamard(loss.deriv(outputs.back(), target), fn.deriv(outputs[layer_count], calls.back()));
        // Average gradients across the batch
        vec<float> avg_delta(delta.row);
        for(int j = 0; j < delta.row; j++) {
            float sum = 0.0f;
            for(int i = 0; i < delta.col; i++) {
                sum += delta(j,i);
            }
            avg_delta(j) = sum / delta.col;
        }
        
        matrix<float> partial = transpose(outputs[layer_count-1] * transpose(delta));
        weights[layer_count-1] = weights[layer_count-1] - learning_rate * partial;
        biases[layer_count-1] = biases[layer_count-1] - learning_rate * avg_delta;
        for(int i = layer_count-2; i >= 0; i--) {
            delta = hadamard(transpose(weights[i+1]) * delta, fn.deriv(preactivation[i], calls[i]));
            // Average gradients across the batch
            avg_delta = vec<float>(delta.row);
            for(int j = 0; j < delta.row; j++) {
                float sum = 0.0f;
                for(int k = 0; k < delta.col; k++) {
                    sum += delta(j,k);
                }
                avg_delta(j) = sum / delta.col;
            }

            partial = transpose(outputs[i] * transpose(delta));
            weights[i] = weights[i] - learning_rate * partial;
            biases[i] = biases[i] - learning_rate * avg_delta;
        }
    }

    void train(vecs<float> inputs, vecs<float> targets, int num_epochs = -1) {
        if(num_epochs < 0) num_epochs = epochs;
        if(inputs.num_of_vecs() != targets.num_of_vecs()) {
            std::cerr << "Error: Number of inputs and targets must match" << std::endl;
            return;
        }
        
        int num_samples = inputs.num_of_vecs();
        if(num_samples == 0) {
            std::cerr << "Error: No samples provided" << std::endl;
            return;
        }
        matricies<float> batches;
        matrix<float> last_batch = partition_inputs(inputs, batch_size, batches);
        matricies<float> targets_batches;
        matrix<float> last_target_batch = partition_inputs(targets, batch_size, targets_batches);
        
        for(int epoch = 0; epoch < num_epochs; epoch++){
            if(batches.size() != 0){
                init_outputs_preactivation_mats(batches.size());
                for(int i = 0; i < batches.size(); i++){
                    matrix<float> input = batches(i);
                    matrix<float> target = targets_batches(i);
                    forward(input);
                    backward(target);
                }
            }
            if(last_batch.row != 0){
                init_outputs_preactivation_mats(inputs.num_of_vecs());
                forward(last_batch);
                backward(last_target_batch);
            }
        }
    }

    float calculate_batch_loss(matrix<float>& predictions, matrix<float>& targets) {
        float total_loss = 0.0f;
        for(int i = 0; i < predictions.row; i++) {
            vec<float> pred_row = predictions.get_row(i);
            vec<float> target_row = targets.get_row(i);
            total_loss += loss(pred_row, target_row);
        }
        return total_loss / predictions.row;
    }
};
*/

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