#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <cstdarg>
#include <iomanip>
//#include "raylib.h"
#include "Lin_alg.hpp"



namespace sm {
    template <typename T>
    T f_distribution_pdf(T x, int df1, int df2) {
        T numerator = std::pow(df1 * x, df1) * std::pow(df2, df2);
        T denominator = std::pow((df1 * x) + df2, df1 + df2);
        T beta = std::tgamma(df1 * 0.5) * std::tgamma(df2 * 0.5) / std::tgamma((df1 + df2)*0.5);
        return sqrt(numerator / denominator) / (x*beta);
    };
    
    // Test F-statistic significance
    template <typename T>
    bool is_f_stat_significant(T f_stat, int df1, int df2, T confidence_level) {
        T p_value = 1.0 - f_distribution_pdf(f_stat, df1, df2);
        return p_value < (1.0 - confidence_level);
    };
    double betacf(double x, double a, double b, int max_iter = 100, double epsilon = 1e-12) {
        double qab = a + b;
        double qap = a + 1.0;
        double qam = a - 1.0;
        double c = 1.0;
        double d = 1.0 - (qab * x / qap);
        if (std::fabs(d) < epsilon) d = epsilon;
        d = 1.0 / d;
        double result = d;
    
        for (int m = 1; m <= max_iter; ++m) {
            int m2 = 2 * m;
            double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if (std::fabs(d) < epsilon) d = epsilon;
            c = 1.0 + aa / c;
            if (std::fabs(c) < epsilon) c = epsilon;
            d = 1.0 / d;
            result *= d * c;
    
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if (std::fabs(d) < epsilon) d = epsilon;
            c = 1.0 + aa / c;
            if (std::fabs(c) < epsilon) c = epsilon;
            d = 1.0 / d;
            double del = d * c;
            result *= del;
    
            if (std::fabs(del - 1.0) < epsilon) break;
        }
    
        return result;
    }
    
    // Compute the regularized incomplete beta function
    double betaIncomplete(double x, double a, double b) {
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
    
        double ln_beta = std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b);
        double front = std::exp(ln_beta + a * std::log(x) + b * std::log(1.0 - x));
    
        return front * betacf(x, a, b) / a;
    }
    
    // Compute the cumulative distribution function (CDF) for the F-distribution
    double f_distribution_cdf(double F, int df1, int df2) {
        if (F <= 0) return 0.0;
        double x = (df1 * F) / (df1 * F + df2);  // Convert to Beta function input
        return betaIncomplete(x, df1 / 2.0, df2 / 2.0);  // Compute CDF using the Beta function
    }

    enum modeltype{
        poly,
        logarithmic
    };
    template <typename T>
    class ConfusionMatrix {
    private:
        matrix<T> conf_matrix;
        T accuracy;
        T precision;
        T recall;
        T f1_score;
        int num_classes;
        int min_class;

    public:
        ConfusionMatrix(const vec<T>& actual, const vec<T>& predicted) {
            if (actual.size != predicted.size) {
                throw std::runtime_error("Actual and predicted vectors must be the same size");
            }

            // Find min and max class labels
            min_class = static_cast<int>(actual(0));
            int max_class = min_class;
            for (int i = 0; i < actual.size; i++) {
                min_class = std::min(min_class, static_cast<int>(actual(i)));
                min_class = std::min(min_class, static_cast<int>(predicted(i)));
                max_class = std::max(max_class, static_cast<int>(actual(i)));
                max_class = std::max(max_class, static_cast<int>(predicted(i)));
            }

            num_classes = max_class - min_class + 1;
            conf_matrix = matrix<T>(num_classes, num_classes);

            // Fill confusion matrix
            for (int i = 0; i < actual.size; i++) {
                int act = static_cast<int>(actual(i)) - min_class;
                int pred = static_cast<int>(predicted(i)) - min_class;
                conf_matrix(act, pred)++;
            }

            // Calculate metrics
            T total_samples = actual.size;
            T total_correct = 0;
            T total_precision = 0;
            T total_recall = 0;
            int valid_classes = 0;

            // Calculate per-class metrics and average them
            for (int i = 0; i < num_classes; i++) {
                T true_positive = conf_matrix(i, i);
                T false_positive = 0;
                T false_negative = 0;
                
                // Calculate row and column sums excluding true positive
                for (int j = 0; j < num_classes; j++) {
                    if (i != j) {
                        false_positive += conf_matrix(j, i);  // Sum of column minus TP
                        false_negative += conf_matrix(i, j);  // Sum of row minus TP
                    }
                }

                total_correct += true_positive;

                // Only include classes that appear in the data
                if (true_positive + false_positive + false_negative > 0) {
                    T class_precision = (true_positive + false_positive > 0) ? 
                        true_positive / (true_positive + false_positive) : 0;
                    T class_recall = (true_positive + false_negative > 0) ? 
                        true_positive / (true_positive + false_negative) : 0;
                    
                    total_precision += class_precision;
                    total_recall += class_recall;
                    valid_classes++;
                }
            }

            // Calculate final metrics
            accuracy = total_correct / total_samples;
            
            if (valid_classes > 0) {
                precision = total_precision / valid_classes;
                recall = total_recall / valid_classes;
                f1_score = (precision + recall > 0) ? 
                    2 * (precision * recall) / (precision + recall) : 0;
            } else {
                precision = 0;
                recall = 0;
                f1_score = 0;
            }
        }

        void print_metrics() const {
            std::cout << "\nModel Performance Metrics:" << std::endl;
            std::cout << "Accuracy:  " << std::fixed << std::setprecision(2) 
                     << accuracy * 100 << "%" << std::endl;
            std::cout << "Precision: " << precision * 100 << "%" << std::endl;
            std::cout << "Recall:    " << recall * 100 << "%" << std::endl;
            std::cout << "F1 Score:  " << f1_score * 100 << "%" << std::endl;
        }

        // Getters
        T get_accuracy() const { return accuracy; }
        T get_precision() const { return precision; }
        T get_recall() const { return recall; }
        T get_f1_score() const { return f1_score; }
    };

    template <typename T>
    class Regression_model{
        private:
            vec<T> coefficients;
            modeltype model;
            bool multiple;
            bool scaled;
            bool interact;
            vec<T> residuals;
            vec<T> y_hat;
            T MSE;
            T MSR;
            T SSE;
            T SSR;
            T Y_bar;
            T F;
            T p;
            int df1;
            int df2;
            int tdf;
            int deg;
        public:
            /*
                vec<T> y_v(n);
                y_v.set(Y_bar);
                SSE = residuals*residuals;
                SSR = (y_hat-y_v) * (y_hat-y_v);
                MSE = SSE/(df2);
                MSR = SSR/(df1);
                F = MSR/MSE;
                p = 1-f_distribution_cdf(F, df1, df2);
                std::cout << p << '\n';
                */
            void fit(vecs<T> vs, modeltype scheme = poly, int degree = 1, bool interaction = false){
                interact = interaction;
                deg = degree;
                int n = vs.size();
                matrix<T> X = design_matrix(vs, interaction, scheme, degree);
                df2 = n-(X.col-1);
                df1 = (X.col-1);
                tdf = n-1;
                vec<T> y(vs.back());
                matrix<T> Xt = transpose(X);
                matrix<T> XtX = Xt*X;
                vec<T> XtY = Xt*y;
                matrix<T> XtX_inv = inverse(XtX);
                coefficients = XtX_inv * XtY;

                y_hat = X * coefficients;
                if(scheme == logarithmic){
                    for(int i =0; i < n; i++){
                        y_hat(i) = pow(10,y_hat(i));
                        y(i) = pow(10,y(i));
                    }
                }
                T sum = y.sum();
                residuals = y_hat - y;
                Y_bar = sum/n;
            };
            
            vec<T> get_coefficients(){
                return coefficients;
            };
            T get_F(){
                 return F;
            };
            /*
                each predictor vector will be a represented by a row in the matrix X.
            */
            
            //with the new vecs class I think I can aggregate the two fit functions into one, and try and be more clever with it. Down side is the multiple regression will be a pia
            //I will need to check for multicollinearity. and for that I need to find some resources, since the matrix isn't square I will probably need to use the svd, and then scale the predicitors with the x- x_bar
            //I could looking into finding a g inverse via an itrative method 


            //takes in a collections of vectors, assumes the last vector is the response variable. 
            /*
            template <typename T>
            void fit(vecs<T> vs, modeltype shceme, int degree = 1){

            }
            */
            // Add this to the public section of your Regression_model class
            ConfusionMatrix<T> get_confusion_matrix(T threshold = 0.5) {
                if (residuals.size == 0) {
                    throw std::runtime_error("Model must be fitted before generating confusion matrix");
                }
                
                // Get actual and predicted values
                vec<T> actual = vec<T>(residuals.size);
                vec<T> predicted = vec<T>(y_hat);
                
                // Calculate predicted values
                for (int i = 0; i < residuals.size; i++) {
                    actual(i) = residuals(i) + predicted(i);
                }
                
                return ConfusionMatrix<T>(actual, predicted);
            };
            
            // Add these methods to the public section of the Regression_model class
            vec<T> predict(vecs<T>& X_test) {
                matrix<T> X = design_matrix(X_test, interact, model, deg);
                return X * coefficients;
            }
            
            ConfusionMatrix<T> test(vecs<T>& test_data, T threshold = 0.5) {
                if (coefficients.size == 0) {
                    throw std::runtime_error("Model must be fitted before testing");
                }
                
                // Get predictions for test data
                vec<T> y_test = test_data.back();  // Last column is target
                vec<T> y_pred = predict(test_data);
                
                // Create confusion matrix from test results
                return ConfusionMatrix<T>(y_test, y_pred);
            }
    };
    template <typename T>
    matrix<T> design_matrix(vecs<T> xs, bool interactions = true, modeltype scheme = poly, int degree = 1) {
        int p = xs.num_of_vecs()-1;  // Number of predictors
        int n = xs.size();         // Number of observations
        int m;                     // Number of columns in design matrix

        if (p == 1) {
            // Single predictor case (plus intercept)
            m = degree + 1;
            matrix<T> X(n, m);
            for (int i = 0; i < n; i++) {
                X(i,0) = 1;  // Intercept term
                for (int j = 1; j <= degree; j++) {
                    X(i,j) = pow(xs(0)(i), j);  // Polynomial terms
                }
            }
            return X;
        } else if (p > 1) {
            // Multiple predictors case
            if (interactions) {
                m = 1 + p*degree;   
                // Add interaction terms
                for (int i = 0; i < p-1; i++) {
                    for (int j = i+1; j < p; j++) {
                        m++;  // Add each interaction term
                    }
                }
                matrix<T> X(n, m);
                for (int i = 0; i < n; i++) {
                    int col = 0;
                    X(i, col++) = 1;
                    for (int d = 1; d <= degree; d++) {
                        for (int j = 0; j < p; j++) {
                            X(i, col++) = pow(xs(j)(i), d);
                        }
                    }
                    for (int j = 0; j < p; j++) {
                        for (int k = j+1; k < p; k++) {
                            X(i, col++) = xs(j)(i) * xs(k)(i);
                        }
                    }
                }
                return X;
            } 
        }
        m = (p*degree + 1); 
        matrix<T> X(n, m);
        for (int i = 0; i < n; i++) {
            X(i,0) = 1;  // Intercept
            for(int j = 1; j <= degree; j++){
                for (int k = 0; k < p; k++) {
                    X(i, (j-1)*p + k + 1) = std::pow(xs(k)(i), j);
                }
            }
        }
        return X;
    
    };
    
    template <typename T>
    void read_data_from_file(std::string filename, vec<T> &X, vec<T> &Y) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file");
        }
        std::vector<T> feature_values;
        std::vector<T> target_values;
        std::string line;
        std::getline(file, line); // Skip header
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            T concentration, hours;
            if (iss >> concentration >> hours) {
                feature_values.push_back(concentration);
                target_values.push_back(hours);
            }
        }
        file.close();
        // Convert std::vector to vec<T>
        vec<T> x(feature_values.size());
        vec<T> y(target_values.size());
        for (size_t i = 0; i < feature_values.size(); i++) {
            x(i) = feature_values[i];
            y(i) = target_values[i];
        }
        X = x;
        Y = y;
        
    };
    template <typename T>
    void read_csv(std::string filename, vecs<T> &vectors, bool header = false, char delimiter = ',') {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file");
        }
        std::string line;
        if (header) {
            std::getline(file, line); // Skip header line
        }
        std::vector<std::vector<T>> rows;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::vector<T> row;
            std::string value;
            while (std::getline(ss, value, delimiter)) { // Properly handling CSV format
                std::stringstream valueStream(value);
                T convertedValue;
                valueStream >> convertedValue;
                row.push_back(convertedValue);
            }
            if (!row.empty()) {
                rows.push_back(row);
            }
        }
        file.close();

        if (rows.empty()) {
            throw std::runtime_error("CSV file is empty or improperly formatted");
        }

        int num_cols = rows[0].size();
        std::cout << "Number of columns: " << num_cols << std::endl;
        int num_rows = rows.size();
        std::cout << "Number of rows: " << num_rows << std::endl;
        vecs<T> temp_vecs(num_cols, num_rows);
        for (int j = 0; j < num_cols; j++) {
            for (int i = 0; i < num_rows; i++) {
                temp_vecs(j)(i) = rows[i][j];  // Ensure correct assignment
            }
        }
        vectors = temp_vecs;
    };

    // Add this after the read_csv function but before the Regression_model class
    

    /*
    template <typename T>
    void graph_data(const vec<T>& x, const vec<T>& y, const vec<T>& coefficients) {
        const int screenWidth = 800;
        const int screenHeight = 600;
        
        InitWindow(screenWidth, screenHeight, "Polynomial Regression Graph");
        SetTargetFPS(60);

        // Graph scaling
        const int padding = 50;
        const float xMin = *std::min_element(&x(0), &x(0) + x.size);
        const float xMax = *std::max_element(&x(0), &x(0) + x.size);
        const float yMin = *std::min_element(&y(0), &y(0) + y.size);
        const float yMax = *std::max_element(&y(0), &y(0) + y.size);

        while (!WindowShouldClose()) {
            BeginDrawing();
            ClearBackground(RAYWHITE);

            // Draw X and Y axis
            DrawLine(padding, screenHeight - padding, screenWidth - padding, screenHeight - padding, BLACK); // X-axis
            DrawLine(padding, padding, padding, screenHeight - padding, BLACK); // Y-axis

            // Draw grid lines
            for (int i = 1; i <= 10; i++) {
                int xPos = padding + i * (screenWidth - 2 * padding) / 10;
                int yPos = screenHeight - padding - i * (screenHeight - 2 * padding) / 10;
                DrawLine(xPos, padding, xPos, screenHeight - padding, LIGHTGRAY);
                DrawLine(padding, yPos, screenWidth - padding, yPos, LIGHTGRAY);
            }

            // Draw data points
            for (int i = 0; i < x.size; i++) {
                int xPixel = padding + ((x(i) - xMin) / (xMax - xMin)) * (screenWidth - 2 * padding);
                int yPixel = screenHeight - padding - ((y(i) - yMin) / (yMax - yMin)) * (screenHeight - 2 * padding);
                DrawCircle(xPixel, yPixel, 5, RED);
            }

            // Draw regression line
            for (int i = 0; i < screenWidth - 2 * padding; i++) {
                float xVal = xMin + (i / (float)(screenWidth - 2 * padding)) * (xMax - xMin);
                float yVal = coefficients(0);
                for(int j =1; j< coefficients.size;j++){
                    yVal += coefficients(j)*pow(xVal,j);
                }

                int xPixel = padding + i;
                int yPixel = screenHeight - padding - ((yVal - yMin) / (yMax - yMin)) * (screenHeight - 2 * padding);

                if (yPixel >= padding && yPixel <= screenHeight - padding) {
                    DrawPixel(xPixel, yPixel, BLUE);
                }
            }

            // Labels
            DrawText("X-axis", screenWidth / 2, screenHeight - 30, 20, BLACK);
            DrawText("Y-axis", 10, screenHeight / 2, 20, BLACK);

            EndDrawing();
        }

        CloseWindow();
    };
    */

    // Add this after the existing code in the sm namespace

    template <typename T>
    class SVM {
    private:
        matrix<T> all_weights;
        vec<T> all_biases;
        T learning_rate;
        T lambda;
        int max_iter;
        int num_classes;
        int min_class;  // Add this to private members
        vec<T> feature_means;    // Add feature scaling
        vec<T> feature_stds;     // Add feature scaling

    public:
        SVM(T learning_rate = 0.01, T lambda = 0.01, int max_iter = 1000)
            : learning_rate(learning_rate), lambda(lambda), max_iter(max_iter) {}

        void fit(vecs<T>& data) {
            vecs<T> X = data.subset(0, data.num_of_vecs() - 1);
            vec<T> y = data.back();
            
            // Scale features
            scale_features(X);
            
            // Find number of classes and store min_class
            min_class = static_cast<int>(y(0));
            T max_class = y(0);
            for (int i = 0; i < y.size; i++) {
                min_class = std::min(min_class, static_cast<int>(y(i)));
                max_class = std::max(max_class, y(i));
            }
            num_classes = static_cast<int>(max_class - min_class + 1);
            
            int n_samples = X.size();
            int n_features = X.num_of_vecs();
            
            // Initialize weights and biases
            all_weights = matrix<T>(num_classes, n_features);
            all_biases = vec<T>(num_classes);
            all_biases.set(0);

            // Train one classifier per class
            for (int c = 0; c < num_classes; c++) {
                // Create binary labels
                vec<T> binary_labels(n_samples);
                for (int i = 0; i < n_samples; i++) {
                    binary_labels(i) = (y(i) == (c + min_class)) ? 1 : -1;
                }

                // Train binary classifier
                T prev_loss = std::numeric_limits<T>::max();
                for (int iter = 0; iter < max_iter; iter++) {
                    T loss = 0;
                    for (int i = 0; i < n_samples; i++) {
                        T linear_output = dot_product(X, i, c) + all_biases(c);
                        T hinge_loss = std::max(T(0), T(1) - binary_labels(i) * linear_output);
                        loss += hinge_loss;

                        if (hinge_loss > 0) {
                            // Update weights and bias for misclassified samples
                            for (int j = 0; j < n_features; j++) {
                                T gradient = -binary_labels(i) * X(j)(i) + 2 * lambda * all_weights(c, j);
                                all_weights(c, j) -= learning_rate * gradient;
                            }
                            all_biases(c) += learning_rate * binary_labels(i);
                        }
                    }
                    
                    // Early stopping if loss improvement is small
                    if (std::abs(loss - prev_loss) < 1e-5) break;
                    prev_loss = loss;
                }
            }
        }

        vec<T> predict(vecs<T>& X_input) {
            // Scale input features using stored means and stds
            vecs<T> X = scale_input(X_input);
            
            int n_samples = X.size();
            vec<T> predictions(n_samples);

            for (int i = 0; i < n_samples; i++) {
                T max_score = -std::numeric_limits<T>::max();
                int best_class = min_class;  // Start with minimum class value

                for (int c = 0; c < num_classes; c++) {
                    T score = dot_product(X, i, c) + all_biases(c);
                    if (score > max_score) {
                        max_score = score;
                        best_class = min_class + c;  // Adjust class label by min_class
                    }
                }
                predictions(i) = best_class;
            }

            return predictions;
        }

        ConfusionMatrix<T> test(vecs<T>& test_data) {
            vecs<T> X = test_data.subset(0, test_data.num_of_vecs() - 1);
            vec<T> y = test_data.back();
            vec<T> predictions = predict(X);
            return ConfusionMatrix<T>(y, predictions);
        }

    private:
        void scale_features(vecs<T>& X) {
            int n_features = X.num_of_vecs();
            feature_means = vec<T>(n_features);
            feature_stds = vec<T>(n_features);
            
            for (int j = 0; j < n_features; j++) {
                // Calculate mean
                T mean = 0;
                for (int i = 0; i < X.size(); i++) {
                    mean += X(j)(i);
                }
                mean /= X.size();
                feature_means(j) = mean;
                
                // Calculate std
                T std = 0;
                for (int i = 0; i < X.size(); i++) {
                    std += (X(j)(i) - mean) * (X(j)(i) - mean);
                }
                std = std::sqrt(std / X.size());
                if (std < 1e-10) std = 1;
                feature_stds(j) = std;
                
                // Scale features
                for (int i = 0; i < X.size(); i++) {
                    X(j)(i) = (X(j)(i) - mean) / std;
                }
            }
        }

        vecs<T> scale_input(vecs<T>& X) {
            vecs<T> X_scaled = X;
            for (int j = 0; j < X.num_of_vecs(); j++) {
                for (int i = 0; i < X.size(); i++) {
                    X_scaled(j)(i) = (X(j)(i) - feature_means(j)) / feature_stds(j);
                }
            }
            return X_scaled;
        }

        T dot_product(vecs<T>& X, int sample_index, int class_index) {
            T result = 0;
            for (int j = 0; j < X.num_of_vecs(); j++) {
                result += all_weights(class_index, j) * X(j)(sample_index);
            }
            return result;
        }
    };

    template <typename T>
    class KNN {
    private:
        vecs<T> X_train;  // Training features
        vec<T> y_train;   // Training labels
        int k;            // Number of neighbors

    public:
        KNN(int k_neighbors = 3) : k(k_neighbors) {}

        void fit(vecs<T>& data) {
            X_train = data.subset(0, data.num_of_vecs() - 1);
            y_train = data.back();
        }

        vec<T> predict(vecs<T>& X) {
            int n_samples = X.size();
            vec<T> predictions(n_samples);

            for (int i = 0; i < n_samples; i++) {
                // Compute distances to all training samples
                vec<T> distances(X_train.size());
                for (int j = 0; j < X_train.size(); j++) {
                    distances(j) = euclidean_distance(X, i, X_train, j);
                }

                // Find the indices of the k smallest distances
                vec<int> nearest_neighbors(k);
                for (int j = 0; j < k; j++) {
                    T min_distance = std::numeric_limits<T>::max();
                    int min_index = -1;
                    for (int l = 0; l < distances.size; l++) {
                        if (distances(l) < min_distance) {
                            min_distance = distances(l);
                            min_index = l;
                        }
                    }
                    nearest_neighbors(j) = min_index;
                    distances(min_index) = std::numeric_limits<T>::max();  // Mark as visited
                }

                // Count the labels of the k nearest neighbors
                vec<int> label_counts(y_train.max() + 1);  // Assuming labels are non-negative integers
                label_counts;  // Initialize counts to zero
                for (int j = 0; j < k; j++) {
                    int label = static_cast<int>(y_train(nearest_neighbors(j)));
                    label_counts(label)++;
                }

                // Find the label with the highest count
                int majority_label = 0;
                int max_count = label_counts(0);
                for (int j = 1; j < label_counts.size; j++) {
                    if (label_counts(j) > max_count) {
                        majority_label = j;
                        max_count = label_counts(j);
                    }
                }

                predictions(i) = majority_label;
            }

            return predictions;
        }

        ConfusionMatrix<T> test(vecs<T>& data, T threshold = 0.5) {
            vecs<T> X = data.subset(0, data.num_of_vecs() - 1);
            vec<T> predictions = predict(X);
            return ConfusionMatrix<T>(data.back() , predictions);
        }

    private:
        T euclidean_distance(vecs<T>& X1, int idx1, vecs<T>& X2, int idx2) {
            T distance = 0;
            for (int i = 0; i < X1.num_of_vecs(); i++) {
                distance += std::pow(X1(i)(idx1) - X2(i)(idx2), 2);
            }
            return std::sqrt(distance);
        }
    };
};