#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <cstdarg>
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
    class Regression_model{
        private:
            vec<T> coefficients;
            modeltype model;
            bool multiple;
            vec<T> residuals;
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
        public:
            void fit(vec<T> x, vec<T> y, modeltype scheme = poly, int degree = 1){
                int n = x.size;
                matrix<T> X(n, degree+1);
                df2 = n-(degree+1);
                df1 = (degree);
                tdf = n-1;
                multiple = false;
                if(scheme == logarithmic){
                    for(int i = 0; i < n; i++){
                        X(i,0) = 1;
                        y(i) = log(y(i));
                        for(int j = 1; j <= degree;j++){
                            X(i,j) = X(i,j-1) * x(i);
                        }
                    }
                }else{
                    for(int i = 0; i < n; i++){
                        X(i,0) = 1;
                        for(int j = 1; j <= degree;j++){
                            X(i,j) = X(i,j-1) * x(i);
                        }
                    }
                }
                matrix<T> Xt = transpose(X);
                matrix<T> XtX = Xt*X;
                vec<T> XtY = Xt*y;
                matrix<T> XtX_inv = inverse(XtX);
                coefficients = XtX_inv * XtY;
                
                vec<T> y_hat = X * coefficients;
                if(scheme == logarithmic){
                    for(int i =0; i < n; i++){
                        y_hat(i) = pow(10,y_hat(i));
                        y(i) = pow(10,y(i));
                    }
                }
                T sum = y.sum();
                residuals = y_hat - y;
                Y_bar = sum/n;
                //I might want to take the stuff below the comment line and place it into a seperate function for a better analysis of 
                vec<T> y_v(n);
                y_v.set(Y_bar);
                SSE = residuals*residuals;
                SSR = (y_hat-y_v) * (y_hat-y_v);
                MSE = SSE/(df2);
                MSR = SSR/(df1);
                F = MSR/MSE;
                p = 1-f_distribution_cdf(F, df1, df2);
                std::cout << p << '\n';
            };
            vec<T> get_coefficients() const {
                return coefficients;
            };
            T get_F(){
                 return F;
            }


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
    void read_csv(std::string filename, vecs<T> &vectors, bool header = false) {
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
            while (std::getline(ss, value, ',')) { // Properly handling CSV format
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
        int num_rows = rows.size();
        vecs<T> temp_vecs(num_cols, num_rows);
        for (int j = 0; j < num_cols; j++) {
            for (int i = 0; i < num_rows; i++) {
                temp_vecs(j)(i) = rows[i][j];  // Ensure correct assignment
            }
        }
        vectors = temp_vecs;
    };

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
};
