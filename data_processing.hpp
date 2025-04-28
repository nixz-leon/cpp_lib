#include "Lin_alg.hpp"


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

template <typename T>
vecs<T> classification_data(vec<T> data){
    // Convert a vector of class labels to one-hot encoded matrix
    // Each row represents one data point, with a 1 in the column corresponding to its class
    int num_classes = 0;
    
    // Find the maximum class value to determine number of classes
    for (int i = 0; i < data.size; i++) {
        if (data(i) > num_classes) {
            num_classes = static_cast<int>(data(i));
        }
    }
    num_classes++; // Add 1 since classes are zero-indexed
    
    // Create one-hot encoded matrix
    vecs<T> one_hot(data.size, num_classes);
    
    // Fill the matrix
    for (int i = 0; i < data.size; i++) {
        int class_idx = static_cast<int>(data(i));
        for (int j = 0; j < num_classes; j++) {
            one_hot(i)(j) = (j == class_idx) ? 1 : 0;
        }
    }
    
    return one_hot;
}


template <typename T>
void switch_major_minor(vecs<T> &data){
    // Switch rows and columns in a vecs container in-place
    // Transforms data from [num_features, num_samples] to [num_samples, num_features]
    int num_rows = data.num_of_vecs();
    int num_cols = data.size();
    
    // Create a temporary copy of the original data
    vecs<T> temp = data;
    
    // Resize the original data to have swapped dimensions
    data = vecs<T>(num_cols, num_rows);
    
    // Fill with transposed data
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            data(j)(i) = temp(i)(j);
        }
    }
}