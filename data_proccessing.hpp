#ifndef DATA_PROCESSING_HPP
#define DATA_PROCESSING_HPP
#ifndef MATRICIES_HPP
#include "matricies.hpp"
#endif

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


//this function should take in a vector where each element is the class label, and return a vecs with num_of_vecs is equal to labels.size where there is a one in the index of the class label
//so each vector in vecs has a one in the index of the class label
template <typename T>
vecs<T> data_classification(vec<T> &labels){
    // Find the maximum class label to determine vector size
    T max_class = 0;
    for (int i = 0; i < labels.size; i++) {
        if (labels(i) > max_class) {
            max_class = labels(i);
        }
    }
    
    // Number of classes is max + 1 (assuming 0-indexed classes)
    int num_classes = static_cast<int>(max_class) + 1;
    
    // Create one-hot encoded vectors (one vector per label)
    vecs<T> one_hot_vectors(labels.size, num_classes);
    
    // For each label
    for (int i = 0; i < labels.size; i++) {
        // Initialize all values to 0
        for (int j = 0; j < num_classes; j++) {
            one_hot_vectors(i)(j) = 0;
        }
        
        // Set the 1 at the position indicated by the class label
        int class_idx = static_cast<int>(labels(i));
        // Ensure class index is within bounds
        if (class_idx >= 0 && class_idx < num_classes) {
            one_hot_vectors(i)(class_idx) = 1;
        }
    }
    
    return one_hot_vectors;
}

template <typename T>
vec<T> normalize_data(vec<T> &data){
    T min = data.min();
    T max = data.max();
    T range = max - min;
    vec<T> normalized_data(data.size);
    for (int i = 0; i < data.size; i++) {
        normalized_data(i) = (data(i) - min) / range;
    }
    return normalized_data;
}


template <typename T>
void get_data(std::string filename, vecs<T> &data, vecs<T> &labels, bool header = false, char delimiter = ','){
    read_csv(filename, data, header, delimiter);
    vec<T> temp = data.back();
    labels = data_classification(temp);
    data = data.subset(0, data.num_of_vecs() - 1);
    
    for(int i = 0; i < data.num_of_vecs(); i++){
        data(i) = normalize_data(data(i));
    }
    data = switch_major_minor(data);
}
#endif