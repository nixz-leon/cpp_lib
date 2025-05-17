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
        while (std::getline(ss, value, delimiter)) {
            // Trim whitespace
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);
            
            // Handle conversion with error checking
            try {
                if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                    // For floating-point types
                    row.push_back(static_cast<T>(std::stod(value)));
                } 
                else if constexpr (std::is_same_v<T, int>) {
                    // For integer type
                    row.push_back(std::stoi(value));
                }
                else {
                    // For other types, use stringstream
                    std::stringstream valueStream(value);
                    T convertedValue;
                    if (!(valueStream >> convertedValue)) {
                        throw std::runtime_error("Conversion failed");
                    }
                    row.push_back(convertedValue);
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Warning: Could not convert value '" << value << "': " << e.what() << std::endl;
                // Either skip, use default, or throw depending on requirements
                row.push_back(T{}); // Default construct (zero for numeric types)
            }
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
    
    // Check if all rows have the same number of columns
    for (size_t i = 1; i < rows.size(); i++) {
        if (rows[i].size() != static_cast<size_t>(num_cols)) {
            throw std::runtime_error("Inconsistent number of columns in CSV row " + std::to_string(i+1));
        }
    }
    
    vecs<T> temp_vecs(num_cols, num_rows);
    for (int j = 0; j < num_cols; j++) {
        for (int i = 0; i < num_rows; i++) {
            temp_vecs(j)(i) = rows[i][j];
        }
    }
    vectors = temp_vecs;
}


//this function should take in a vector where each element is the class label, and return a vecs with num_of_vecs is equal to labels.size where there is a one in the index of the class label
//so each vector in vecs has a one in the index of the class label
template <typename T>
vecs<T> data_classification(vec<T> &labels){
    // Find the maximum class label to determine vector size
    T max_class = labels.max();
    T min_class = labels.min();
    // Number of classes is max + 1 (assuming 0-indexed classes)
    int num_classes = static_cast<int>(max_class) - static_cast<int>(min_class) + 1;
    
    // Create one-hot encoded vectors (one vector per label)
    vecs<T> one_hot_vectors(labels.size, num_classes);
    
    // For each label
    for (int i = 0; i < labels.size; i++) {      
        // Set the 1 at the position indicated by the class label
        int class_idx = static_cast<int>(labels(i)) - static_cast<int>(min_class);
        if(i == 0){
            std::cout << "class_idx: " << class_idx << std::endl;
        }
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
vecs<T> switch_major_minor(vecs<T> &data){
    int num_vecs = data.num_of_vecs();
    int vec_size = data.size();
    
    // Create new vecs with swapped dimensions
    vecs<T> result(vec_size, num_vecs);
    
    // Copy data with swapped indices
    for(int i = 0; i < num_vecs; i++) {
        for(int j = 0; j < vec_size; j++) {
            result(j)(i) = data(i)(j);
        }
    }
    
    return result;
}

template <typename T>
void get_data(std::string filename, vecs<T> &data, vecs<T> &labels, bool header = false, char delimiter = ','){
    read_csv(filename, data, header, delimiter);
    vec<T> temp = data.back();
    labels = data_classification(temp);
    
    data = data.subset(0, data.num_of_vecs() - 2);
    std::cout << data.num_of_vecs() << std::endl;
    /*
    for(int i = 0; i < data.num_of_vecs(); i++){
        data(i) = normalize_data(data(i));
    */
    data = switch_major_minor(data);
}

template <typename T>
void get_data_single(std::string filename, vecs<T> &data, vecs<T> &labels, bool header = false, char delimiter = ','){
    read_csv(filename, data, header, delimiter);
    vecs<T> temp(data.num_of_vecs(), 1);
    for(int i = 0; i < data.num_of_vecs(); i++){
        temp(i)(0) = data(i)(data.size() - 1);
    }
    labels = temp;
    data = data.subset(0, data.num_of_vecs() - 1);
    for(int i = 0; i < data.num_of_vecs(); i++){
        data(i) = normalize_data(data(i));
    }
    data = switch_major_minor(data);
}

template <typename T>
void normalize_data(vecs<T> &data){
    for(int i = 0; i < data.num_of_vecs(); i++){
        data(i) = normalize_data(data(i));
    }
}

#endif