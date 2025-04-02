#pragma once
#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <cstdlib>
#include <chrono>

#include "vec.hpp"
#include "matricies.hpp"

template <typename T> 
inline void gausian_reduction(matrix<T> &a){
    int biggest_row;
    for(int i =0; i < a.cols;i++){
        biggest_row = i;
        for(int j = i+1; j < a.rows;j++){
            if(a(j,i) > a(biggest_row,i)){
                biggest_row = j;
            }
        }
        a.row_swap(biggest_row, i);
    }
    //gaussian elimination to ref
    for(int i = 0; i < a.rows;i++ ){
        for(int j = i+1; j <a.cols;j++){
            T m = (a(j,i)/a(i,i))*(-1.f);
            a.row_add(j,i,m);
        }
    }
}

template <typename T> 
inline vec<T> lin_solver(matrix<T> a,vec<T> b){
    matrix<T> tempa(a.row,a.col);
    vec<T> tempb(b.size),x(b.size);
    //Two if else statements for size matching
    //partial pivoting
    //iterate through each coloum
    tempa = a;
    tempb = b;
    int biggest_row;
    for(int i =0; i < tempa.col;i++){
        biggest_row = i;
        for(int j = i+1; j < tempa.row;j++){
            if(tempa(j,i) > tempa(biggest_row,i)){
                biggest_row = j;
            }
        }
        tempa.row_swap(biggest_row, i);
        tempb.row_swap(biggest_row, i);
    }
    //gaussian elimination to ref
    for(int i = 0; i < tempa.row;i++ ){
        for(int j = i+1; j <tempa.col;j++){
            T m = (tempa(j,i)/tempa(i,i))*(-1.f);
            tempa.row_add(j,i,m);
            tempb.row_add(j,i,m);
        }
    }
    //back sub
    T sum = 0;
    for(int i = tempa.row-1; i>-1; i--){
        for(int j = tempa.row-1; j>i-1; j--){
            if(i == tempa.row-1){
                sum += 0;
            }else{
                sum += tempa(i,j)*x(j);
            }
        }
        x(i) = (tempb(i)-sum)/(tempa(i,i));
        sum = 0;
    }
    return x;
};

template <typename T> 
inline vec<T> lin_solver_np(matrix<T> a,vec<T> b){
    matrix<T> tempa(a);
    vec<T> tempb(b);
    vec<T> x(b.size);
    //gaussian elimination to ref
    for(int i = 0; i < tempa.row;i++ ){
        for(int j = i+1; j <tempa.col;j++){
            T m = (tempa(j,i)/tempa(i,i))*(-1.f);
            tempa.row_add(j,i,m);
            tempb.row_add(j,i,m);
        }
    }
    tempa.printout();
    tempb.printout();
    x.printout();
    //back sub
    //there is something going wrong with this back sub
    T sum = 0;
    for(int i = tempa.row-1; i>-1; i--){
        for(int j = tempa.row-1; j>i-1; j--){
            if(i == tempa.row-1){
                sum += 0;
            }else{
                sum += tempa(i,j)*x(j);
            }
        }
        x(i) = (tempb(i)-sum)/(tempa(i,i));
        sum = 0;
    }
    x.printout();
    return x;
};

template <typename T>
inline void lu_decomp_np(matrix<T> a, matrix<T> &l, matrix<T> &u){
    u = a;
    l.set_diag(1.0);
    for(int i = 0; i < u.row;i++ ){
        for(int j = i+1; j <u.col;j++){
            T m = (u(j,i)/u(i,i))*(-1);
            u.row_add(j,i,m);
            l(j,i) = m*(-1);
        }
    }
}

template <typename T>
inline void lu_decomp(matrix<T> a, matrix<T> &l, matrix<T> &u, matrix<T> &p){
    //assumptions about inputs l u and p are zeros
    p.set_diag(1);
    u = a;
    if(a.rows!=a.cols){
        std::cerr << "matrix not square" << '\n';
        return;
    }
    //partial pivoting for numerical stability
    int biggest_row;
    for(int i =0; i < u.cols;i++){
        biggest_row = i;
        for(int j = i+1; j < u.rows;j++){
            if(u(j,i) > u(biggest_row,i)){
                biggest_row = j;
            }
        }
        u.row_swap(biggest_row, i);
        p.row_swap(biggest_row, i);
    }
    for(int i = 0; i < u.rows;i++ ){
        for(int j = i+1; j <u.cols;j++){
            T m = (u(j,i)/u(i,i))*(-1);
            u.row_add(j,i,m);
            l(j,i) = m*(-1);
        }
    }
}

template <typename T>
inline vec<T> lu_solver(matrix<T> A, vec<T> b) {
    matrix<T> L(A.row, A.col), U(A.row, A.col);
    lu_decomp_np(A, L, U);
    // Forward substitution (Ly = b)
    vec<T> y(b.size);
    for (int i = 0; i < A.row; i++) {
        y(i) = b(i);
        for (int j = 0; j < i; j++) {
            y(i) -= L(i, j) * y(j);
        }
        y(i) /= L(i, i);
    }
    vec<T> x(b.size);
    for (int i = A.row - 1; i >= 0; i--) {
        x(i) = y(i);
        for (int j = i + 1; j < A.col; j++) {
            x(i) -= U(i, j) * x(j);
        }
        x(i) /= U(i, i);
    }
    return x;
}

template <typename T>
inline T determinant(matrix<T> a){
    T det = 1;
    if(a.rows!=a.cols){
        std::cerr << "matrix not square" << '\n';
        return det;
    }
    matrix<T> u(a.rows,a.cols);
    u = a;
    int biggest_row =0;
    int swap_count=0;
    for(int i =0; i < u.cols;i++){
        biggest_row = i;
        for(int j = i+1; j < u.rows;j++){
            if(u(j,i) > u(biggest_row,i)){
                biggest_row = j;
            }
        }
        u.row_swap(biggest_row, i);
        if(biggest_row != i){
            swap_count++;
        }
    }
    for(int i = 0; i < u.rows;i++ ){
        for(int j = i+1; j <u.cols;j++){
            T m = (u(j,i)/u(i,i))*(-1);
            u.row_add(j,i,m);
        }
    }
    for(int i = 0; i< u.rows;i++){
        det *= u(i,i);
        if(det == 0){
            return det;
        }
    }
    if(swap_count%2 == 1){
        det *= -1;
    }
    return det;
};

template<typename T >
constexpr T norm2(vec<T> a){
    return sqrt(dot(a,a));
}

template<typename T>
inline T dot(vec<T>a, vec<T>b){
    T sum = 0;
    if(a.size != b.size){
        std::cerr << "dim do not match, dotProduct" << '\n';
        return sum;
    }
    for(int i = 0; i < a.size; i ++){
        sum +=(a(i))*(b(i));
    }
    return sum;
}

template <typename T>
inline matrix<T> alt_house_holder(vec<T> a, int c){
    matrix<T> A(a.size, a.size);
    vec<T> tempv(a.size);
    T alpha = 0;
    T beta = 0;
    tempv = a;
    for(int i =0; i < c; i++){
        tempv(i) = 0;
    }
    for(int i = c; i < A.col; i++){
        for(int j = i; j < A.row;j++){
            T entry = tempv(i) * tempv(j);
            if(i == j){
                A(i,i) = entry;
                alpha += entry;
            }else{
                A(i,j) = entry;
                A(j,i) = entry;
            }
        }
    }
    //now we have the uncorrected uuT and alpha
    tempv *= sqrt(alpha); // create correction vector
    tempv(c) *=2;
    //apply corrective matrix
    for(int i = c; i < tempv.size; i++){
        if(i == c){A(i,i)+= (alpha + tempv(i));}
        else{A(i,c)+=tempv(i);A(c,i)+=tempv(i);}
    }
    beta = 2*alpha + tempv(c);
    //A is corrected
    //Apply normalization factor of 1/alpha
    //Apply factor -2
    if(beta != 0){
        A/= beta;    
    }  
    A*=-2;
    for(int i =0; i<A.col; i++){
        A(i,i) += 1;
    }
    return A;
}

template <typename T>
inline vec<T> get_col(matrix<T> a, int c){
    vec<T> temp(a.row);
    for(int i =0; i < temp.size; i++){
        temp(i) = a(i,c);
    }
    return temp;
};

template <typename T>
inline void qr(matrix<T> a, matrix<T> &q, matrix<T> &r){
    matrix<T> h(a.row,a.row);
    h.set_diag(1);
    matrix<T> temp(a.row, a.row);
    r = a;
    for(int c = 0; c < a.col; c++){
        vec<T> tempv = get_col(r,c);
        temp = alt_house_holder(tempv,c);
        r = temp * r;
        h = h*temp;
    }
    q = h;
};

template <typename T>
matrix<T> inverse(matrix<T> A) {
    int n = A.row;
    
    matrix<T> I(n, n);
    I.set_diag(1);

    matrix<T> L(n, n), U(n, n);
    lu_decomp_np(A, L, U);

    matrix<T> invA(n, n);

    
    for (int i = 0; i < n; i++) {
        vec<T> e(n);
        e(i) = 1;
        vec<T> y(n);
        for (int j = 0; j < n; j++) {
            y(j) = e(j);
            for (int k = 0; k < j; k++) {
                y(j) -= L(j, k) * y(k);
            }
            y(j) /= L(j, j);
        }
        vec<T> x(n);
        for (int j = n - 1; j >= 0; j--) {
            x(j) = y(j);
            for (int k = j + 1; k < n; k++) {
                x(j) -= U(j, k) * x(k);
            }
            x(j) /= U(j, j);
        }
        for (int j = 0; j < n; j++) {
            invA(j, i) = x(j);
        }
    }

    return invA;
}