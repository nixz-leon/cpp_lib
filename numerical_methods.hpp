#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <cstdlib>
#include <chrono>
#include "vec.hpp"
#include "matrix.hpp"
namespace nm{

template <typename T>
//all of these are of size x.size
inline vec<T> interpolate(vec<T> x, vec<T> fx)
{
    vec<T> poly(x.size);
    vec<T> temp(x.size);
    T prod;
    int i, j, k;
    for (i = 0; i < x.size; i++)
        poly(i) = 0;
    for (i = 0; i < x.size; i++) {
        prod = 1;
        for (j = 0; j < x.size; j++) {
            temp(j) = 0;
        }
        for (j = 0; j < x.size; j++) {
            if (i == j)
                continue;
            prod *= (x(i) - x(j));
        }
        prod = fx(i) / prod;
        temp(0) = prod;
        for (j = 0; j < x.size; j++) {
            if (i == j)
                continue;
            for (k = x.size - 1; k > 0; k--) {
                temp(k) += temp(k - 1);
                temp(k - 1) *= (-x(j));
            }
        }
        for (j = 0; j < x.size; j++) {
            poly(j) += temp(j);
        }
    }
    return poly;
};

// store each iteration in a vector; and then display the vector 
template <typename T>
inline T euler(T (*f)(T, T), T h, T a, T b, T inital){
    int its = int((b-a)/h);
    //vec<T> val(its+1);
    T val;
    val = inital;
    T t = a;
    for(int i =1; i < its; i++){
        val = (val + h*f(t, val));
        t+=h;
    }
    return val;
};

template <typename T>
inline T change_h(T diff, T h, T tol){
    T temp1 = 0.9*h*std::pow(tol/diff, 1./6.0);
    T temp2 = 10*h;
    return temp1<temp2 ? temp1 : temp2;
};

//i will want to do the h calc In this function
template <typename T>
inline T rkdps(T (*f)(T,T), T &h, T y, T t ,matrix<T> &a, vec<T> &b, vec<T> &b2 ,vec<T> &c, T &diff,T tol){
    vec<T> temp(2);
    vec<T> k(7);
    k(0) = h * f(t,y);
    k(1) = h * f(t + c(1)*h, y + h*a(1,0)*k(0));
    k(2) = h * f(t + c(2)*h, y + h*a(2,0)*k(0) + h*a(2,1)*k(1));
    k(3) = h * f(t + c(3)*h, y + h*a(3,0)*k(0) + h*a(3,1)*k(1) + h*a(3,2)*k(2));
    k(4) = h * f(t + c(4)*h, y + h*a(4,0)*k(0) + h*a(4,1)*k(1) + h*a(4,2)*k(2) + h*a(4,3)*k(3));
    k(5) = h * f(t + c(5)*h, y + h*a(5,0)*k(0) + h*a(5,1)*k(1) + h*a(5,2)*k(2) + h*a(5,3)*k(3) + h*a(5,4)*k(4));
    k(6) = h * f(t + c(6)*h, y + h*a(6,0)*k(0) + h*a(6,1)*k(1) + h*a(6,2)*k(2) + h*a(6,3)*k(3) + h*a(6,4)*k(4) + a(6,5)*k(5));
    temp(0) = y +  b(0)*k(0) + b(1)*k(1) + b(2)*k(2) + b(3)*k(3) + b(4)*k(4) + b(5)*k(5);
    temp(1) = y +  b2(0)*k(0) + b2(1)*k(1) + b2(2)*k(2) + b2(3)*k(3) + b2(4)*k(4) + b2(5)*k(5) + b2(6)*k(6);
    diff = norm2(temp);
    //diff = abs(temp(1)-temp(0));
    if(diff > h*tol){
        h=change_h(diff, h, tol);
        std::cout << "hit\n"; 
    }
    return temp(0);
};



//One of the matrices or vectors is either not doing a proper deep copy, or 
template <typename T>
inline T rkdp(T (*f)(T,T), T h, T a, T b, T inital, T tol){
    matrix<T>aii(7,7);
    vec<T>bi(7),bi2(7),ci(7);
    aii(1,0) = 1./5.;
    aii(2,0) = 3./40.;         aii(2,1) = 9./40.;
    aii(3,0) = 44./45.;        aii(3,1) = -56./15.;        aii(3,2) = 32./9.;
    aii(4,0) = 19372.0/6561.0; aii(4,1) = -25360./2187.;   aii(4,2) = 64448.0/6561.0; aii(4,3) = -212./729.;
    aii(5,0) = 9017./3168.;    aii(5,1) = -355./33.;       aii(5,2) = 46732./5247.;   aii(5,3) = 49./176.;    aii(5,4) = -5103./18656.;
    aii(6,0) = 35./384.;       aii(6,1) = 0.;              aii(6,2) = 500./1113.;     aii(6,3) = 125.0/192.;  aii(6,4) = -2187.0/6784.0;  aii(6,5) = 11./84.;
    bi(0) = 35./384.;          bi(1)=0.;                   bi(2)=500./1113.;          bi(3) = 125./192.;      bi(4)=-2187.0/6784.0;       bi(5)=11./84.;       bi(6)=0.;
    bi2(0) = 5179.0/57600.0;   bi2(1)=0.0;                 bi2(2)=7571.0/16695.0;     bi2(3) = 393.0/640.0;   bi2(4)=-92097.0/339200.0;   bi2(5)=187./2100.;   bi2(6)=1./40.;
    ci(0) = 0.0;               ci(1)=0.2;                  ci(2)=0.3;                 ci(3) = 0.8;            ci(4)=8.0/9.0;              ci(5)=1.0;           ci(6)=1.0;
    T y = inital; 
    T t = a;
    T diff;
    while(t < b){
        //std::cout << h << '\n';
        y = rkdps(f, h, y, t, aii,bi,bi2,ci,diff, tol);
        t+=h;
    }
    return y;
};



template <typename T>
inline T rk4(T (*f)(T,T), T h, T a, T b, T inital){
    int its = int((b-a)/h);
    T k0,k1,k2,k3,val;
    T t = a;
    T h2 = h*0.5;
    val = inital;
    for(int i =0; i < its; i++){
        k0 =  f(t, val);
        k1 = f(t+h2, val+h2*k0);
        k2 = f(t+h2, val+h2*k1);
        k3 = f(t+h, val+h*k2);
        val = val + h*(1.0/6.0)*(k0 + k1*2.f + k2*2.f + k3);
        t += h;
    }
    return val;
};
template <typename T>
inline void rk4(T (*f)(T,T), T h, T a, T b, T inital, vec<T> &val){
    int its = int((b-a)/h);
    //vec<T> vals(its);
    T k0,k1,k2,k3;
    T t = a;
    T h2 = h*0.5;
    val(0) = inital;
    for(int i =0; i < its; i++){
        k0 =  f(t, val(i));
        k1 = f(t+h2, val(i)+h2*k0);
        k2 = f(t+h2, val(i)+h2*k1);
        k3 = f(t+h, val(i)+h*k2);
        val(i+1) = val(i) + h*(1.0/6.0)*(k0 + k1*2.f + k2*2.f + k3);
        t += h;
    }
};

template <typename T>
inline T rk4(T (*f)(T,T), T h, T a, T val){
    T k0,k1,k2,k3;
    T h2 = h*0.5f;
    k1 = f(a+h2, val+(h2*k0));
    k2 = f(a+h2, val+(h2*k1));
    k3 = f(a+h, val+(h*k2));
    return val + h*(1.0/6.0)*(k0 + k2*2.f + k3*2.f + k3);
};

template <typename T>
inline vec<T> rk4(vec<T> (*f)(vec<T>,vec<T>), T h, vec<T> a, vec<T> val){
    vec<T> k0(a.size),k1(a.size),k2(a.size),k3(a.size);
    T h2 = h*0.5;
    k0 = f(a, val);
    k1 = f(a+h2, val+(h2*k0));
    k2 = f(a+h2, val+(h2*k1));
    k3 = f(a+h, val+(h*k2));
    return val + h*(1.0/6.0)*(k0 + k2*2.f + k3*2.f + k3);
};


template <typename T>
inline vec<T> AB4(T (*f)(T,T), T h, T a, T b, T inital){
    int its = int((b-a)/h);
    vec<T> val(its+1);
    vec<T> func_vals(4);
    val(0) = inital;
    rk4(f, h, a, a+3*h, inital,val);
    T t=a;
    func_vals(0) = f(t,val(0));t+=h;
    func_vals(1) = f(t,val(1));t+=h;
    func_vals(2) = f(t,val(2));t+=h;
    func_vals(3) = f(t,val(3));
    int i = 4;
    while(t<b){
        val(i) = val(i-1) + (h)*(55.*func_vals(3) - 59.*func_vals(2) + 37.*func_vals(1) -9.*func_vals(0))/(24.);
        t+=h;
        func_vals(0) = func_vals(1);func_vals(1) = func_vals(2);func_vals(2) = func_vals(3);
        func_vals(3) = f(t,val(i));
        i++;
    }
    return val;
};

template <typename T>
inline vec<T> pece(T (*f)(T,T), T h, T a, T b, T inital){
    int its = int((b-a)/h);
    vec<T> val(its+1);
    vec<T> func_vals(4);
    val(0) = inital;
    rk4(f, h, a, a+3*h, inital,val);
    T t=a;
    func_vals(0) = f(t,val(0));t+=h;
    func_vals(1) = f(t,val(1));t+=h;
    func_vals(2) = f(t,val(2));t+=h;
    func_vals(3) = f(t,val(3));
    int i = 4;
    T temp;
    while(t<b){
        val(i) = val(i-1) + (h)*(55.*func_vals(3) - 59.*func_vals(2) + 37.*func_vals(1) -9.*func_vals(0))/(24.);
        t+=h;
        temp = f(t,val(i));
        val(i) = val(i-1) + (h)*(251.0*temp + 646.0*func_vals(3) -264.0*func_vals(2)+106.0*func_vals(1)-19.0*func_vals(0))/(720.0);
        func_vals(0) = func_vals(1);func_vals(1) = func_vals(2);func_vals(2) = func_vals(3);
        func_vals(3) = f(t,val(i));
        i++;
    }
    return val;
};

template <typename T>
inline vec<T> AB4(T (*f)(T,T), T h, T a, T b, vec<T> inital){
    int its = int((b-a)/h);
    vec<T> val(its+1);
    vec<T> func_vals(4);
    val(0)= inital(0);
    T t=a;
    func_vals(0) = f(t,inital(0));t+=h;
    func_vals(1) = f(t,inital(1));t+=h;
    func_vals(2) = f(t,inital(2));t+=h;
    func_vals(3) = f(t,inital(3));
    int i = 4;
    while(t<b){
        val(i) = val(i-1) + (h)*(55.*func_vals(3) - 59.*func_vals(2) + 37.*func_vals(1) -9.*func_vals(0))/(24.);
        t+=h;
        func_vals(0) = func_vals(1);func_vals(1) = func_vals(2);func_vals(2) = func_vals(3);
        func_vals(3) = f(t,val(i));
        i++;
    }
    return val;
};
template <typename T>
inline vec<T> pece(T (*f)(T,T), T h, T a, T b, vec<T> inital){
    int its = int((b-a)/h);
    vec<T> val(its+1);
    vec<T> func_vals(4);
    val(0)= inital(0);
    T t=a;
    func_vals(0) = f(t,inital(0));t+=h;
    func_vals(1) = f(t,inital(1));t+=h;
    func_vals(2) = f(t,inital(2));t+=h;
    func_vals(3) = f(t,inital(3));
    int i = 4;
    T temp;
    while(t<b){
        val(i) = val(i-1) + (h)*(55.*func_vals(3) - 59.*func_vals(2) + 37.*func_vals(1) -9.*func_vals(0))/(24.);
        t+=h;
        temp = f(t,val(i));
        val(i) = val(i-1) + (h)*(251.0*temp + 646.0*func_vals(3) -264.0*func_vals(2)+106.0*func_vals(1)-19.0*func_vals(0))/(720.0);
        func_vals(0) = func_vals(1);func_vals(1) = func_vals(2);func_vals(2) = func_vals(3);
        func_vals(3) = f(t,val(i));
        i++;
    }
    return val;
};

template <typename T>
inline void write_to_file(std::string filename, T b, T a, T h, vec<T> holder){
    std::ofstream file;
    file.open(filename);
    for(int i =0; i < holder.size; i++){
        file << a << ',' << holder(i) << '\n';
        a+=h;
    }
    file.close();
};

// for the finite difference code, i will basically need to generate a matrix and then use my linear solver to get the results
//to generate this matrix, the function that is being used as an input must return a vector of size 3, that corresponds:
//p(xi), q(xi), r(xi)
template <typename T>
inline vec<T> finite_differences(vec<T> (*f)(T), T h, T a, T b, T y0, T yn){
    int its = (int)(((b-a)/h)+0.5f);//0.5f for rounding this caused an issue with the code from last week
    T x = a+h;
    vec<T> b_vec(its+1);
    b_vec(its) = yn;
    b_vec(0) = y0;
    matrix<T>  A_mat(its+1,its+1);//default constructor allocates all the elements to be 0
    A_mat(0,0) = 1;A_mat(its,its) = 1;//setting the corners to 1;
    vec<T> temp(3);//this is a place holder so I don't allocate memory every iteration of the loop, and only need to do one function call;
    //T  l_elm, c, r_elm;
    T c1 = 1 / (h*h);
    T c2 = (1/h)*0.5;
    for(int i =1; i < its; i++){//bounded so it doesn't disturb the preset elements
        temp = f(x);
        b_vec(i) = temp(2);
        A_mat(i,i) = ((T)-1)*(temp(1)+ 2.0*c1);
        A_mat(i,i-1) = c1 + c2*temp(0);
        A_mat(i,i+1) = c1 - c2*temp(0);
        x+=h; 
    }
    vec<T> result(b_vec.size);
    result = lin_solver_np(A_mat, b_vec);
    return result;
};

template <typename T>
inline vec<T> rk4(vec<T> (*f)(T,vec<T>), T h, T t, vec<T> val){//single iteration, mean to handle 2nd order eqs
    vec<T> k0(val.size),k1(val.size),k2(val.size),k3(val.size);
    T h2 = h*0.5;
    k0 = f(t, val);
    k1 = f(t+h2, val+(h2*k0));
    k2 = f(t+h2, val+(h2*k1));
    k3 = f(t+h, val+(h*k2));
    return val + h*(1.0/6.0)*(k0 + k2*2.f + k3*2.f + k3);
};
template <typename T>
inline vec<T> shot(vec<T> (*f)(T, vec<T>), T h, T a, T b, T y0, T yp){
    int its = int(((b-a)/h)+0.5);
    vec<T> results(its+1);results(0)= y0;
    T t=a;
    vec<T> val(2);
    val = {y0, yp};
    val.printout();
    for(int i = 1; i< results.size; i++){
        val = rk4(f, h, t, val);
        //val.printout();
        results(i) = val(0);
        t+=h;
    }
    return results;
}

//I have a vectorized form of rk which makes this so much easier
template <typename T>
inline vec<T> linear_shooting(vec<T> (*fg)(T, vec<T>),vec<T> (*fp)(T, vec<T>), T h, T a, T b, T y0, T y1){
    int its = int(((b-a)/h)+0.5);
    vec<T> low_vec(its+1), high_vec(its+1);
    T c = (y1-y0)/(b-a);
    std::cout << "C: " << c << '\n';
    
    low_vec = shot(fp, h, a, b, y0, (T)0);
    high_vec = shot(fg, h, a, b, (T)0, c);
    low_vec.printout();
    high_vec.printout();
    T scalar = (y1-low_vec.back())/high_vec.back();
    low_vec = low_vec + scalar*high_vec;
    return low_vec;
}

template <typename T>
inline void write_to_file(std::string filename, T sb, T sa, T tb, T ta, T h, T k, int sk, int sh, vec<T> *holder){
    std::ofstream file;
    file.open(filename);
    float min = 0, max = 0;
    double temp_number;
    for(int i = 0; i < sk; i++){
        for(int j =0; j < holder[i].size;j++){
            temp_number = holder[i](j);
            if(temp_number > max){
                max = temp_number;
            }
            if(temp_number < min){
                min = temp_number;
            }
        }
    }
    
    file << sb << ',' << sa << ',' << tb << ',' << ta <<  ',' << max << ',' << min << ',' << h << ',' << k << ',' << sk*sh << '\n';
    std::cout << sb << ',' << sa << ',' << tb << ',' << ta <<  ',' << max << ',' << min << ',' << h << ',' << k << ',' << sk*sh << '\n';

    for(int i =0; i < sk; i++){
        file << (float)holder[i](0);
        //std::cout << (float)holder[i](0);
        for(int j = 1; j < holder[i].size; j++){
            file << ',' << (float)holder[i](j);
            //std::cout << ',' << (float)holder[i](j);
        }
        file <<'\n';
        //std::cout << '\n';
    }
    file.close();
};

template <typename T>
inline void FTBS(T alpha, T h, T k, T ta, T tb, T sa, T sb, vec<T> v0, vec<T> bv, std::string filename){
    int sk = ((tb-ta)/k)+0.5;sk++;
    int sh = ((sb-sa)/h)+0.5;sh++;
    T first = 1 - (alpha * (k/h));
    T second = (alpha * (k/h));
    vec<T> *holder = (vec<T>*)calloc( sk, sizeof(v0)); //place holder to write to file
    matrix<T> step(sh,sh);
    step(0,0) =1;

    for(int i =1; i<sh;i++){
        step(i,i) = first;
        step(i,i-1) = second;
    }
    step.printout();
    holder[0] = v0;
    for(int i = 1; i < sk;i++){
        holder[i] = step*holder[i-1];
        holder[i](0) = bv(i);
    }
    holder[0].printout();
    write_to_file(filename, sb,sa,tb,ta,h,k,sk, sh, holder);
}

template <typename T>
inline void FTFS(T alpha, T h, T k, T ta, T tb, T sa, T sb, vec<T> v0, vec<T> bv, std::string filename){
    int sk = ((tb-ta)/k)+0.5;sk++;
    int sh = ((sb-sa)/h)+0.5;sh++;
    T first = 1 + (alpha * (k/h));
    T second = -(alpha * (k/h));
    vec<T> *holder = (vec<T>*)calloc( sk, sizeof(v0));
    matrix<T> step(sh,sh);
    step(0,0) =1;
    for(int i =1; i<sh-1;i++){
        step(i,i) = first;
        step(i,i+1) = second;
    }
    step(sh-1,sh-1)= first;
    holder[0] = v0;
    for(int i = 1; i < sk;i++){
        holder[i] = step*holder[i-1];
        holder[i](0) = bv(i);
    }
    
    write_to_file(filename, sb,sa,tb,ta,h,k,sk, sh, holder);
}
template <typename T>
inline void FTCS(T alpha, T h, T k, T ta, T tb, T sa, T sb, vec<T> v0, vec<T> bv, std::string filename){
    int sk = ((tb-ta)/k)+0.5;sk++;
    int sh = ((sb-sa)/h)+0.5;sh++;
    T first = 1 - (alpha * (k/h));
    T second = (alpha * (k/h));
    vec<T> *holder = (vec<T>*)calloc( sk, sizeof(v0));
    matrix<T> step(sh,sh);
    step(0,0) =1;
    for(int i =1; i<sh-1;i++){
        step(i,i-1) = second*-0.5f;
        step(i,i) = 1;
        step(i,i+1) = second*0.5f;
    }
    step(sh-1,sh-1)= first;
    step(sh-1,sh-2)= second;
    holder[0] = v0;
    for(int i = 1; i < sk;i++){
        holder[i] = step*holder[i-1];
        holder[i](0) = bv(i);
    }
    write_to_file(filename, sb,sa,tb,ta,h,k,sk, sh, holder);
}

template <typename T>
inline void FTCS_Heat(T alpha2, T h, T k, T ta, T tb, T sa, T sb, vec<T> v0, vec<T> bv, std::string filename){
    int sk = ((tb-ta)/k)+0.5;sk++;
    int sh = ((sb-sa)/h)+0.5;sh++;
    T first = 1 - (2*alpha2 * (k/(h*h)));
    T second = (alpha2 * (k/(h*h)));
    vec<T> *holder = (vec<T>*)calloc( sk, sizeof(v0));
    matrix<T> step(sh,sh);
    for(int i = 1; i < sh-1;i++){
        step(i,i-1) = second;
        step(i,i) = first;
        step(i,i+1) = second;
    }
    step(0,0) = first;
    step(0,1) = second;
    step(sh-1,sh-1) = first;
    step(sh-1,sh-2) = second;
    holder[0] = v0; 
    step.printout();
    for(int i = 1; i < sk; i++){
        holder[i] = step*holder[i-1];
    }
    write_to_file(filename, sb,sa,tb,ta,h,k,sk, sh, holder);
}


template <typename T>
inline void BTCS_Heat(T alpha2, T h, T k, T ta, T tb, T sa, T sb, vec<T> v0, vec<T> bv1, vec<T> bv2 ,std::string filename){
    int sk = ((tb-ta)/k)+0.5;sk++;
    int sh = ((sb-sa)/h)+0.5;sh++;
    T first = 1 + (2*alpha2 * (k/(h*h)));
    T second = -(alpha2 * (k/(h*h)));
    vec<T> *holder = (vec<T>*)calloc( sk, sizeof(v0));
    matrix<T> step(sh,sh);
    for(int i = 1; i < sh-1;i++){
        step(i,i-1) = second;
        step(i,i) = first;
        step(i,i+1) = second;
    }
    step(0,0) = first;
    step(0,1) = second;
    step(sh-1,sh-1) = first;
    step(sh-1,sh-2) = second;
    holder[0] = v0; 
    for(int i = 1; i < sk; i++){
        holder[i-1](0) = bv1(i-1);
        holder[i-1](sh-1) = bv2(i-1); 
        holder[i] = lin_solver(step,holder[i-1]);
    }
    holder[sk-1](0) = bv1(sk-1);
    holder[sk-1](sh-1) = bv2(sk-1);
    write_to_file(filename, sb,sa,tb,ta,h,k,sk, sh, holder);
}
template <typename T>
inline void Crank_Heat(T alpha2, T h, T k, T ta, T tb, T sa, T sb, vec<T> v0, vec<T> bv, std::string filename){
    int sk = ((tb-ta)/k)+0.5;sk++;
    int sh = ((sb-sa)/h)+0.5;sh++;
    T first = 1 + (alpha2 * (k/(h*h)));
    T second = -(alpha2 * (k/(h*h)))*0.5f;
    T third = 1 - (alpha2 * (k/(h*h)));
    T fourth = (alpha2 * (k/(h*h)))*0.5f;
    vec<T> *holder = (vec<T>*)calloc( sk, sizeof(v0));
    matrix<T> step(sh,sh);
    matrix<T> step2(sh,sh);
    for(int i = 1; i < sh-1;i++){
        step(i,i-1) = second;
        step(i,i) = first;
        step(i,i+1) = second;
        step2(i,i-1)= fourth;
        step2(i,i) = third;
        step2(i,i+1) = fourth;
    }
    step(0,0) = first;step(0,1) = second;
    step(sh-1,sh-1) = first;step(sh-1,sh-2) = second;

    step2(0,0) = third;step2(0,1) = fourth;
    step2(sh-1,sh-1) = third;step2(sh-1,sh-2) = fourth;

    holder[0] = v0; 
    vec<T> temp_v(v0);//this is just some fancy stuff to basically make a copy, size and all
    for(int i = 1; i < sk; i++){
        temp_v = step2*holder[i-1];
        holder[i] = lin_solver(step, temp_v);
    }
    write_to_file(filename, sb,sa,tb,ta,h,k,sk, sh, holder);
}



template <typename T>
inline vec<T> FTCS_Heat(T alpha2, T h, T k, T ta, int sk, T sa, int sh, vec<T> v0, vec<T> bv1, vec<T> bv2){
    T first = 1 - (2*alpha2 * (k/(h*h)));
    T second = (alpha2 * (k/(h*h)));
    matrix<T> step(sh,sh);
    for(int i = 1; i < sh-1;i++){
        step(i,i-1) = second;
        step(i,i) = first;
        step(i,i+1) = second;
    }
    step(0,0) = first;
    step(0,1) = second;
    step(sh-1,sh-1) = first;
    step(sh-1,sh-2) = second;
    v0(0) = bv1(0);
    v0(sh-1) = bv2(0);
    return step*v0;
}

template <typename T>
inline void Leap_Frog(T alpha2, T h, T k, T ta, T tb, T sa, T sb, vec<T> v0, vec<T> bv0, vec<T> bv1, std::string filename){
    int sk = ((tb-ta)/k)+0.5;sk++;
    int sh = ((sb-sa)/h)+0.5;sh++;
    matrix<T> init(sh,sh);
    matrix<T> step(sh,sh);
    vec<T> *holder = (vec<T>*)calloc(sk, sizeof(v0));//holder for all the points
    vec<T> temp(v0.size);
    temp = FTCS_Heat(alpha2, -h, -k, ta, sk, sa, sh, v0, bv0, bv1);// for initalization, using a single step method backwards.
    T first = -4 * alpha2 * (k/(h*h));
    T second = 2 * alpha2 * (k/(h*h));
    //generating the matrix 
    step(0,0) = first;
    step(0,1) = second;
    for(int i = 1; i < sh-1;i++){
        step(i,i-1) = second;
        step(i,i) = first;
        step(i,i+1) = second;
    }

    holder[0] = v0;
    holder[1] = (step*v0) + temp; // first entry using our derived entry.
    for(int i =2; i < sk; i++){
        holder[i-1](0) = bv0(i);//boundry cons
        holder[i-1](sh-1) = bv1(i); // boundry cons
        holder[i] = (step*holder[i-1]) + holder[i-2];
    }
    holder[sk-1](0) = 0;
    holder[sk-1](sh-1) = 0;
    write_to_file(filename, sb,sa,tb,ta,h,k,sk, sh, holder);
}

//false means derichlet 
//true means von neuman boundary conditions
template <typename T>
inline void CTCS_wave(T c, T h, T sa, T sb, T k, T ta, T tb, vec<T> v0, vec<T> bv1, bool c1, vec<T> bv2, bool c2, std::string filename){
    
    int sk = ((tb-ta)/k)+0.5;sk++; //amount of total time steps 
    int sh = ((sb-sa)/h)+0.5;sh++; //amount of total spacial steps 
    matrix<T> step(sh,sh); //the step matrix
    vec<T> *holder = (vec<T>*)calloc( sk, sizeof(v0)); //structure to store each step, don't stress about it its a vector of vectors
    T fact = 2*h;
    holder[0] = v0; //adding first iteration into the place holder
    if(!c1){holder[0](0)=bv1(sk-1);}else{holder[0](0) = holder[0](1) - bv1(0)*fact;} //boundry condition stuffs 
    if(!c2){holder[0](sh-1)=bv1(sk-1);}else{holder[0](sh-1) = holder[0](sh-2) + bv1(0)*fact;}
    std::cout << "1\n";
    T first = c * c * (k/h) * (k/h); //adding the first constant
    if(!c1){step(0,0) = 1;}else{step(0,1) = first;}
    if(!c2){step(sh-1,sh-1) = 1;}else{step(sh-1,sh-2) = first;}
    T second = 1 - first; //the second constant to avoid repetative compute
    first*=0.5; //adjusting the frist constant

    std::cout << "2\n";
    for(int i = 1; i < sh-1; i++){//generating the matrix 
        step(i, i-1) = first;
        step(i,i) = second;
        step(i, i+1) = first;
    }
    holder[1] = (step*holder[0]); //+ (k*bv1);
    step*=2;

    if(!c1){step(0,0)=1;} //boundry condition stuffs
    if(!c2){step(sh-1,sh-1)=1;}

    for(int i = 2; i < sk; i++){ //each iteration 
        if(!c1){holder[i-1](0)=bv1(sk-1);}else{holder[i-1](0) = holder[i-1](1) - bv1(i-1)*fact;} //boundry condition stuffs 
        if(!c2){holder[i-1](sh-1)=bv1(sk-1);}else{holder[i-1](sh-1) = holder[i-1](sh-2) + bv1(i-1)*fact;}
        holder[i] = step*holder[i-1] - holder[i-2];
    }

    
    if(!c1){holder[sk-1](0)=bv1(sk-1);}else{holder[sk-1](0) = holder[sk-1](1) - bv1(sk-1)*fact;} //boundry condition stuffs 
    if(!c2){holder[sk-1](sh-1)=bv1(sk-1);}else{holder[sk-1](sh-1) = holder[sk-1](sh-2) + bv1(sk-1)*fact;}
    
    write_to_file(filename, sb,sa,tb,ta,h,k,sk, sh, holder);//placing in a file to then graph
}

template <typename T>
inline void write_to_file(std::string filename, T sb, T sa, T tb, T ta, T h, T k, int sk, int sh, matrix<T> holder){
    std::ofstream file;
    file.open(filename);
    float min = 0, max = 0;
    double temp_number;
    holder.printout();
    std::cout << holder.col << " vs " << sk << '\n';
    std::cout << holder.row << " vs " << sh << '\n';
    for(int i = 0; i < holder.col; i++){
        for(int j =0; j < holder.row;j++){
            temp_number = holder(j,i);
            if(temp_number > max){
                max = temp_number;
            }
            if(temp_number < min){
                min = temp_number;
            }
        }
    }
    std::cout << "Max: " << max << ", Min: " << min << '\n';
    
    file << sb << ',' << sa << ',' << tb << ',' << ta <<  ',' << max << ',' << min << ',' << h << ',' << k << ',' << sk*sh << '\n';

    for(int i =0; i < holder.col; i++){
        file << (float)holder(0, i);
        for(int j = 1; j < holder.row; j++){
            file << ',' << (float)holder(j, i);
        }
        file <<'\n';
    }
    file.close();
}

template <typename T>
inline matrix<T> to_mat(vec<T> in, int n, int m){
    matrix<T> a(n,m);
    for(int i = 0; i < a.col; i++){
        for(int j = 0; j < a.row; j++){
            a(j,i) = in((i*a.row)+j);
        }
    }
    return a;
}


//This version is assuming Dirichlet conditons
//h, step in x, sa start in x, sb end in x
//k, step in t, ta start in t, tb end in t
//v0, values at t = 0 along x, v1, values at end of time domain along x
//bv0,  values at x = 0 along t, bv1 values at end of space domain along t
//Filename, resultant file
template<typename T>
inline matrix<T> laplace_CTCS(T h, T sa, T sb, T k, T ta, T tb, vec<T> v0, vec<T> v1,vec<T> bv0, vec<T> bv1, std::string filename){
    int sh = (int)(((sb-sa)/h)+0.5)+1; //number of steps in space
    int sk = (int)(((tb-ta)/k)+0.5)+1; //number of steps in time
    vec<T> b(1);
    matrix<T> temp_sol(sh, sk); //generating the boundry conditions
    for(int i =0; i < sk; i++){
        temp_sol(0,i) = bv0(i); //left and right edges of the matrix
        temp_sol(sh-1,i) = bv1(i);
    }
    for(int i =0; i < sh; i++){//top and bottom edges of the matrix
        temp_sol(i,0) = v0(i);
        temp_sol(i,sk-1) = v1(i);
    }
    b = vectorize(temp_sol);//turning the matrix into a vector, of columns stacked on to each other
    matrix<T> a(b.size, b.size);
    for(int i = 0; i < b.size; i++){
        a(i,i) = 1;
    }
    T const1 = 2*(1 + (k*k)/(h*h));
    T const2 = -1*((k*k)/(h*h));
    //this needs to be redone, based on known values;
    for(int i = (sh+1); i < (b.size-(sh+1)); i+=sh){//this loops through each column
        for(int j = 0; j <sh-2;j++){//this inner one fills in for the unknown entries
            a(i+j,i+j) = const1;
            a(i+j,i+j-1) = const2;
            a(i+j,i+j+1) = const2;
            a(i+j,i+j-sh) = -1;
            a(i+j,i+j+sh) = -1;
        }
    }
    a.printout();
    matrix<T> solution(1,1);
    solution = to_mat(lin_solver(a,b), sh, sk);
    write_to_file(filename, sb,sa,tb,ta,h,k,sk, sh, solution);
    return solution;
};



template <typename T>
static vec<T> slowHT(vec<T> input)
{
    vec<T> copy(0),out(0);
    copy = input;
    out = input;
    T n = input.size;
    for(int k=0; k<n; ++k){
	    input(k)= 0.0;
	    for(int i=0; i<input.size; ++i){
            float angle = 2*M_PI*i*k/(T)(n);
            out(k) += copy(i)*(cos(angle)+sin(angle));
	    }
    }
    return out;
}
}
