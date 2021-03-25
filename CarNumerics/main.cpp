//
//  main.cpp
//  HelloWorld
//
//  Created by Kevin Liu on 12/10/2020.
//

#include <iostream>
#include <cmath>
#include "Solver.hpp"
#include <ctime>

using namespace std;
using namespace arma;

void test(){
    vec test;
    test.ones(10);
    vec test2 = linspace(1, 20, 20);
    
    double da = 2*M_PI/10;
    vec test3 = linspace(da, 2*M_PI, 10);
    vec test4 = test % test3;
    
    vec test5 = 10*arma::cos(test3);
    for(int i = 0; i<10; i++)
    {
        cout << test5(i) <<", ";
    }
    cout << endl;
    
    cout << "test out: " <<endl;
    cout << arma::sum(test) <<endl;
    cout << arma::dot(test(span(0,1)), test2(span(0,1))) << endl;
}


int main(int argc, const char * argv[]) {
    time_t t1 = 0;
    time_t t2 = 0;
    string dir = "/Users/kevinliu/temp/CarNumericsData/";
    
    //test();
    
    cout<<"running..."<<endl;
    t1 = time(0);
    int m = 100;
    int n = 100;
    int T = 100;
    
    Solver s(m, n, T, 1.0f/10, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,(float) -1.0f, 0.0f, 0.0f, 0.0f);
    fmat psi(m, n);
    fmat pressure(m,n);
    fmat theta1_integrand(m,n);
    fvec psi0(n);
    
    for(int t = 1; t<T; t++)
    {
        cout<< "t: " << t << endl;
        // solve for psi
        s.populate_psi_grid(t);
        psi = s.solve_psi();
        psi0 = s.solve_origin(psi);
        
        // solve for pressure
        pressure = s.solve_pressure(psi);
        // solve for h
        float h_force = s.double_integral(pressure);
        s.solve_var(h_force, t+1, *(s._h));

        // save h
        cout<< "h lift:" << h_force << endl;
        
        // solve for theta1
        theta1_integrand = s.compute_theta1_integrand(pressure);
        float theta1_force = s.double_integral(theta1_integrand);
        s.solve_var(theta1_force, t+1, *(s._theta1));
        
        // save theta1
        theta1_integrand.save(dir+"theta1_integrand.csv",csv_ascii);
        cout<< "theta1 force:" << theta1_force << endl;
    }
    
    psi.save(dir+"psi.csv", csv_ascii);
    psi0.save(dir+"psi0.csv", csv_ascii);
    pressure.save(dir+"pressure.csv",csv_ascii);
    s.save_h(dir+"h.csv");
    s.save_theta1(dir+"theta1.csv");

    //s.save_neumann_indices("neumann_indices.csv");
    t2 = time(0);
    cout<<t2-t1<<" secs."<<endl;
    //x.print("Sparse x:");
    
    return 0;
}
