//
//  main.cpp
//
//  Created by Kevin Liu on 12/10/2020.
//

#include <iostream>
#include <cmath>
#include <ctime>
#include <vector>
#include "Solver.hpp"

using namespace std;
using namespace arma;

int main(int argc, const char * argv[]) {
    time_t t1 = 0;
    time_t t2 = 0;
    
    string dir = "/Users/kevinliu/temp/NumericsData/";
    
    cout<<"running..."<<endl;
    t1 = time(0);
    int m = 100;
    int n = 200;
    int T = 10;
    // body mass
    float M = 1.0f;
    // moment of inertia j1
    float J1 = 1.0f;
    // moment of inertia j2
    //float J2 = 1.0f;
    
    vector<float> vec_h_force;
    vector<float> vec_torque;
    for(float q = 1.0f; q<=1.0f; q=q+0.5f)
    {
        cout << "q: " << q << endl;
        Solver s(m, n, T, 0.1f, q, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        fmat psi(m, n);
        fmat pressure(m,n);
        fmat theta1_integrand(m,n);
        fvec psi0(n);
        fmat vel_u(m,n);
        fmat vel_v(m,n);
        float h_force = 0.0f;
        float theta1_force = 0.0f;
        
        for(int t = 1; t<T; t++)
        {
            //cout<< "t: " << t << endl;
            // solve for psi
            s.populate_psi_grid(t);
            psi = s.solve_psi();
            psi0 = s.solve_origin(psi);
            
            // solve for pressure
            pressure = s.solve_pressure(psi);
            vel_u = s.solve_velocity_u(psi);
            vel_v = s.solve_velocity_v(psi);
            
            // solve for h
            h_force = s.double_integral(pressure);
            s.solve_var(h_force, t+1, M, *(s._h));
            
            // save h
            cout<< "h lift:" << h_force << endl;
            // solve for theta1
            theta1_integrand = s.compute_theta1_integrand(pressure);
            theta1_force = s.double_integral(theta1_integrand);
            s.solve_var(theta1_force, t+1, J1, *(s._theta1));
            
            // save theta1
            //theta1_integrand.save(dir+"theta1_integrand.csv",csv_ascii);
            cout<< "theta1 force:" << theta1_force << endl;
        }
        vec_h_force.push_back(h_force);
        vec_torque.push_back(theta1_force);
        
        psi.save(dir+"psi_" + to_string(q) +".csv", csv_ascii);
        psi0.save(dir+"psi0_" + to_string(q) + ".csv", csv_ascii);
        pressure.save(dir+"pressure_" + to_string(q) +".csv",csv_ascii);
        vel_u.save(dir+"vel_u_" + to_string(q) +".csv",csv_ascii);
        vel_v.save(dir+"vel_v_" + to_string(q) +".csv",csv_ascii);

        s.save_h(dir+"h_" + to_string(q) +".csv");
        s.save_theta1(dir+"theta_A_" + to_string(q) + ".csv");
        
        //s.save(dir);
    }

    cout << "h force:" << endl;
    // declare iterator
    vector<float>::iterator iter;
    // use iterator with for loop
    for (iter = vec_h_force.begin(); iter != vec_h_force.end(); ++iter) {
        cout << *iter << ",";
    }
    cout << endl;
    
    cout << "torque force:" << endl;
    // declare iterator
    //vector<float>::iterator iter;
    // use iterator with for loop
    for (iter = vec_torque.begin(); iter != vec_torque.end(); ++iter) {
        cout << *iter << ",";
    }
    cout << endl;
    
    //s.save_neumann_indices("neumann_indices.csv");
    t2 = time(0);
    cout<<t2-t1<<" secs."<<endl;
    //x.print("Sparse x:");
    
    return 0;
}
