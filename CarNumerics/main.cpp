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
    // version number, used to identify the saved csv files.
    string version_info = "5";
    
    cout<<"running..."<<endl;
    t1 = time(0);
    int m = 500;
    int n = 200;
    float dt = 0.1;
    // number of dt to run
    int N_t = 50;
    // the elliptical axis ratio, q=1 is a circle
    float q = 1.0f;
    // body mass
    float M = 1.0f;
    //2.0f;
    // moment of inertia j1
    float J1 = 1.0f;
    //0.3f;
    // moment of inertia j2
    //float J2 = 1.0f;
    
    // mem alloc
    fmat psi(m, n);
    fmat pressure(m,n);
    fmat theta1_integrand(m,n);
    fvec psi0(n);
    fmat vel_u(m,n);
    fmat vel_v(m,n);
    float h_force = 0.0f;
    float theta1_force = 0.0f;
    vector<float> vec_h_force;
    vector<float> vec_torque;

    // omega used for g(x,y,t)
    float omega = 0;
    Solver s(m, n, N_t, dt, q, omega, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    // set the g(x,y,t) function
    s.set_body_g_params(0.8, 0.1, 0.1);
    
    for(int t = 1; t<N_t; t++)
    {
        if (t % 10 == 0){cout<< "t: " << t << endl;}
        
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
        //cout<< "h lift:" << h_force << endl;
        // solve for theta1
        theta1_integrand = s.compute_theta1_integrand(pressure);
        theta1_force = s.double_integral(theta1_integrand);
        s.solve_var(theta1_force, t+1, J1, *(s._theta1));
        
        // save theta1
        //theta1_integrand.save(dir+"theta1_integrand.csv",csv_ascii);
        //cout<< "theta1 force:" << theta1_force << endl;
    }
    vec_h_force.push_back(h_force);
    vec_torque.push_back(theta1_force);
        
    psi.save(dir+"psi_" + version_info +".csv", csv_ascii);
    psi0.save(dir+"psi0_" + version_info + ".csv", csv_ascii);
    pressure.save(dir+"pressure_" + version_info +".csv",csv_ascii);
    vel_u.save(dir+"vel_u_" + version_info +".csv",csv_ascii);
    vel_v.save(dir+"vel_v_" + version_info +".csv",csv_ascii);
    s.save_h(dir+"h_" + version_info +".csv");
    s.save_theta1(dir+"theta_A_" + version_info + ".csv");
    
    //s.save(dir);

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
