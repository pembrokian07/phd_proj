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

class NumericalSolver
{
    private:
        // debug setting, print info if verbose set to true
        bool _verbose = true;
        // absolute file path to save the computed data
        string saveDir;

        // body mass
        float M = 1.0f;
        // moment of inertia j1
        float J1 = 1.0f;
        // moment of inertia j2
        float J2 = 1.0f;

        // grid sizes
        int m = 100;
        int n = 200;
        int T = 10;

        void solverTimeMarch(int T, Solver &s, fmat &psi, fmat &pressure, fmat &theta1_integrand, fmat &vel_u, fmat &vel_v, fvec &psi0)
        {
            for(int t=1; t!=T; t++)
            {
                // solve for psi
                s.populate_psi_grid(t);
                psi = s.solve_psi();
                psi0 = s.solve_origin(psi);
                
                // solve for pressure
                pressure = s.solve_pressure(psi);
                vel_u = s.solve_velocity_u(psi);
                vel_v = s.solve_velocity_v(psi);
                
                // solve for h
                float h_force = s.double_integral(pressure);
                s.solve_var(h_force, t+1, M, *(s._h));
                
                // solve for theta1
                theta1_integrand = s.compute_theta1_integrand(pressure);
                float theta1_force = s.double_integral(theta1_integrand);
                s.solve_var(theta1_force, t+1, J1, *(s._theta1));
                
                //theta1_integrand.save(saveDir+"theta1_integrand.csv",csv_ascii);
                if (_verbose)
                {
                    std::cout<< "time t:" << t << ", h_force:" << h_force << ", theta1_force:" << theta1_force << endl;                    
                }                
            }            
        }

    public:
        NumericalSolver(string dir, int T) : saveDir(dir), T(T) {}

        void solve(vector<int> &qList)
        {
            // stream functions to solve for
            fmat psi(m, n);
            // stream function at origin
            fvec psi0(n);
            fmat pressure(m,n);
            fmat theta1_integrand(m,n);
            fmat vel_u(m,n);
            fmat vel_v(m,n);

            // iterate through all 'q' (i.e. b/a) configurations
            for(auto qIter = qList.cbegin(); qIter!= qList.cend(); qIter++)
            {
                time_t t1 = time(0);
                std::cout << "Solving for q = " << *qIter << endl;

                // initialise the solver for this new 'q' configuration
                Solver s(m, n, T, 0.1f, *qIter);                
               
                psi.reset();
                pressure.reset();
                theta1_integrand.reset();
                psi0.reset();
                vel_u.reset();
                vel_v.reset();

                // time march for solution up to time T
                solverTimeMarch(T, s, psi, pressure, theta1_integrand, vel_u, vel_v, psi0);                               
                
                // save the results to csv
                psi.save(saveDir+"psi_" + to_string(*qIter) +".csv", arma::file_type::csv_ascii);
                psi0.save(saveDir+"psi0_" + to_string(*qIter) + ".csv", arma::file_type::csv_ascii);
                pressure.save(saveDir+"pressure_" + to_string(*qIter) +".csv",arma::file_type::csv_ascii);
                vel_u.save(saveDir+"vel_u_" + to_string(*qIter) +".csv",arma::file_type::csv_ascii);
                vel_v.save(saveDir+"vel_v_" + to_string(*qIter) +".csv",arma::file_type::csv_ascii);
                s.save_h(saveDir+"h_" + to_string(*qIter) +".csv");
                s.save_theta1(saveDir+"theta_A_" + to_string(*qIter) + ".csv");

                time_t t2 = time(0);
                std::cout << "Done! (" << t2-t1 << " secs.) " << endl;
            }
        }
};

int main() 
{
    time_t t1 = 0;
    time_t t2 = 0;
    
    string dir = "/home/kevin/Documents/gitlab/phd_proj/CarNumerics/NumericsData/";
    int T = 20;
    std::cout<<"running..."<<endl;
    t1 = time(0);
    NumericalSolver ns(dir,T);
    vector<int> qList = {1};
    ns.solve(qList);
    
    t2 = time(0);
    cout<<t2-t1<<" secs."<<endl;    

    return 0;
}