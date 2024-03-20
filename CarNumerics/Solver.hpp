//
//  Solver.hpp
//
//  Created by Kevin Liu on 08/11/2020.
//

#ifndef Solver_hpp
#define Solver_hpp

#define ARMA_USE_SUPERLU
#define ARMA_USE_HDF5

#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <boost/range/irange.hpp>
#include <boost/range/algorithm_ext/push_back.hpp>
#include <armadillo>

using namespace arma;

class Solver
{
public:
    Solver(int m, int n, int t, float dt, float q);
    Solver(int m, int n, int t, float dt, float q, float h0, float v0, float theta1, float d_theta1, float theta2, float d_theta2);
    ~Solver();
    void reset();
    float df_dx(int i, int j, int s);
    void save(std::string dir_name);
    
    fmat solve_psi();
    fvec solve_origin(fmat& psi);
    float approx_df_dr(fmat& f, int i, int j);
    float approx_df_da(fmat& f, int i, int j);
    fmat solve_pressure(fmat& psi);
    fmat solve_velocity_u(fmat& psi);
    fmat solve_velocity_v(fmat& psi);
    float double_integral(fmat& f);
    
    void solve_var(float force, int s, float mass, fvec& vars);
    void const save_h(std::string file_name);
    fmat compute_theta1_integrand(fmat& pressure);
    void const save_theta1(std::string file_name);

    void populate_psi_grid(int s);
    void populate_inner(int s);
    void populate_origin(int s);

    void populate_dirichlet();
    void populate_neumann();
    void populate_neumann2();
    float dirichlet_bc(int j);
    float neumann_bc(int j);    
    void save_neumann_indices(std::string fileName);
    
    fvec *_h;
    fvec *_theta1;
    fvec *_theta2;
    
private:
    const float _R = 1.0f;
    const float _x_c = 0.0f;
    const float _y_c = 0.0f;
    float _P;
    float _inv_P2;
    int _m;
    int _n;
    int _t;
    int _d;
    float _dr;
    float _da;
    float _dt;
    int _neumann_indice_count;
    
    sp_fmat *_A;
    fvec *_b;
    fvec *_x;
    
    std::vector<int> _neumann_indices;
    std::vector<int> _dirichlet_indices;
    fvec _integrate_alpha(fmat& f);

    void _set_inner_eqns(int i, int j, int s);
    int p_j(int j);
};

#endif /* Solver_hpp */
