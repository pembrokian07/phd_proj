//
//  Solver.cpp
//  CarNumerics
//
//  Created by Kevin Liu on 08/11/2020.
//

#include "Solver.hpp"

using namespace arma;

Solver::Solver(int m, int n, int t, float dt, float M, float I1, float I2, float h0, float v0, float theta1, float d_theta1, float theta2, float d_theta2)
    :_m(m),_n(n),_t(t),_dt(dt),_M(M),_I1(I1), _I2(I2)
{
    _dr = _R/_m;
    _da = (float) (2.0f*M_PI/_n);
    
    // number of unknowns to solve
    _neumann_indice_count = (int)round(_n/2.0f)-1;
    _d = _m*_n;
    _A = new sp_fmat(_d, _d);
    _A->zeros();
    _b = new fvec(_d, fill::zeros);
    
    // initialize h with h0 and v0
    _h = new fvec(_t+1, fill::zeros);
    // use first order difference to approx h(-dt)
    _h->at(0) = h0 - v0*_dt;
    _h->at(1) = h0;
    
    _theta1 = new fvec(_t+1, fill::zeros);
    _theta1->at(0) = theta1 - d_theta1*_dt;
    _theta1->at(1) = theta1;
    
    _theta2 = new fvec(_t+1, fill::zeros);
    _theta2->at(0) = theta2 - d_theta2*_dt;
    _theta2->at(1) = theta2;
    
    cout<<"dr: "<<_dr<<endl;
    cout<<"da: "<<_da<<endl;
    
    boost::push_back(_neumann_indices, boost::irange(0, (int) round(_n/4.0f)));
    boost::push_back(_neumann_indices, boost::irange((int)round(3.0f/4*_n) + 1, _n));
    
    // populate the dirichlet indices
    std::vector<int> full_set;
    boost::push_back(full_set, boost::irange(0, _n));
    std::set_difference(full_set.begin(), full_set.end(), _neumann_indices.begin(), _neumann_indices.end(), std::inserter(_dirichlet_indices, _dirichlet_indices.begin()));
    
    //boost::push_back(_neumann_indices, boost::irange(0,_n));
}

Solver::~Solver()
{
    delete _A;
    delete _b;
}

void Solver::reset()
{
    _A->zeros();
    _b->zeros();
}

void Solver::save_neumann_indices(std::string fileName)
{
    vec indices(_neumann_indice_count);
    int k = 0;
    for(int i: _neumann_indices)
    {
        indices(k++) = i;
    }
    indices.save(fileName, csv_ascii);
}

void Solver::print()
{
    _A->print("A:");
    _b->t().print("b:");
}

void Solver::save()
{
    _A->save("/Users/kevinliu/Documents/cpp_proj/CarNumerics/CarNumerics/A.csv", csv_ascii);
    _b->save("/Users/kevinliu/Documents/cpp_proj/CarNumerics/CarNumerics/b.csv", csv_ascii);
}

float Solver::df_dx(int i, int j, int s)
{
    float alpha = p_j(j+1)*_da;
    
    return 2*(i+1)*_dr*cosf(alpha) + _theta1->at(s);
}

// round angle index between d_alpha and 2*pi
int Solver::p_j(int j)
{
    int res = j;
    
    if(j<0)
    {
        res = j + _n;
    }
    // if we have exceeded 2pi, reset it according to periodicity
    else if(j>=_n)
    {
        res = j - _n;
    }
        
    return res;
}

void Solver::populate_inner(int s)
{
    // populate the full circles for r in [2*dr, (m-2)*dr]
    for(int i=1; i<=_m-2; i++)
    {
        for(int j=0; j<_n; j++)
        {
            this->_set_inner_eqns(i,j,s);
        }
    }
}

void Solver::_set_inner_eqns(int i, int j, int s)
{
    int k = i*_n+j;
    float alpha = (j+1)*_da;

    float inv_da2 = powf((i+1)*_da, -2.0f);
    float c1 = powf(cosf(alpha), 2.0f) + powf(sinf(alpha), 2.0f)*_inv_P2;
    float c2 = powf(sinf(alpha), 2.0f) + powf(cosf(alpha), 2.0f)*_inv_P2;
    float c3 = (1 - _inv_P2)*sinf(2*alpha)/_da;
    
    // setting the k-th equation
    (*_A)(k,k) = -2.0f*(c1 + c2*inv_da2);
    (*_A)(k,i*_n+p_j(j+1)) = inv_da2*c2 + c3/(2.0f*powf(i+1, 2.0f));
    (*_A)(k,i*_n+p_j(j-1)) = inv_da2*c2 - c3/(2.0f*powf(i+1, 2.0f));
    
    // when we increment dr we may step onto the boundary, so need to check
    (*_A)(k,(i+1)*_n+j) = c1 + c2/(2*(i+1));
    (*_A)(k,(i-1)*_n+j) = c1 - c2/(2*(i+1));
    (*_A)(k,(i+1)*_n+ p_j(j+1)) = -c3/(4.0f*(i+1));
    (*_A)(k,(i+1)*_n+ p_j(j-1)) = c3/(4.0f*(i+1));
    (*_A)(k,(i-1)*_n+ p_j(j+1)) = c3/(4.0f*(i+1));
    (*_A)(k,(i-1)*_n+ p_j(j-1)) = -c3/(4.0f*(i+1));

    (*_b)(k) = -powf(_dr, 2.0f)*df_dx(i, j, s);
}

void Solver::populate_neumann()
{
    // fill the values for r = M*dr
    for(auto j: this->_neumann_indices)
    {
        float alpha = (j+1)*_da;
        float ita = sinf(alpha)/(_m*_da);
        int i = _m-1;
        int r = i*_n;
        int k = r+p_j(j);
        
        (*_A)(k,k) = 25.0f/12.0f*cosf(alpha);
        (*_A)(k,r+p_j(j+1)) = -2.0f/3.0f*ita;
        (*_A)(k,r+p_j(j+2)) = 1.0f/12.0f*ita;
        (*_A)(k,r+p_j(j-1)) = 2.0f/3.0f*ita;
        (*_A)(k,r+p_j(j-2)) = -1.0f/12.0f*ita;
        
        (*_A)(k,(i-1)*_n + j) = -4.0f*cosf(alpha);
        (*_A)(k,(i-2)*_n + j) = 3.0f*cosf(alpha);
        (*_A)(k,(i-3)*_n + j) = -4.0f/3.0f*cosf(alpha);
        (*_A)(k,(i-4)*_n + j) = 1.0f/4.0f*cosf(alpha);
        
        (*_b)(k) = 0.0f; //_m*powf(_dr, 2.0f)
    }
}

// for i = M-1 (as i = M is given) and j*_da in [pi/2, 3*pi/2]
void Solver::populate_dirichlet()
{
    for(auto j: _dirichlet_indices)
    {
        int i = _m-1;
        int r = i*_n;
        int k = r + j;
        
        (*_A)(k,k) = 1;
        (*_b)(k) = 0.0f;
    }
}

void Solver::populate_origin(int s)
{
    float inv_da2 = powf(_da, -2.0f);
    
    // i is 0
    for (int j=0; j<_n; j++)
    {
        float alpha = (j+1)*_da;
        float c1 = powf(cosf(alpha), 2.0f) + powf(sinf(alpha), 2.0f)*_inv_P2;
        float c2 = powf(sinf(alpha), 2.0f) + powf(cosf(alpha), 2.0f)*_inv_P2;
        float c3 = (1 - _inv_P2)*sinf(2*alpha)/_da;
        
        (*_A)(j,_n+j) = 0.5f*(c1+1.5f*c2);
        (*_A)(j,_n+p_j(j + round(_n/2.0f))) = 1.0f/6*(c1 - 0.5f*c2);
        (*_A)(j,_n+p_j(j-1+round(_n/2.0f))) = -c3/24.0f;
        (*_A)(j,_n+p_j(j+1+round(_n/2.0f))) = c3/24.0f;
        
        (*_A)(j,_n+p_j(j+1)) = -3.0f/8*c3;
        (*_A)(j,_n+p_j(j-1)) = 3.0f/8*c3;
        
        
        (*_A)(j,j) = -2.0f/3*c2 -2*inv_da2*c2 - 2.0f/3*c1;
        (*_A)(j,p_j(j-1)) = c2*inv_da2 - 5.0f/6*c3;
        (*_A)(j,p_j(j+1)) = c2*inv_da2 + 5.0f/6*c3;

        (*_b)(j) = -powf(_dr, 2.0f)*df_dx(0, j, s);
    }
}

void Solver::populate_psi_grid(int s)
{
    this->populate_origin(s);
    this->populate_inner(s);
    this->populate_neumann();
    this->populate_dirichlet();
}

fmat Solver::solve_psi()
{
    fvec x(_b->n_rows);
    spsolve(x, *_A, *_b);
    fmat y(x);
    y.reshape(_n,_m);
    return y.t();
}

fvec Solver::solve_origin(fmat& psi)
{
    fvec x(_n);
    
    for (int j = 0; j<_n; j++)
    {
        x(j) = 4.0f/3*psi(0,j) - 1.0f/2*psi(1,j) + 1.0f/6*psi(1,p_j(j-round(_n/2.0f)));
    }
    
    return x;
}

float Solver::approx_df_dr(fmat& f, int i, int j)
{
    float df_dr = 0.f;
    
    // approx the circle enclosing origin, i = 0, r=dr
    if(i == 0)
    {
        df_dr = (3.0f/4*f(1,j) - 2.0f/3*f(0,j) - 1.0f/12*f(1,p_j(j-round(_n/2.0f))))/_dr;
    }
    else if(i == _m-1)
    {
        df_dr = (25.0f/12*f(i,j) - 4*f((i-1),j) + 3*f(i-2,j) - 4.0f/3*f(i-3,j) + 1.0f/4*f(i-4,j))/_dr;
    }
    else
    {
        df_dr = (f(i+1, j) - f(i-1, j))/(2*_dr);
    }
    
    return df_dr;
}

float Solver::approx_df_da(fmat& f, int i, int j)
{
    float df_da = 0.f;
    
    df_da = (f(i, p_j(j+1)) - f(i, p_j(j-1)))/(2*_da);
        
    return df_da;
}

fmat Solver::solve_pressure(fmat& psi)
{
    fmat pressure(size(psi));
    for(int i=0; i<_m; i++)
    {
        for(int j=0; j<_n; j++)
        {
            float alpha = (j+1)*_da;
            pressure(i,j) = -approx_df_dr(psi, i, j)*cosf(alpha) + approx_df_da(psi, i, j)*sinf(alpha)/((i+1)*_dr);
        }
    }
    return pressure;
}


float Solver::double_integral(fmat& f)
{
    fvec int_alpha = _integrate_alpha(f);
    // element wise multiplication
    vec g = linspace(_dr, _R, _m) % int_alpha;
    
    float result = (float) _dr*(0.5*(g(0) + g(_m-1)) + sum(g(span(1, _m-2))));
    
    return result;
}

fvec Solver::_integrate_alpha(fmat& f)
{
    fvec int_alpha(_m);
    
    for(int i=0; i<_m; i++)
    {
        int_alpha(i) = _da*(0.5f*(f(i,0) + f(i,_n-1)) + accu(f(span(i, i), span(1, _n-2))));
    }
    
    return int_alpha;
}

void Solver::solve_var(float force, int s, fvec& vars)
{
    vars(s) = force/_M*powf(_dt, 2) + 2*vars(s-1) - vars(s-2);
}

void const Solver::save_h(std::string file_name)
{
    _h->save(file_name, csv_ascii);
}

void const Solver::save_theta1(std::string file_name)
{
    _theta1->save(file_name, csv_ascii);
}

fmat Solver::compute_theta1_integrand(fmat& pressure)
{
    fmat row_a = conv_to<fmat>::from(linspace(_da, 2.0f*M_PI, _n)).transform([](float val) { return (cosf(val));}).t();
    fmat col_r = conv_to<fmat>::from(linspace(_dr, _R, _m));
    fmat integrad = col_r*row_a - _x_c;
    
    // element-wise multiplication
    return integrad%pressure;
}

