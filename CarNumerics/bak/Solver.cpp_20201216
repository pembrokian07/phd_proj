//
//  Solver.cpp
//  CarNumerics
//
//  Created by Kevin Liu on 08/11/2020.
//

#include "Solver.hpp"

using namespace arma;

Solver::Solver(int m, int n, float theta1, float theta2):_m(m),_n(n),_theta1(theta1),_theta2(theta2)
{
    _dr = _R/_m;
    cout<<"dr: "<<_dr<<endl;
    _da = (float) (2.0f*M_PI/_n);
    cout<<"da: "<<_da<<endl;
    // number of unknowns to solve
    _neumann_indice_count = 0;//(int)round(_n/2.0f)-1;
    _d = (_m-1)*_n+_neumann_indice_count;
    _A = new sp_fmat(_d, _d);
    _A->zeros();
    _b = new fvec(_d, fill::zeros);
    
    /*
    cout << "nb1: " << (int) round(_n/4.0f-1) << endl;
    cout << "nb2: " << (int) round(3.0f/4*_n) + 1 << endl;
    
    boost::push_back(_neumann_indices, boost::irange(0, (int) round(_n/4.0f)));
    boost::push_back(_neumann_indices, boost::irange((int)round(3.0f/4*_n) + 1, _n));
    
    // populate the dirichlet indices
    std::vector<int> full_set;
    boost::push_back(full_set, boost::irange(0, _n));
    std::set_difference(full_set.begin(), full_set.end(), _neumann_indices.begin(), _neumann_indices.end(), std::inserter(_dirichlet_indices, _dirichlet_indices.begin()));
    */
    boost::push_back(_dirichlet_indices, boost::irange(0,_n));
    
    /*
    cout << "dirichlet: ";
    for (int j: _dirichlet_indices)
    {
     cout << j << " ";
    }
    cout << endl;
    
    cout << "neumann: ";
    for (int j: _neumann_indices)
    {
        cout << j << " ";
    }
    cout << endl;
    */
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

void Solver::set_angles(float theta1, float theta2)
{
    _theta1 = theta1;
    _theta2 = theta2;
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

// round angle index between d_alpha and 2*pi
int Solver::p_j(int j, int i)
{
    int res = j;
    
    // if this is for angle mapping away from the outter most boundary
    // todo: change this to _m-2 for mixed BCs, _m for all neumann
    if (i <= _m-2)
    {
        if(j<0)
        {
            res = j + _n;
        }
        // if we have exceeded 2pi, reset it according to periodicity
        else if(j>=_n)
        {
            res = j - _n;
        }
    }
    else
    {
        // if we have exceeded 2pi, reset it according to periodicity first
        if(j < 0)
        {
            res = _neumann_indice_count + j;
        }
        else if(j >= _n)
        {
            res = j - _n;
        }
        // map the 4th quadrant to the 2nd quadrant indices
        else if((j > (int)round(_n/4.0f*3.0f)) && (j < _n))
        {
            res = (int) round(j-_n/2.0f) - 1;
        }
        // first quadrant remains as it is
        else if(j <= (int)round(_n/4.0f - 1))
        {
            res = j;
        }
        // 2nd and 3rd quadrants do not map
        else
        {
            res = -1;
        }
        
    }
    return res;
}

float Solver::df_dr(int j)
{
    float alpha = (j+1)*_da;
    return cosf(alpha)*_theta1 + sinf(alpha)*_theta2;
}

float Solver::df_da(int i, int j)
{
    float alpha = p_j(j+1)*_da;
    float r = (i+1)*_dr;
    return r*(cosf(alpha)*_theta2 - sinf(alpha)*_theta1);
}

float Solver::neumann_bc(int j)
{
    float res;
    float alpha = (j+1)*_da;
    res = (sinf(alpha)*cosf(alpha) + powf(cosf(alpha),2.0f) + 1);
    
    /*
    // this is only for mixed boundary conditions
    if(j < _dirichlet_indices.front() || j > _dirichlet_indices.back())
    {
        float alpha = (j+1)*_da;
        res = sinf(alpha)*cosf(alpha) + powf(cosf(alpha),2.0f) + 1;
        //res = 0.0f;
    }
    else
    {
        cout << "neumann bc: " << j << " - NaN" << endl;
        res = std::nanf("");
    }*/
    return res;
}

fvec Solver::solve_system()
{
    fvec x(_b->n_rows);
    spsolve(x, *_A, *_b);
    return x;
}

/*
// for i in [M-1, M] and j*_da in (-pi/2, pi/2)
// int d = (_m-1)*_n+(int)(_n/2)-1;
void Solver::populate_neumann()
{
    // fill the values for r = (M-1)*dr
    for(auto j: this->_neumann_indices)
    {
        this->_set_inner_eqns(_m-2, j);
    }
    
    // fill the values for r = M*dr
    for(auto j: this->_neumann_indices)
    {
        float alpha = (j+1)*_da;
        float ita = 1+1.0f/(2.0f*_m);
        float inv_da2 = powf(_m*_da, -2.0f);
        int i = _m-1;
        int r = i*_n;
        int k = r+p_j(j,i);
        
        (*_A)(k,k) = -2*cosf(alpha)*(1+inv_da2);
        (*_A)(k,(i-1)*_n + j) = 2.0f*cosf(alpha);
        
        int mapped_index = p_j(j-1,i);
        if(mapped_index != -1)
        {
            (*_A)(k,r+mapped_index) = cosf(alpha)*inv_da2 - ita*sinf(alpha)/(_m*_da);
            //-(beta + beta/(2*_m) - cosf(alpha)*inv_da2);
            //cosf(alpha)*inv_da2 - beta*sinf(alpha)/(_m*_da);
            //
        }
        
        mapped_index = p_j(j+1,i);
        if(mapped_index != -1)
        {
            (*_A)(k,r+mapped_index) = cosf(alpha)*inv_da2 + ita*sinf(alpha)/(_m*_da);
                //(beta + beta/(2*_m) + cosf(alpha)*inv_da2);
                //cosf(alpha)*inv_da2 + beta*sinf(alpha)/(_m*_da);
        }
        
        (*_b)(k) = (powf(_dr, 2.0f)*3.0f*(sinf(alpha)+cosf(alpha))*cosf(alpha) - ita*2.0f*_m*powf(_dr, 2.0f)*(sinf(alpha)*cosf(alpha) + powf(cosf(alpha), 2.0f) + 1));
        //(cosf(alpha)*3*powf(_dr, 2.0f)*(sinf(alpha)+cosf(alpha)) -  powf(_dr, 2.0f)*neumann_bc(j)*2*cosf(alpha)*beta);
            // -powf(_dr*cosf(alpha),2.0f)*df_dr(j) + _dr/_m*sinf(alpha)*cosf(alpha)*df_da(i, j) -  neumann_bc(j)*(1+1.0f/(2*_m));
            //-powf(_dr, 2.0f)*_theta1*(powf(cosf(alpha), 3.0f) + powf(sinf(alpha), 2.0f)*cosf(alpha));
    }
}
*/


void Solver::populate_neumann()
{
    // fill the values for r = (M-1)*dr
    for(auto j: this->_neumann_indices)
    {
        this->_set_inner_eqns(_m-2, j);
    }
    
    // fill the values for r = M*dr
    for(auto j: this->_neumann_indices)
    {
        float alpha = (j+1)*_da;
        float ita = sinf(alpha)/(_m*_da);
        int i = _m-1;
        int r = i*_n;
        int k = r+p_j(j,i);
        
        (*_A)(k,k) = 25.0f/12.0f*cosf(alpha);
        
        int mapped_index = p_j(j+1,i);
        if(mapped_index != -1)
        {
            (*_A)(k,r+mapped_index) = -2.0f/3.0f*ita;
        }
        
        mapped_index = p_j(j-1,i);
        if(mapped_index != -1)
        {
            (*_A)(k,r+mapped_index) = 2.0f/3.0f*ita;
        }
        
        mapped_index = p_j(j+2,i);
        if(mapped_index != -1)
        {
            (*_A)(k,r+mapped_index) = 1.0f/12.0f*ita;
        }
        
        mapped_index = p_j(j-2,i);
        if(mapped_index != -1)
        {
            (*_A)(k,r+mapped_index) = -1.0f/12.0f*ita;
        }
        
        (*_A)(k,(i-1)*_n + j) = -4.0f*cosf(alpha);
        (*_A)(k,(i-2)*_n + j) = 3.0f*cosf(alpha);
        (*_A)(k,(i-3)*_n + j) = -4.0f/3.0f*cosf(alpha);
        (*_A)(k,(i-4)*_n + j) = 1.0f/4.0f*cosf(alpha);
        
        (*_b)(k) = _m*powf(_dr, 2.0f)*(sinf(alpha)*cosf(alpha) + powf(cosf(alpha), 2.0f) + 1); //_m*powf(_dr, 2.0f)
    }
}

float Solver::dirichlet_bc(int j)
{
    float res;
    if(j >= _dirichlet_indices.front() && j <= _dirichlet_indices.back())
    {
        float alpha = (j+1)*_da;
        res = powf(_R, 2.0f)*(sinf(alpha) + cosf(alpha));
        //res = 0.0f;
    }
    else
    {
        cout << "dirichlet bc: " << j << " - NaN" << endl;

        res = std::nanf("");
    }
    return res;
}

// for i = M-1 (as i = M is given) and j*_da in [pi/2, 3*pi/2]
void Solver::populate_dirichlet()
{
    for(auto j: _dirichlet_indices)
    {
        int i = _m-2;
        int r = i*_n;
        int k = r + j;
        float inv_da2 = powf((i+1)*_da, -2.0f);
        //float beta = powf((i+1)*_da, -2.0f);
        float ita = 1.0f/(2.0f*(i+1));
        // setting the k-th equation
        (*_A)(k,k) = -2.0f*(1+inv_da2);
        (*_A)(k,r+p_j(j+1)) = inv_da2;
        (*_A)(k,r+p_j(j-1)) = inv_da2;
        (*_A)(k,(i-1)*_n+j) = 1-ita;
        float alpha = (j+1)*_da;
        (*_b)(k) = (3.0f*powf(_dr, 2.0f)*(sinf(alpha)+cosf(alpha)) - (1+ita)*dirichlet_bc(j));
        // -powf(_dr,2.0f)*cosf(alpha)*df_dr(j) + _dr/(i+1)*sinf(alpha)*df_da(i,j) -  dirichlet_bc(j)*(1 + ita);
    }
}

//int d = (_m-1)*_n+(int)(_n/2)-1;

void Solver::_set_inner_eqns(int i, int j)
{
    int k = i*_n+j;
    float inv_da2 = powf((i+1)*_da, -2.0f);
    float ita = 1.0f/(2.0f*(i+1));
    // setting the k-th equation
    (*_A)(k,k) = -2.0f*(1+inv_da2);
    (*_A)(k,i*_n+p_j(j+1)) = inv_da2;
    (*_A)(k,i*_n+p_j(j-1)) = inv_da2;
    // when we increment dr we may step onto the boundary, so need to check
    (*_A)(k,(i+1)*_n+ p_j(j,(i+1))) = (1+ita);
    (*_A)(k,(i-1)*_n+j) = (1-ita);
    float alpha = (j+1)*_da;
    (*_b)(k) = 3*powf(_dr, 2.0f)*(sinf(alpha)+cosf(alpha));
        //-powf(_dr,2.0f)*cosf(alpha)*df_dr(j) + _dr/(i+1)*sinf(alpha)*df_da(i,j);
}

/*
void Solver::_set_inner_eqns(int i, int j)
{
    int k = i*_n+j;
    float inv_i2 = powf(i+1.0f, -2.0f);
    float inv_i = 1.0f/(i+1.0f);
    // setting the k-th equation
    (*_A)(k,k) = -2.5f*(1.0f+inv_i2);
    (*_A)(k,(i+2)*_n+j) = -1.0f/12.0f*(1.0f+inv_i);
    (*_A)(k,(i-2)*_n+j) = -1.0f/12.0f*(1.0f-inv_i);
    (*_A)(k,(i+1)*_n+j) = (4.0f/3.0f + 2.0f/3.0f*inv_i);
    (*_A)(k,(i-1)*_n+j) = (4.0f/3.0f - 2.0f/3.0f*inv_i);
    (*_A)(k,i*_n+p_j(j+2)) = -1.0f/12.0f*inv_i2;
    (*_A)(k,i*_n+p_j(j-2)) = -1.0f/12.0f*inv_i2;
    (*_A)(k,i*_n+p_j(j+1)) = 4.0f/3.0f*inv_i2;
    (*_A)(k,i*_n+p_j(j-1)) = 4.0f/3.0f*inv_i2;

    float alpha = (j+1)*_da;
    (*_b)(k) = 3*powf(_dr, 2.0f)*(sinf(alpha)+cosf(alpha));
        //-powf(_dr,2.0f)*cosf(alpha)*df_dr(j) + _dr/(i+1)*sinf(alpha)*df_da(i,j);
}*/

void Solver::populate_inner()
{
    // populate the full circles for r in [2*dr, (m-2)*dr]
    for(int i=1; i<_m-2; i++)
    {
        for(int j=0; j<_n; j++)
        {
            this->_set_inner_eqns(i,j);
        }
    }
}

void Solver::populate_origin()
{
    float inv_da2 = powf(_da, -2.0f);
    
    // i is 0
    for (int j=0; j<_n; j++)
    {
        (*_A)(j,j) = -(4.0f/3.0f + 2.0f*inv_da2);
        (*_A)(j,p_j(j+1)) = inv_da2;
        (*_A)(j,p_j(j-1)) = inv_da2;
        (*_A)(j,_n+j) = 1.25f;
        // reflect over the origin
        (*_A)(j,_n+p_j(j+(int)(_n/2))) = 1.0f/12;
        
        float alpha = (j+1)*_da;
        (*_b)(j) = 3*powf(_dr, 2.0f)*(sinf(alpha)+cosf(alpha));
            //1.5f*_dr*(-_dr*cosf(alpha)*df_dr(j) + sinf(alpha)*df_da(0, j));
    }
}

void Solver::populate_system()
{
    this->populate_origin();
    this->populate_inner();
    //this->populate_neumann();
    this->populate_dirichlet();
}

void Solver::print()
{
    _A->print("A:");
    _b->t().print("b:");
}

void Solver::save()
{
    _A->save("A.csv", csv_ascii);
    _b->save("b.csv", csv_ascii);
}
