#ifndef _Matrix_h_
#define _Matrix_h_

#include <AFEPack/BilinearOperator.h>
#include <AFEPack/Functional.h>
#include <AFEPack/Operator.h>
#include <AFEPack/TemplateElement.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/Geometry.h>
#include <AFEPack/HGeometry.h>
#include <AFEPack/DGFEMSpace.h>
#include "Options.h"

/*
template<typename value_type>
struct ElementCache : public ElementAdditionalData<value_type, DIM>
{
};
*/

// Define a Matrix based on the StiffMatrix in AFEPack to add the element cache to it to accelerate the computation.
template<typename value_type>
class Matrix : public StiffMatrix<DIM,double,DIM,DIM,value_type>
{
public:
  Matrix(){};
  Matrix(FEMSpace<double,DIM>& sp, std::vector<ElementAdditionalData<double,DIM>>& ec) :
    StiffMatrix<DIM,double,DIM,DIM,value_type>(sp), element_cache(ec){};
  virtual ~Matrix(){};
  
private:
  std::vector<ElementAdditionalData<double,DIM>> element_cache;
public:
  virtual void
  getElementMatrix(const Element<double,DIM>& element0,
		   const Element<double,DIM>& element1,
		   const ActiveElementPairIterator<DIM>::State state);
};

template<>
void Matrix<double>::getElementMatrix(const Element<double,DIM>& element0,
				      const Element<double,DIM>& element1,
				      const ActiveElementPairIterator<DIM>::State state)
{
  const std::vector<int>& ele_dof0 = element0.dof();
  const std::vector<int>& ele_dof1 = element1.dof();
  int n_element_dof0 = ele_dof0.size();
  int n_element_dof1 = ele_dof1.size();

  int ele_idx = element0.index();

  ElementAdditionalData<double,DIM>& the_ec = element_cache[ele_idx];

  const int& n_quadrature_point = the_ec.n_quadrature_point;
  for (int l = 0;l < n_quadrature_point;l ++)
    {
      for (int j = 0;j < n_element_dof0;j ++)
	{
	  for (int k = 0;k < n_element_dof1;k ++)
	    {
	      elementMatrix(j,k) += the_ec.Jxw[l]*innerProduct(the_ec.basis_gradient[j][l],the_ec.basis_gradient[k][l]);
	    }
	}
    }
}

template<>
void Matrix<float>::getElementMatrix(const Element<double,DIM>& element0,
				      const Element<double,DIM>& element1,
				      const ActiveElementPairIterator<DIM>::State state)
{
  const std::vector<int>& ele_dof0 = element0.dof();
  const std::vector<int>& ele_dof1 = element1.dof();
  int n_element_dof0 = ele_dof0.size();
  int n_element_dof1 = ele_dof1.size();

  int ele_idx = element0.index();

  ElementAdditionalData<double,DIM>& the_ec = element_cache[ele_idx];

  const int& n_quadrature_point = the_ec.n_quadrature_point;
  
  for (int l = 0;l < n_quadrature_point;l ++)
    {
      for (int j = 0;j < n_element_dof0;j ++)
	{
	  for (int k = 0;k < n_element_dof1;k ++)
	    {
	      elementMatrix(j,k) += float(the_ec.Jxw[l])*float(innerProduct(the_ec.basis_gradient[j][l], the_ec.basis_gradient[k][l]));
	    }
	}
    }
}
 
#endif
