#ifndef _mixed_precision_J4_h_
#define _mixed_precision_J4_h_

#include <lac/sparse_matrix.h>
#include <lac/sparsity_pattern.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <string>
#include <omp.h>
#include <type_traits>

#include <AFEPack/BilinearOperator.h>
#include <AFEPack/Functional.h>
#include <AFEPack/Operator.h>
#include <AFEPack/EasyMesh.h>
#include <AFEPack/TemplateElement.h>
#include <AFEPack/FEMSpace.h>
#include <AFEPack/Geometry.h>
#include <AFEPack/BoundaryCondition.h>
#include <AFEPack/HGeometry.h>
#include <AFEPack/DGFEMSpace.h>
#include <AFEPack/AMGSolver_new.h>
#include <AFEPack/AMGSolver_float.h>
#include <AFEPack/AMGSolver.h>

#include "infield_J4.h"
#include "Matrix.h"
#include "Options.h"
 
float float_u2d(const double * p)
{
  // Example 2 with J_3: float version for a special function u with two sharp in the field;
  return 50.0*(1-p[0]*p[0])*(1-p[1]*p[1])*exp(1-pow(p[1],-4));
} 


double double_u2d(const double * p)
{
  // Example 2 with J_3: double version for a special function u with two sharp in the field;
  return 50.0*(1-p[0]*p[0])*(1-p[1]*p[1])*exp(1-pow(p[1],-4));
}


float _f_float2d(const double * p)
{
  //define the x2 = x^2, y2 = y^2;
  double x2 = p[0]*p[0];
  double y2 = p[1]*p[1];
  
  return -50.0*( (2.0*p[1]*p[1]-2.0)*exp(1-pow(p[1],-4))+(2.0*p[0]*p[0]-2.0)*exp(1-pow(p[1], -4)) - 16.0*(1-p[0]*p[0])*exp(1-pow(p[1],-4))*pow(p[1],-4) - 20.0*(1-p[0]*p[0])*(1-p[1]*p[1])*exp(1-pow(p[1],-4))*pow(p[1],-6) + 16.0*(1-p[0]*p[0])*(1-p[1]*p[1])*exp(1-pow(p[1],-4))*pow(p[1],-10) );
}

double _f_double2d(const double * p)
{
  //define the x2 = x^2, y2 = y^2;
  double x2 = p[0]*p[0];
  double y2 = p[1]*p[1];

  return -50.0*( (2.0*p[1]*p[1]-2.0)*exp(1-pow(p[1],-4))+(2.0*p[0]*p[0]-2.0)*exp(1-pow(p[1], -4)) - 16.0*(1-p[0]*p[0])*exp(1-pow(p[1],-4))*pow(p[1],-4) - 20.0*(1-p[0]*p[0])*(1-p[1]*p[1])*exp(1-pow(p[1],-4))*pow(p[1],-6) + 16.0*(1-p[0]*p[0])*(1-p[1]*p[1])*exp(1-pow(p[1],-4))*pow(p[1],-10) );
}

 
//////////////////////////////////////////////////////////////

// value_type1: the precision in which to compute the primal solution.
// value_type2: the precision in which to compute the dual solution.
template<typename value_type1, typename value_type2>
class MixDWR
{
public:
  MixDWR();
  MixDWR(HGeometryTree<DIM>* _h_tree,
	 IrregularMesh<DIM>* _ir_mesh);

  /**
   * This function reads the input .mesh file to get the mesh info.
   */
  void readMesh(const std::string& fileName,
		const int& refine_times);

  /**
   * This function is used to generate mesh for the computation.
   */
  void genMesh(const std::string& fileName,
	       const int& refine_times);

  /**
   * This function is used to initialize the basis function according to 
   * different value_type.
   */
  void initialize_basis_fun();
  
  /**
   * This function is used to initialization, including read mesh 
   * data, build the template elent.
   */
  void initialize();
  
  /**
   * This function is used to build the finite element space for KS
   * equations to get the wavefunctions.
   */
  void buildspace();

  /**
   * This function is used to build the Element Cache to accelerate
   * the adapt mesh process.
   */
  void updateElementCache();

  /**
   * This function is used to compute the rhs for primal solution of the 
   * the i-th basis function at the l-th Gaussian Quadrature 
   * point on the ele-idx th element.
   */
  value_type1 get_part_RHS4prime(int ele_idx, int i, int l);
  
  /**
   * This function computes the rhs vector corresponding to the 
   * poisson equation of Hartree potential.
   */
  void getRHS4prime();

  /**
   * This function is used to compute the rhs for dual problem of the 
   * the i-th basis function at the l-th Gaussian Quadrature 
   * point on the ele-idx th element.
   */
  value_type2 get_part_RHS4dual(int ele_idx, int i, int l);

  /**
   * This function is used to compute the primal solution.
   */
  void gen_primal_AMG();

  /**
   * This function is used to build AMGSolver to compute the dual solution.
   */
  void gen_dual_AMG();

  /**
   * This function is used to compute the dual solution.
   */
  void gen_dual_solution();

  /**
   * This function is used to add the boundary condition of the prime problem
   * into the related stiff_matrix, rhs, and solution;
   */
  void addBoundaryCondition4prime();

  /**
   * This function is used to add the boundary condition of the dual problem
   * into the related stiff_matrix, rhs, and solution;
   */
  void addBoundaryCondition4dual();
  
  /**
   * Generate the solution by solving PDE.
   */
  void gen_prime_solution();

  /**
   * initialize the solution_PDE FEMFunction<value_type, DIM>*;
   */
  void init_solution_PDE();

  void printMesh();

  /**
   * This function solve the poisson equation to get u 
   */
  //void solve();

  /**
   * This function is used to build the matrix for solving the primal solution;
   */
  void buildMatrix4prime();

  /**
   * This function is used to build the matrix for solving the dual solution;
   */
  void buildMatrix4dual();

  /**
   * This function adapts the mesh with a designed indicators,
   *
   */
  void adaptMesh(double adapt_tol = 1.0e-03);

  /**
   * This function is used to solve the function w to get the 
   * error indicator.
   */
  void gen_solution_w();

  /**
   * This function is used to solve the dual problem of Point-value error
   * In the Page 31 Example 3.3 of the Adaptive book.
   */

  void init_element_cache()
  {
    std::cout<<"This is function init_element_cache();\n";
    element_cache.clear();
    const int& n_element = fem_space->n_element();
    element_cache.resize(n_element);
    std::cout<<"Finish initialize element cache!\n";
  };

  void write_prime_Data();

  value_type1 get_part_primal_error(int ele_idx, int l, value_type1 temp_primal_value);

  void get_prime_error();

  void get_dual_error();

  void set_amg_tolerance();


  double get_J_error(){return new_J_error;};

  double get_J_l2_error(){return new_J_error_l2norm;};

  int get_number_of_dof(){return number_of_dof;};
  
  bool is_need_post_con();
  
  void post_con();

  void getIndicator();

  Indicator<DIM> get_Indicator(){return ind;};

  std::vector<double> get_Jump(){return Jump;};

  
private:
  HGeometryTree<DIM> * h_tree;
  IrregularMesh<DIM> * irregular_mesh;
  IrregularMesh<DIM> * old_irregular_mesh;
  std::vector<ElementAdditionalData<double,DIM>> element_cache;

    
  DGFEMSpace<double,DIM> * fem_space;
  DGFEMSpace<double,DIM> * old_fem_space;
  
  //FEMFunction<value_type,DIM> * u_h;
  FEMFunction<double,DIM,DIM,DIM,value_type1> * prime_solution;
  FEMFunction<double,DIM,DIM,DIM,value_type2> * dual_solution;
  FEMFunction<double,DIM,DIM,DIM,double> * final_solution;
  
  Vector<value_type1> * rhs_type1;
  Vector<value_type2> * rhs_type2;
  
  NewMatrix<DIM,double,DIM,DIM,value_type1> * stiff_matrix_type1;
  NewMatrix<DIM,double,DIM,DIM,value_type2> * stiff_matrix_type2;
  NewMatrix<DIM,double,DIM,DIM,double> * final_stiff_matrix;
  
  // FEMFunction<value_type, DIM, DIM, DIM, value_type> * solution;

    // for regular template 
  TemplateGeometry<DIM> template_geometry;
  CoordTransform<DIM,DIM> coord_transform;
  TemplateDOF<DIM> template_dof;
  BasisFunctionAdmin<double,DIM,DIM> basis_function;
  UnitOutNormal<DIM> unit_out_normal;

  /// for twin_template
  TemplateGeometry<DIM> twin_template_geometry;
  CoordTransform<DIM, DIM> twin_coord_transform;
  TemplateDOF<DIM> twin_template_dof;
  BasisFunctionAdmin<double, DIM, DIM> twin_basis_function;
  UnitOutNormal<DIM> twin_unit_out_normal;

  /// for four_template
  TemplateGeometry<DIM> four_template_geometry;
  CoordTransform<DIM, DIM> four_coord_transform;
  TemplateDOF<DIM> four_template_dof;
  BasisFunctionAdmin<double, DIM, DIM> four_basis_function;
  UnitOutNormal<DIM> four_unit_out_normal;

  /// for boundary 
  TemplateGeometry<DIM-1> triangle_template_geometry;
  CoordTransform<DIM-1,DIM> triangle_to3d_coord_transform;

  TemplateGeometry<DIM-1> twin_triangle_template_geometry;
  CoordTransform<DIM-1,DIM> twin_triangle_to3d_coord_transform;
  
  std::vector<TemplateElement<double,DIM,DIM>> template_element;
  std::vector<TemplateDGElement<DIM-1,DIM> > edge_template_element;

  MeshAdaptor<DIM> * mesh_adaptor; 
  Indicator<DIM> ind;
  std::vector<double> Jump;
  
  double amg_tolerance4dual = 1.0e-8;
  double amg_tolerance4prime = 1.0e-8;
  //  double total_volume = 0.;
  double total_volume = 0.15;//4.0;

  double J_error = 0.;
  double J_error_l2norm = 0.;
  double new_J_error = 0.;
  double new_J_error_l2norm = 0.;

  int number_of_dof = 0.;

};

#define TEMPLATE template<typename value_type1, typename value_type2>
#define THIS MixDWR<value_type1,value_type2>

TEMPLATE
THIS::MixDWR()
{
  rhs_type1 = new Vector<value_type1>();
  rhs_type2 = new Vector<value_type2>();
  fem_space = new DGFEMSpace<double, DIM>();

  stiff_matrix_type1 = new NewMatrix<DIM,double,DIM,DIM,value_type1>();
  stiff_matrix_type2 = new NewMatrix<DIM,double,DIM,DIM,value_type2>();
  final_stiff_matrix = new NewMatrix<DIM,double,DIM,DIM,double>();
};

TEMPLATE
THIS::MixDWR(HGeometryTree<DIM>* _h_tree,
	     IrregularMesh<DIM>* _ir_mesh)
{
  h_tree = _h_tree;
  irregular_mesh = _ir_mesh;
  rhs_type1 = new Vector<value_type1>();
  rhs_type2 = new Vector<value_type2>();
  fem_space = new DGFEMSpace<double, DIM>();
  
  stiff_matrix_type1 = new NewMatrix<DIM,double,DIM,DIM,value_type1>();
  stiff_matrix_type2 = new NewMatrix<DIM,double,DIM,DIM,value_type2>();
  final_stiff_matrix = new NewMatrix<DIM,double,DIM,DIM,double>();
};

TEMPLATE
void THIS::readMesh(const std::string& fileName,
		    const int& refine_times)
{
  h_tree = new HGeometryTree<DIM>();
  h_tree->readMesh(fileName);

  irregular_mesh = new IrregularMesh<DIM>();
  irregular_mesh->reinit(*h_tree);
  irregular_mesh->globalRefine(refine_times);
  irregular_mesh->semiregularize();
  irregular_mesh->regularize(false);
};

TEMPLATE
void THIS::genMesh(const std::string& fileName,
		   const int& refine_times)
{
  h_tree = new HGeometryTree<DIM>();
  h_tree->readMesh(fileName);

  irregular_mesh = new IrregularMesh<DIM>();
  irregular_mesh->reinit(*h_tree);
 // irregular_mesh->globalRefine(refine_times);
 // irregular_mesh->semiregularize();
 // irregular_mesh->regularize(false);

#if 1
  irregular_mesh->globalRefine(refine_times);
#else
  AFEPack::Point<DIM> ori_pnt(0., 0., 0.);
  HTools tools;
  for(int k = 0;k < refine_times;++ k){
    IrregularMesh<DIM>::ActiveIterator
      the_ele = irregular_mesh->beginActiveElement(),
      end_ele = irregular_mesh_>endActiveElement();
    for(;the_ele != end_ele;++ the_ele){
      if(the_ele->isIncludePoint(ori_pnt)){
        the_ele->value = 1;
        tools.setGeometryUnusedRecursively(*(the_ele->h_element));
        the_ele->refine();
        
        const int& n_child = the_ele->n_child;
        for(int i = 0;i < n_child;++ i){
          the_ele->child[i]->value=0;
          tools.setGeometryUsed(*(the_ele->child[i]->h_element));
        }

        break;
      }
    }
  }
  irregular_mesh->globalRefine(refine_times);  
#endif
  
  irregular_mesh->semiregularize();
  irregular_mesh->regularize(false);

}


TEMPLATE
void THIS::initialize()
{
  if(DIM==3){
  template_geometry.readData("tetrahedron.tmp_geo");
  coord_transform.readData("tetrahedron.crd_trs");
  unit_out_normal.readData("tetrahedron.out_nrm");
  template_dof.reinit(template_geometry);

  twin_template_geometry.readData("twin_tetrahedron.tmp_geo");
  twin_coord_transform.readData("twin_tetrahedron.crd_trs");
  twin_unit_out_normal.readData("twin_tetrahedron.out_nrm");
  twin_template_dof.reinit(twin_template_geometry);

  four_template_geometry.readData("four_tetrahedron.tmp_geo");
  four_coord_transform.readData("four_tetrahedron.crd_trs");
  four_unit_out_normal.readData("four_tetrahedron.out_nrm");
  four_template_dof.reinit(four_template_geometry);

  // initialize the basis function according to the value_type through the
  // functions.
  //initialize_basis_fun();
  template_dof.readData("tetrahedron.1.tmp_dof");
  basis_function.reinit(template_dof);
  basis_function.readData("tetrahedron.1.bas_fun");

  twin_template_dof.readData("twin_tetrahedron.1.tmp_dof");
  twin_basis_function.reinit(twin_template_dof);
  twin_basis_function.readData("twin_tetrahedron.1.bas_fun");

  four_template_dof.readData("four_tetrahedron.1.tmp_dof");
  four_basis_function.reinit(four_template_dof);
  four_basis_function.readData("four_tetrahedron.1.bas_fun");  
  
  template_element.resize(3);
  template_element[0].reinit(template_geometry,
			     template_dof,
			     coord_transform,
			     basis_function,
			     unit_out_normal);

  template_element[1].reinit(twin_template_geometry,
			     twin_template_dof,
			     twin_coord_transform,
			     twin_basis_function,
			     twin_unit_out_normal);

  template_element[2].reinit(four_template_geometry,
			     four_template_dof,
			     four_coord_transform,
			     four_basis_function,
			     four_unit_out_normal);

  triangle_template_geometry.readData("triangle.tmp_geo");
  triangle_to3d_coord_transform.readData("triangle.to3d.crd_trs");

  twin_triangle_template_geometry.readData("twin_triangle.tmp_geo");
  twin_triangle_to3d_coord_transform.readData("twin_triangle.to3d.crd_trs");

  edge_template_element.resize(2);
  edge_template_element[0].reinit(triangle_template_geometry,
				  triangle_to3d_coord_transform);
  edge_template_element[1].reinit(twin_triangle_template_geometry,
				  twin_triangle_to3d_coord_transform);
  }

  if(DIM==2){
  template_geometry.readData("triangle.tmp_geo");
  coord_transform.readData("triangle.crd_trs");
  unit_out_normal.readData("triangle.out_nrm");
  template_dof.reinit(template_geometry);

  twin_template_geometry.readData("twin_triangle.tmp_geo");
  twin_coord_transform.readData("twin_triangle.crd_trs");
  twin_unit_out_normal.readData("twin_triangle.out_nrm");
  twin_template_dof.reinit(twin_template_geometry);

  //initialize_basis_fun();
  template_dof.readData("triangle.1.tmp_dof");
  basis_function.reinit(template_dof);
  basis_function.readData("triangle.1.bas_fun");

  twin_template_dof.readData("twin_triangle.1.tmp_dof");
  twin_basis_function.reinit(twin_template_dof);
  twin_basis_function.readData("twin_triangle.1.bas_fun");

  template_element.resize(2);
  template_element[0].reinit(template_geometry,
                             template_dof,
                             coord_transform,
                             basis_function,
                             unit_out_normal);

  template_element[1].reinit(twin_template_geometry,
                             twin_template_dof,
                             twin_coord_transform,
                             twin_basis_function,
                             twin_unit_out_normal);

  triangle_template_geometry.readData("interval.tmp_geo");
  triangle_to3d_coord_transform.readData("interval.to2d.crd_trs");
  edge_template_element.resize(1);
  edge_template_element[0].reinit(triangle_template_geometry,
		  		  triangle_to3d_coord_transform);
  }

}


TEMPLATE
void THIS::buildspace()
{
  std::cout<<"Begin to build fem space!\n";
  RegularMesh<DIM>& regular_mesh = irregular_mesh->regularMesh();
  //std::cout<<"build a regular mesh!\n";
  fem_space = new DGFEMSpace<double,DIM>(); 
  fem_space->reinit(regular_mesh, template_element, edge_template_element);
  // std::cout<<"reinitialzie the fem space!\n";
  int n_element = regular_mesh.n_geometry(DIM);
  fem_space->element().resize(n_element);
  for (u_int i = 0;i < n_element;++ i)
    {
      GeometryBM& the_geo = regular_mesh.geometry(DIM, i);
      const int& n_vtx = the_geo.n_vertex();
      
      if(DIM==3){
      switch(n_vtx){
      case 4:
	fem_space->element(i).reinit(*fem_space, i, 0);
	break;
      case 5:
	fem_space->element(i).reinit(*fem_space, i, 1);
	break;
      case 7:
	fem_space->element(i).reinit(*fem_space, i, 2);
	break;
      default:
	std::cout<<"there is something wrong with template_element" << std::endl;
	getchar();
      }

      }
      //fem_space->element(i).reinit(*fem_space, i, 0);
      ////////////////////////////////////////////////////////

      if(DIM==2){
      switch(n_vtx){
      case 3:
	fem_space->element(i).reinit(*fem_space, i, 0);
	break;
      case 4:
	fem_space->element(i).reinit(*fem_space, i, 1);
	break;
      default:
	std::cout<<"there is something wrong with template_element" << std::endl;
	getchar();
      }

      }
    }
  
  fem_space->buildElement();
  fem_space->buildDof();
  fem_space->buildDofBoundaryMark();
  std::cout << "building fem space is done..." << std::endl;

  // for 2D edge elements 
  u_int n_edge = regular_mesh.n_geometry(DIM-1);
  fem_space->dgElement().resize(n_edge);

  for(u_int i = 0;i < n_edge;++ i)
    {
      GeometryBM& the_geo = regular_mesh.geometry(DIM - 1, i);
      const int& n_vtx = the_geo.n_vertex();

      if(DIM==3){
      switch(n_vtx){
      case 3:
	fem_space->dgElement(i).reinit(*fem_space, i, 0);
	break;
      case 4:
	fem_space->dgElement(i).reinit(*fem_space, i, 1);
	break;
      default:
	std::cout << "there is something wrong with edge_template_element" <<std::endl;
	getchar();
      }

      }
    //  fem_space->dgElement(i).reinit(*fem_space,i,0);
    ///////////////////////////////////////////////////////

      if(DIM==2){
      switch(n_vtx){
      case 2:
	fem_space->dgElement(i).reinit(*fem_space, i, 0);
	break;
      default:
        std::cout << "there is something wrong with edge_template_element" <<std::endl;
        getchar();
      }

      }
    }
  fem_space->buildDGElement();
  std::cout<<"fem space is built..."<<std::endl;
}


template<>
void MixDWR<double,float>::set_amg_tolerance()
{
  amg_tolerance4prime = 1.0e-15;
  amg_tolerance4dual = 1.0e-8;
}

template<>
void MixDWR<float,double>::set_amg_tolerance()
{
  amg_tolerance4prime = 1.0e-8;
  amg_tolerance4dual = 1.0e-15;
}


TEMPLATE
void THIS::updateElementCache()
{
  //init_element_cache();
  std::cout<<"Begin to update the element cache..."<<std::endl;
  element_cache.clear();
  const int& n_element = fem_space->n_element();
  element_cache.resize(n_element);
  const Mesh<DIM>& r_mesh = fem_space->mesh();


  //#pragma omp parallel for
  for(int i = 0;i < n_element;++ i){
    Element<double, DIM>& the_ele = fem_space->element(i);
    const int& ele_idx = the_ele.index();

    ElementAdditionalData<double,DIM>& the_ec = element_cache[ele_idx];
    GeometryBM& the_geo = the_ele.geometry();

    const int& n_vtx = the_geo.n_vertex();
    std::vector<AFEPack::Point<DIM> > vertex(n_vtx);
    for(int j = 0;j < n_vtx;++ j){
      vertex[j] = r_mesh.point()[the_geo.vertex()[j]];
    }
  
    //std::cout<<"Begin to compute quadrature points Flag1!!!\n"; 
    int& n_quadrature_point = the_ec.n_quadrature_point;
    double& volume = the_ec.volume;
    AFEPack::Point<DIM>& bc = the_ec.bc;
    std::vector<double>& Jxw = the_ec.Jxw;
    
    std::vector<AFEPack::Point<DIM> >& q_point = the_ec.q_point;

    
    std::vector<std::vector<double> >& basis_value = the_ec.basis_value;
    
    std::vector<std::vector<std::vector<double> > >& basis_gradient = the_ec.basis_gradient;

    barycenter(vertex, bc);

    double Tvolume = the_ele.templateElement().volume();
    const QuadratureInfo<DIM>& quad_info = the_ele.findQuadratureInfo(ACC);

    std::vector<double> jacobian = the_ele.local_to_global_jacobian(quad_info.quadraturePoint());
    
    n_quadrature_point = quad_info.n_quadraturePoint();
    q_point = the_ele.local_to_global(quad_info.quadraturePoint());

    basis_value = the_ele.basis_function_value(q_point);
    basis_gradient = the_ele.basis_function_gradient(q_point);

    volume = 0.;
    Jxw.resize(n_quadrature_point);
    for (int l = 0;l < n_quadrature_point;l ++) {
      Jxw[l] = quad_info.weight(l)*jacobian[l]*Tvolume;
      volume += Jxw[l];      
    }
  }
  std::cout<<"Finish updating the ElementCache!"<<std::endl;
}

template<>
double MixDWR<double,float>::get_part_RHS4prime(int ele_idx, int i, int l)
{
  double f_val = _f_double2d(element_cache[ele_idx].q_point[l]);
  double rhs_value = 0.;
  rhs_value = element_cache[ele_idx].Jxw[l]*(f_val*element_cache[ele_idx].basis_value[i][l]);
  return rhs_value;
}

template<>
float MixDWR<float,double>::get_part_RHS4prime(int ele_idx, int i, int l)
{
  float f_val = _f_float2d(element_cache[ele_idx].q_point[l]);
  float rhs_value = 0.;
  rhs_value = float(element_cache[ele_idx].Jxw[l])*(f_val*float(element_cache[ele_idx].basis_value[i][l]));
  return rhs_value;
}

template<>
void MixDWR<float,double>::gen_primal_AMG()
{
  AMGSolver_float solver;
  solver.lazyReinit(*stiff_matrix_type1);

  set_amg_tolerance();
  
  solver.solve(*prime_solution, *rhs_type1, amg_tolerance4prime, amg_iter);
}

template<>
void MixDWR<double,float>::gen_primal_AMG()
{
  AMGSolver solver;
  solver.lazyReinit(*stiff_matrix_type1);

  set_amg_tolerance();
  
  solver.solve(*prime_solution, *rhs_type1, amg_tolerance4prime, amg_iter);
}




TEMPLATE
void THIS::getRHS4prime()
{
  std::cout<<"Begin to build RHS vector..."<<std::endl;
  rhs_type1 = new Vector<value_type1>();
  rhs_type1->reinit(fem_space->n_dof());
  rhs_type1->Vector<value_type1>::operator = (0.0); 

  const int& n_element = fem_space->n_element();

  RegularMesh<DIM>& r_mesh = irregular_mesh->regularMesh();
  
  for(int j = 0;j < n_element;++ j)
    {
      const int& ele_idx = j;
      Element<double, DIM> the_ele = fem_space->element(ele_idx);

      const int& n_quadrature_point = element_cache[ele_idx].n_quadrature_point;
      const std::vector<int>& ele_dof = the_ele.dof();
      u_int n_ele_dof = ele_dof.size();
      for (u_int l = 0;l < n_quadrature_point;++ l)
	{
	  //double f_val = _f_double2d(element_cache[ele_idx].q_point[l]);
	  for(u_int i = 0;i < n_ele_dof;++ i)
	    {
	      (*rhs_type1)(ele_dof[i]) += get_part_RHS4prime(ele_idx, i, l);
	    }
	  
	}
      
    }
}

template<>
float MixDWR<double,float>::get_part_RHS4dual(int ele_idx, int i, int l)
{
  float rhs_value = 0.;
  rhs_value = float(element_cache[ele_idx].Jxw[l])*float(element_cache[ele_idx].basis_value[i][l]);
  return rhs_value;
}

template<>
double MixDWR<float,double>::get_part_RHS4dual(int ele_idx, int i, int l)
{
  double rhs_value = 0.;
  rhs_value = element_cache[ele_idx].Jxw[l]*element_cache[ele_idx].basis_value[i][l];
  return rhs_value;
}

template<>
void MixDWR<double,float>::gen_dual_AMG()
{
  AMGSolver_float solver;
  solver.lazyReinit(*stiff_matrix_type2);

  set_amg_tolerance();
  solver.solve(*dual_solution, *rhs_type2, amg_tolerance4dual, amg_iter);
}

template<>
void MixDWR<float,double>::gen_dual_AMG()
{
  AMGSolver solver;
  solver.lazyReinit(*stiff_matrix_type2);

  set_amg_tolerance();
  solver.solve(*dual_solution, *rhs_type2, amg_tolerance4dual, amg_iter);
}

TEMPLATE
void THIS::gen_dual_solution()
{
  RegularMesh<DIM>& r_mesh = irregular_mesh->regularMesh();
  dual_solution = new FEMFunction<double, DIM, DIM, DIM, value_type2>(*fem_space);
  rhs_type2 = new Vector<value_type2>();
  rhs_type2->reinit(fem_space->n_dof());
  rhs_type2->Vector<value_type2>::operator = (0.0);

  buildMatrix4dual();
  
  SparsityPattern& sp = stiff_matrix_type2->getSparsityPattern();
  const size_t * row_start = sp.get_rowstart_indices();
  const unsigned int * column = sp.get_column_numbers();
  
  const int& n_element = fem_space->n_element();

  total_volume = 0.;

  for(int i = 0;i < n_element;i ++)
    {
      const int& ele_idx = i;
      Element<double, DIM>the_ele = fem_space->element(ele_idx);

      const int& n_quadrature_point = element_cache[ele_idx].n_quadrature_point;
      const std::vector<int>& ele_dof = the_ele.dof();
      u_int n_ele_dof = ele_dof.size();

      int is_in_field = 0;

      
      for(u_int j = 0;j < n_ele_dof;++ j)
	{
	  GeometryBM& pnt_geo = r_mesh.geometry(0, ele_dof[j]);
	  AFEPack::Point<DIM>& pnt = r_mesh.point(pnt_geo.vertex()[0]);
	 
	  if(infield(pnt))
	    {
	      total_volume += element_cache[ele_idx].volume;

	      is_in_field = 1;
	      
	      break;
	    }
	}
      
      int col_idx = 0;
      for(u_int j = 0;j < n_ele_dof;++ j)
	{
	  GeometryBM&pnt_geo = r_mesh.geometry(0, ele_dof[j]);
	  AFEPack::Point<DIM>& pnt = r_mesh.point(pnt_geo.vertex()[0]);
	
	  //if(infield(pnt))
	  if(is_in_field == 1)
	    {
	      
	      for(u_int l = 0;l < n_quadrature_point;++ l)
		{
		  (*rhs_type2)(ele_dof[j]) += get_part_RHS4dual(ele_idx, j, l);
		  
		}

	    }     
	}
    }

      const int total_n_dof = fem_space->n_dof();

    int col_idx = 0;
    std::cout<<"The total dof of the fem space is: "<<total_n_dof<<std::endl;
    for(u_int j = 0;j < total_n_dof;++ j)
      {
	GeometryBM& pnt_geo = r_mesh.geometry(0, j);
	AFEPack::Point<DIM>& pnt = r_mesh.point(pnt_geo.vertex()[0]);

	if(!infield(pnt))
	  {
	    (*rhs_type2)(j) = 0.;
	  }
      }
    // This total_volume is computed by the sum of the elements which are in the field. That is not accurate enough.
    // Thus I directly use the exact volume of the interested domain.
  std::cout<<"ATTENTION:::::::::::::The total volume of the dual problem solving domain is: "<<total_volume<<std::endl;
  
  total_volume = 0.15;
  
  for(int i = 0;i < rhs_type2->size();i ++)
    {
      (*rhs_type2)(i) /= total_volume;
    }
  
  BoundaryFunction<double, DIM, DIM, DIM, value_type2> boundary;
  boundary.reinit(BoundaryConditionInfo::DIRICHLET, 1, &double_u2d);
  BoundaryConditionAdmin<double, DIM, DIM, DIM, value_type2> boundary_admin(*fem_space);
  boundary_admin.add(boundary);
  boundary_admin.apply(*stiff_matrix_type2, *dual_solution, *rhs_type2);
  
  
  gen_dual_AMG();

  std::ofstream write_dual;
  write_dual.open("dual.txt");
  for(int i = 0;i < dual_solution->size();i ++)
    {
      write_dual<<(*dual_solution)[i]<<std::endl;
    }
  write_dual.close();
  
  //dual_solution->writeOpenDXData("dual_solution.dx");
  std::cout<<"Finish solving the dual problem!"<<std::endl;
}


TEMPLATE
void THIS::buildMatrix4prime()
{
  std::cout << "build a new stiff_matrix..." << std::endl;
  delete stiff_matrix_type1;

  stiff_matrix_type1 = new NewMatrix<DIM,double,DIM,DIM,value_type1>(*fem_space, element_cache);
  
  stiff_matrix_type1->algebricAccuracy() = ACC;
  stiff_matrix_type1->build();
  std::cout << "build stiff_matrix is done..."<<std::endl; 
}

TEMPLATE
void THIS::buildMatrix4dual()
{
  std::cout << "build a new stiff_matrix..." << std::endl;
  delete stiff_matrix_type2;

  stiff_matrix_type2 = new NewMatrix<DIM,double,DIM,DIM,value_type2>(*fem_space, element_cache);

  stiff_matrix_type2->algebricAccuracy() = ACC;
  stiff_matrix_type2->build();
  std::cout << "build stiff_matrix is done..."<<std::endl;  
}

template<>
void MixDWR<double, float>::write_prime_Data()
{
  std::cout<<"write prime solution data!\n";
  prime_solution->writeOpenDXData("prime_solution_double.dx");
  std::cout<<"The writeOpenDXData() is finished..."<<std::endl;  
}

template<>
void MixDWR<float, double>::write_prime_Data()
{
  std::cout<<"write prime solution data!\n";
  prime_solution->writeOpenDXData("prime_solution_float.dx");
  std::cout<<"The writeOpenDXData() is finished..."<<std::endl;  
}



////////////
// Get part of the prime error.
template<>
float MixDWR<float,double>::get_part_primal_error(int ele_idx, int l, float temp_primal_value)
{
  return float(float_u2d(element_cache[ele_idx].q_point[l]) - float(temp_primal_value));
}

template<>
double MixDWR<double,float>::get_part_primal_error(int ele_idx, int l, double temp_primal_value)
{
  return double_u2d(element_cache[ele_idx].q_point[l]) - double(temp_primal_value);
}

TEMPLATE
void THIS::get_prime_error()
{
  value_type1 error = 0.;

  const int& n_element = fem_space->n_element();
  for(int i = 0;i < n_element;i ++)
    {
      const int& ele_idx = i;
      Element<double,DIM> the_ele = fem_space->element(ele_idx);

      const int& n_quadrature_point = element_cache[ele_idx].n_quadrature_point;
      std::vector<double> prime_value = prime_solution->value(element_cache[ele_idx].q_point, the_ele);

      const std::vector<int>& ele_dof = the_ele.dof();
      u_int n_ele_dof = ele_dof.size();
      
      for(int l = 0;l < n_quadrature_point;l ++)
	{
	  value_type1 df_value = get_part_primal_error(ele_idx, l, prime_value[l]);
	  error += float(element_cache[ele_idx].Jxw[l])*df_value*df_value;
	}
    }
  error = sqrt(fabs(error));
 
  std::cerr << "\nL2 error = " << error << std::endl;
}

/*
template<>
void MixDWR<float,double>::get_prime_error()
{
  float error = 0.;

  const int& n_element = fem_space->n_element();
  for(int i = 0;i < n_element;i ++)
    {
      const int& ele_idx = i;
      Element<double,DIM> the_ele = fem_space->element(ele_idx);

      const int& n_quadrature_point = element_cache[ele_idx].n_quadrature_point;
      std::vector<double> prime_value = prime_solution->value(element_cache[ele_idx].q_point, the_ele);

      const std::vector<int>& ele_dof = the_ele.dof();
      u_int n_ele_dof = ele_dof.size();
      
      for(int l = 0;l < n_quadrature_point;l ++)
	{
	  //float Jxw_value = Jxw[l];
	  float df_value = float_u2d(element_cache[ele_idx].q_point[l]) - float(prime_value[l]);
	  error += float(element_cache[ele_idx].Jxw[l])*df_value*df_value;
	}
    }
  error = sqrt(fabs(error));
 
  std::cerr << "\nL2 error = " << error << std::endl;
}
template<>
void MixDWR<double,float>::get_prime_error()
{
  double error = 0.;

  const int& n_element = fem_space->n_element();
  for(int i = 0;i < n_element;i ++)
    {
      const int& ele_idx = i;
      Element<double,DIM> the_ele = fem_space->element(ele_idx);

      const int& n_quadrature_point = element_cache[ele_idx].n_quadrature_point;
      std::vector<double> prime_value = prime_solution->value(element_cache[ele_idx].q_point, the_ele);

      const std::vector<int>& ele_dof = the_ele.dof();
      u_int n_ele_dof = ele_dof.size();
      
      for(int l = 0;l < n_quadrature_point;l ++)
	{
	  //double Jxw_value = Jxw[l];
	  double df_value = double_u2d(element_cache[ele_idx].q_point[l]) - double(prime_value[l]);
	  error += element_cache[ele_idx].Jxw[l]*df_value*df_value;
	}
    }
  error = sqrt(fabs(error));
 
  
  std::cerr << "\nL2 error = " << error << std::endl;
}
*/

template<>
void MixDWR<double,float>::addBoundaryCondition4prime()
{
  BoundaryFunction<double, DIM, DIM, DIM, double> boundary;
  boundary.reinit(BoundaryConditionInfo::DIRICHLET, 1, &double_u2d);

  BoundaryConditionAdmin<double, DIM, DIM, DIM, double> boundary_admin(*fem_space);
  boundary_admin.add(boundary);
  boundary_admin.apply(*stiff_matrix_type1, *prime_solution, *rhs_type1);
}

template<>
void MixDWR<float,double>::addBoundaryCondition4prime()
{
  BoundaryFunction<double, DIM, DIM, DIM, float> boundary;
  boundary.reinit(BoundaryConditionInfo::DIRICHLET, 1, &double_u2d);

  BoundaryConditionAdmin<double, DIM, DIM, DIM, float> boundary_admin(*fem_space);
  boundary_admin.add(boundary);
  boundary_admin.apply(*stiff_matrix_type1, *prime_solution, *rhs_type1);
}

TEMPLATE
void THIS::gen_prime_solution()
{
  std::cout<<"Begin to compute the primal solution..."<<std::endl;
  
  double time_start = 0.;
  time_start = omp_get_wtime();
  prime_solution = new FEMFunction<double,DIM,DIM,DIM,value_type1>(*fem_space);


  buildMatrix4prime();
  
  getRHS4prime();

  addBoundaryCondition4prime();


  gen_primal_AMG();
  
  std::cout<<"The mean_value of the rhs_type1 is: "<<rhs_type1->mean_value()<<std::endl;

  std::cout<<"\nThe CPU time for solving the problem is: "<< omp_get_wtime() - time_start <<std::endl;

  write_prime_Data();
  get_prime_error();
}


TEMPLATE
void THIS::adaptMesh(double adapt_tol)
{
  std::cout<<"Begin to adapt mesh!"<<std::endl;
  mesh_adaptor = new MeshAdaptor<DIM>(*irregular_mesh);

  mesh_adaptor->convergenceOrder() = 1;
  mesh_adaptor->refineStep() = 1;
  mesh_adaptor->setIndicator(ind);
  mesh_adaptor->tolerence() = adapt_tol;
  
  mesh_adaptor->is_refine_only() = true;
  mesh_adaptor->adapt();

  irregular_mesh->semiregularize();
  irregular_mesh->regularize(false);

  std::cout<<"The adapt tolerance is::::::::: "<<adapt_tol<<std::endl;
}

/*
template<>
void MixDWR<float,double>::adaptMesh(double adapt_tol)
{
  std::cout<<"Begin to adapt mesh!"<<std::endl;
  mesh_adaptor = new MeshAdaptor<DIM>(*irregular_mesh);

  mesh_adaptor->convergenceOrder() = 1;
  mesh_adaptor->refineStep() = 1;
  mesh_adaptor->setIndicator(ind);
  mesh_adaptor->tolerence() = adapt_tol;
  //mesh_adaptor->tolerence() = 1.0e-5;
  mesh_adaptor->is_refine_only() = true;
  mesh_adaptor->adapt();

  irregular_mesh->semiregularize();
  irregular_mesh->regularize(false);

  std::cout<<"The adapt tolerance is::::::::: "<<adapt_tol<<std::endl;

}

template<>
void MixDWR<double,float>::adaptMesh(double adapt_tol)
{
  //std::ofstream write;
  //write.open("res_and_dual.txt");
  mesh_adaptor = new MeshAdaptor<DIM>(*irregular_mesh);
  mesh_adaptor->convergenceOrder() = 1;
  mesh_adaptor->refineStep() = 1;
  mesh_adaptor->setIndicator(ind);
  mesh_adaptor->tolerence() = adapt_tol;

  mesh_adaptor->is_refine_only() = true;
  mesh_adaptor->adapt();

  irregular_mesh->semiregularize();
  irregular_mesh->regularize(false);

  std::cout<<"The adapt tolerance is::::::::: "<<adapt_tol<<std::endl;
}
*/

TEMPLATE
void THIS::get_dual_error()
{
  J_error = 0.;
  J_error_l2norm = 0.;
  new_J_error = 0.;
  new_J_error_l2norm = 0.;
  //double J_error_per_ele = 0.;
  
  RegularMesh<DIM>& r_mesh = irregular_mesh->regularMesh();
  const int& n_element = fem_space->n_element();
  std::cout<<"The number of the Elements is:::: "<<n_element<<std::endl;

  
  const int& n_dof = fem_space->n_dof();
  std::cout<<"The number of the dof of the space is:::: "<<n_dof<<std::endl;
  number_of_dof = n_dof;
  
  for(int i = 0;i < n_element;i ++)
    {
      const int& ele_idx = i;
      Element<double,DIM> the_ele = fem_space->element(ele_idx);
      const int& n_quadrature_point = element_cache[ele_idx].n_quadrature_point;
      std::vector<double> u_h_value = prime_solution->value(element_cache[ele_idx].q_point, the_ele);

      const std::vector<int>& ele_dof = the_ele.dof();
      u_int n_ele_dof = ele_dof.size();
      
      double temp_J_error = 0.;
      double temp_error_per = 0.;
      double temp_new = 0.;

      for(u_int j = 0;j < n_ele_dof;++ j)
      {
        GeometryBM& pnt_geo = r_mesh.geometry(0, ele_dof[j]);
        AFEPack::Point<DIM>& pnt = r_mesh.point(pnt_geo.vertex()[0]);
	  if(infield(pnt))
	    {
	      for(u_int l = 0;l < n_quadrature_point;++ l)
		{
		  new_J_error += element_cache[ele_idx].Jxw[l] * fabs(double_u2d(element_cache[ele_idx].q_point[l]) - u_h_value[l]);
		}
	      break;
	    }
      }
      //J_error_per_ele += fabs(temp_error_per/the_ec.volume);    
    }
  std::cout<<"The total volume is: "<<total_volume<<std::endl;
  
  std::cout<<"The new L2-error J(e) of dual problem is::: "<<fabs(new_J_error)<<std::endl;//*total_volume<<std::endl;
  std::cout<<"The new l2-norm error J(e) of dual problem is::: "<<fabs(new_J_error)/sqrt(total_volume)<<std::endl;
  new_J_error_l2norm = fabs(new_J_error)*sqrt(total_volume);
  new_J_error = fabs(new_J_error);  
}

template<>
bool MixDWR<double,float>::is_need_post_con()
{
  return false;
}
template<>
bool MixDWR<float,double>::is_need_post_con()
{
  return true;
}

TEMPLATE
void THIS::getIndicator()
{
  //double time_get_indicator = 0.;
  //time_get_indicator = omp_get_wtime();
  
  RegularMesh<DIM>& r_mesh = irregular_mesh->regularMesh();
  //////////////////////////////////////////////////////////////
  // Begin  to compute the gradient Jump and update the vector Jump;
  Jump.clear();
  Jump.resize(r_mesh.n_geometry(DIM),0.0);

  const int& n_dgEle = fem_space->n_DGElement();

  for(int i = 0;i < n_dgEle;++ i)
    {
      std::vector<double> gradient0(DIM), gradient1(DIM);
      std::vector<double> un(DIM);

    
      DGElement<double, DIM>& the_dgEle = fem_space->dgElement(i);
      const int& dgEle_idx = the_dgEle.index();

      // Only use DG element once, so it might more efficient to compute directly;
      //DGElementAdditionalData<double, DIM>& dg_ec = dgElement_cache[dgEle_idx];
      
      //////////////////////////////////////////////////////
      double template_dg_vol = the_dgEle.templateElement().volume();
      double dg_vol = 0.;

      /// 积分公式，注意它的积分点的维数少一啊
      const QuadratureInfo<DIM-1>& qi = the_dgEle.findQuadratureInfo(ACC);
      u_int n_q_pnt = qi.n_quadraturePoint();

      /// 变换的雅可比行列式
      std::vector<double> jac = the_dgEle.local_to_global_jacobian(qi.quadraturePoint());

      /// 积分点变换到网格中去
      std::vector<AFEPack::Point<DIM> > q_pnt = the_dgEle.local_to_global(qi.quadraturePoint());

      for(int l = 0;l < n_q_pnt;l ++)
	{
	  dg_vol += template_dg_vol * qi.weight(l) * jac[l];
	}
      

      /// 和它邻接的两个体单元的指针
      Element<double,DIM> * p_neigh0 = the_dgEle.p_neighbourElement(0);
      Element<double,DIM> * p_neigh1 = the_dgEle.p_neighbourElement(1);

      /// 法向单位向量，对 0 号邻居来说是外法向
      std::vector<std::vector<double> > unit_normal = unitOutNormal(q_pnt, *p_neigh0, the_dgEle);
      un = unit_normal[0];
      /////////////////////////////////////////////
      //un = dg_ec.unit_normal[0];

      Element<double, DIM> * neighbor0 = the_dgEle.p_neighbourElement(0);
      const int& neighbor0_idx = neighbor0->index();
      ElementAdditionalData<double, DIM>& neighbor0_ec = element_cache[neighbor0_idx];
      AFEPack::Point<DIM>& neighbor0_bc = neighbor0_ec.bc;
      gradient0 = prime_solution->gradient(neighbor0_bc, *neighbor0);

      if(the_dgEle.p_neighbourElement(1) != NULL)
	{
	  Element<double, DIM> * neighbor1 = the_dgEle.p_neighbourElement(1);
	  const int& neighbor1_idx = neighbor1->index();
	  ElementAdditionalData<double, DIM>& neighbor1_ec = element_cache[neighbor1_idx];
	  AFEPack::Point<DIM>& neighbor1_bc = neighbor1_ec.bc;
	  gradient1 = prime_solution->gradient(neighbor1_bc, *neighbor1);
	  
	}
      else
	{
	  for(int k = 0;k < DIM;++ k)
	    {
	      gradient1[k] = 0.;
	      //gradient1[k] = gradient0[k];
	    }
	}
      
    /// now we have gradient0, gradient1, un, and volume
    double indicator = 0;

    for(int k = 0;k < DIM;++ k)
      {
	// original indicator:
	//indicator += pow((gradient0[k] - gradient1[k]) * un[k] ,2.);

	// new
	indicator += (gradient0[k] - gradient1[k]) * un[k];
      }

    // This is original code
    //indicator = 0.5 * sqrt(dg_vol * indicator);
    
    // For the indicator in book.
    double h_E = 0.; // the diameter of the element K.
    //h_E = 2.0 * sqrt(dg_vol / (PI));

    /*
    if(DIM == 2)
      {
	h_E = 1.0;
      }

      */

    indicator = 0.5 * dg_vol * (dg_vol * pow(indicator,2.));
    //indicator = 0.5 * (dg_vol * pow(indicator,2.));
    
    Jump[neighbor0_idx] += indicator;
    if(the_dgEle.p_neighbourElement(1) != NULL)
      {
	Element<double, DIM> * neighbor1 = the_dgEle.p_neighbourElement(1);
	const int& neighbor1_idx = neighbor1->index();
	
	Jump[neighbor1_idx] += indicator;
      }
    }

  //////////////////////////////////////////////////////////////

  ind.reinit(r_mesh);
  const int& n_element = fem_space->n_element();
  std::cout<<"Computing the error indicator!\n";
  for(int i = 0;i < n_element;++ i){
    const int& ele_idx = i;
    Element<double, DIM> the_ele = fem_space->element(ele_idx);

    const int& n_quadrature_point = element_cache[ele_idx].n_quadrature_point;

    ////////////////////////////////////////////////////////////
    // If the element is on the boundary, then ignore it.
    /*
    const std::vector<int>& ele_dof = the_ele.dof();
    u_int n_ele_dof = ele_dof.size();
    int is_on_bound = 0;
    for(int k = 0;k < n_ele_dof;k ++)
      {
	typename FEMSpace<double, DIM>::dof_info_t dof = fem_space->dofInfo(ele_dof[k]);
	if(dof.boundary_mark == 1)
	  {
	    is_on_bound = 1;
	  }
      }
    if(is_on_bound==1)
      {
	//	std::cout<<"This tetrahedron element is on the boundary!"<<std::endl;
	continue;
      }
    */
    /////////////////////////////////////////////////////////////
    

    // Computing the h_K for the K-th element diameter. For 3D case, the diameter equals to the d of the sphere with
    // same volume as the tetrahedron element.
    double h_K = 0.;
    if(DIM == 3)
      {
	h_K = pow(3.0*element_cache[ele_idx].volume/(4.0*PI),1.0/3.0);
      }

    if(DIM == 2)
      {
	h_K = 2.0*sqrt(element_cache[ele_idx].volume/PI);
      }
    
    std::vector<std::vector<double>> gradient = prime_solution->gradient(element_cache[ele_idx].q_point, the_ele);
    //std::vector<value_type2> w_val = w_2->value(q_point, the_ele_float);
    std::vector<double> dual_val = dual_solution->value(element_cache[ele_idx].q_point,the_ele);

    double residual = 0.;
    for(int l = 0;l < n_quadrature_point;++ l){
      
      //double Jxw_value = Jxw[l];
      ind[ele_idx] += element_cache[ele_idx].Jxw[l]*pow(dual_val[l], 2.0);
      // residual += element_cache[ele_idx].Jxw[l]*innerProduct(gradient[l],gradient[l]);

      double f_val = _f_double2d(element_cache[ele_idx].q_point[l]);
      residual += element_cache[ele_idx].Jxw[l]*f_val*f_val;
    }
    //residual = sqrt(residual);
    /////////////

    ////////////////
    // with residual term;
    ind[ele_idx] = sqrt(ind[ele_idx]);// compute l2-norm of || w ||;
    ind[ele_idx] = ind[ele_idx]*sqrt(h_K*h_K*residual+Jump[ele_idx]);

  }
  //std::cout << "Attention: CPU time for get indicator() is: " << omp_get_wtime() - time_get_indicator << std::endl;
  std::cout<<"Finish computing the error indicator!"<<std::endl;
}

TEMPLATE
void THIS::post_con()
{
  if(is_need_post_con())
    {
      ////////////////////////////////////////////
      // Compute the prime solution in the double type in the fem space!.
      std::cout<<"\n Above prime solution is computed in float space, so it need one more step to compute the prime solution in double space!" << std::endl;
      std::cout<<"Begin to compute the prime solution in the double space..."<< std::endl;
      double time_post_start = 0.;
      time_post_start = omp_get_wtime();
      
      final_solution = new FEMFunction<double,DIM,DIM,DIM,double>(*fem_space);
      final_stiff_matrix = new NewMatrix<DIM,double,DIM,DIM,double>(*fem_space, element_cache);
      final_stiff_matrix->algebricAccuracy() = ACC;
  
      final_stiff_matrix->build();

      std::cout<<"Begin to build RHS vector..."<<std::endl;
      Vector<double>* final_rhs;
      final_rhs = new Vector<double>();
      
      final_rhs->reinit(fem_space->n_dof());
      final_rhs->Vector<double>::operator = (0.0); 

  
      //std::cout<<"The l2_norm of rhs is :" <<rhs->l2_norm()<<std::endl;
      const int& n_element = fem_space->n_element();

      RegularMesh<DIM>& r_mesh = irregular_mesh->regularMesh();
  
      for(int j = 0;j < n_element;++ j)
	{
	  const int& ele_idx = j;
	  Element<double, DIM> the_ele = fem_space->element(ele_idx);

	  const int& n_quadrature_point = element_cache[ele_idx].n_quadrature_point;
	  const std::vector<int>& ele_dof = the_ele.dof();
	  u_int n_ele_dof = ele_dof.size();
	  for (u_int l = 0;l < n_quadrature_point;++ l)
	    {
	      double f_val = _f_double2d(element_cache[ele_idx].q_point[l]);

	      for(u_int i = 0;i < n_ele_dof;++ i)
		{
		  (*final_rhs)(ele_dof[i]) += element_cache[ele_idx].Jxw[l]*(f_val*element_cache[ele_idx].basis_value[i][l]);
		}
	  
	    }
      
	}

      ///////////////////////
      // Begin to add boundary condition
      BoundaryFunction<double,DIM,DIM,DIM,double> boundary;
      boundary.reinit(BoundaryConditionInfo::DIRICHLET, 1, &double_u2d);
      BoundaryConditionAdmin<double,DIM,DIM,DIM,double> boundary_admin(*fem_space);
      boundary_admin.add(boundary);
      boundary_admin.apply(*final_stiff_matrix, *final_solution, *final_rhs);

      //////////////////////
      // Begin to solve the linear system;
      AMGSolver solver;
      solver.lazyReinit(*final_stiff_matrix);

      solver.solve(*final_solution, *final_rhs, 1.0e-15, 1000);

      /////////////////////
      // Begin to write the final data;
      final_solution->writeOpenDXData("final_solution.dx");

      double final_error;
      final_error = Functional::L2Error(*final_solution, FunctionFunction<double>(&double_u2d), ACC);
      std::cerr << "\nL2 error = " << final_error << std::endl;

      final_error = Functional::L1Error(*final_solution, FunctionFunction<double>(&double_u2d), ACC);
      std::cerr << "\nL1 error = " << final_error << std::endl;
      std::cout<<"The CPU time for the extra step is: " << omp_get_wtime() - time_post_start << std::endl;

      /////////////////////
      // Begin to compute the J error of the final prime solution:

      new_J_error = 0.;
      new_J_error_l2norm = 0.;
  
      for(int i = 0;i < n_element;i ++)
	{
	  const int& ele_idx = i;
	  Element<double,DIM> the_ele = fem_space->element(ele_idx);

	  const int& n_quadrature_point = element_cache[ele_idx].n_quadrature_point;
	  std::vector<double> u_h_value = final_solution->value(element_cache[ele_idx].q_point, the_ele);

	  const std::vector<int>& ele_dof = the_ele.dof();
	  u_int n_ele_dof = ele_dof.size();
      
	  double temp_J_error = 0.;
	  double temp_error_per = 0.;

	  for(u_int j = 0;j < n_ele_dof;++ j)
	    {
	      GeometryBM& pnt_geo = r_mesh.geometry(0, ele_dof[j]);
	      AFEPack::Point<DIM>& pnt = r_mesh.point(pnt_geo.vertex()[0]);
	      if(infield(pnt))
		{
	  
		  for(u_int l = 0;l < n_quadrature_point;++ l)
		    {
		      new_J_error += element_cache[ele_idx].Jxw[l] * fabs(double_u2d(element_cache[ele_idx].q_point[l]) - u_h_value[l]);
		    }
		  break;

		}
	    }
	}
      std::cout<<"The total volume is: "<<total_volume<<std::endl;
      //new_J_error = fabs(new_J_error)*total_volume;
      //std::cout<<"The new L2-error J(e) of dual problem is::: "<<sqrt(fabs(new_J_error))<<std::endl;
      std::cout<<"The new L2-error J(e) of dual problem is::: "<<fabs(new_J_error)<<std::endl;
      std::cout<<"The new l2-norm error J(e) of dual problem is::: "<<fabs(new_J_error)/sqrt(total_volume)<<std::endl;
      new_J_error_l2norm = fabs(new_J_error)/sqrt(total_volume);
      new_J_error = fabs(new_J_error);
      
      //return;
    }
  else
    {
      std::cout<<"This method compute prime in double, do not need to compute the prime solution again!"<<std::endl;
    }
}


#undef TEMPLATE
#undef THIS
#endif
