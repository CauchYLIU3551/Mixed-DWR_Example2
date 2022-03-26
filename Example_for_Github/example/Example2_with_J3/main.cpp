#include<iostream>
#include<fstream>
#include<string>
#include<cmath>
#include<set>
#include<omp.h>

#include "mixed_precision_J4.h"

int main(int argc, char* argv[])
{
  std::string filename = argv[1];
  int refine_times = atoi(argv[2]);
  //MixAdaptor<double, float> MyAdapt;
  MixDWR<float, double> MyAdapt;
  MyAdapt.genMesh(filename, refine_times);
  std::cout<<"generating mesh is done..."<<std::endl;
  
  int total_adapt_times = 5;
  int the_adapt_idx = 0;
  std::vector<double> tol(15);

  tol[0]=9.0e-3;
  tol[1]=7.0e-3;
  tol[2]=5.0e-3;
  tol[3]=3.0e-3;
  tol[4]=1.0e-3;
  tol[5]=9.0e-4;
  tol[6]=7.0e-4;
  tol[7]=5.0e-4;
  tol[8]=3.0e-4;
  tol[9]=1.0e-4;
  tol[10]=9.0e-5;
  tol[11]=7.0e-5;
  tol[12]=5.0e-5;
  tol[13]=3.0e-5;
  tol[14]=1.0e-5;

  MyAdapt.initialize();
  int i = 0;
  double J_error = 0.;
  int N_dof = 0.;
  double time_start = 0., time_start2 = 0., adapt_time_start = 0.;
  time_start = omp_get_wtime();

  std::vector<double> Jump;
  Indicator<DIM>  indicator;

  std::ofstream Jerror, Ndof, out_Jump, out_Ind;
  Jerror.open("Jerror.txt");
  Ndof.open("Ndof.txt");


  do{
    MyAdapt.buildspace();
    MyAdapt.updateElementCache();
    MyAdapt.gen_prime_solution();
    //MyAdapt.gen_dual_solution();
    MyAdapt.get_dual_error();
    MyAdapt.post_con();

    J_error = MyAdapt.get_J_error();
    Jerror<<J_error<<std::endl;
    N_dof = MyAdapt.get_number_of_dof();
    Ndof<<N_dof<<std::endl;

    //break;
    if(i!=0)
    {
      std::cout<<"The CPU time for one iteration from adapt one mesh and compute the solution and dual error is: "<< omp_get_wtime() - time_start2 << std::endl;
    }
    //if(the_adapt_idx++ > total_adapt_times | J_error<1.0e-2)break;
    if(the_adapt_idx++ > total_adapt_times)break;
    time_start2 = omp_get_wtime();
    adapt_time_start = omp_get_wtime(); 
    MyAdapt.gen_dual_solution();
    MyAdapt.getIndicator();

    std::string str_Jump("Jump"), str_Ind("Indicator");
    std::string str_tmp(std::to_string(the_adapt_idx));
    //str_tmp = to_string(the_adapt_idx);
    str_Jump.append(str_tmp);
    str_Ind.append(str_tmp);   

    //std::stringstream str_Jump, str_Ind;
    //str_Jump<<"Jump"<<

    out_Jump.open(str_Jump+".txt");
    Jump = MyAdapt.get_Jump();
    for(int k = 0;k < Jump.size();k ++)
    {
      out_Jump << Jump[k] << std::endl;
    }
    out_Jump.close();

    out_Ind.open(str_Ind+".txt");
    indicator = MyAdapt.get_Indicator();
    for(int k = 0;k < indicator.size();k ++)
    {
      out_Ind << indicator[k] << std::endl;
    }
    out_Ind.close();

    std::cout << "\nThe CPU time for get error indicator is: " << omp_get_wtime() - adapt_time_start << std::endl;
    //MyAdapt.adaptMesh(tol[i]);
    //MyAdapt.adaptMesh(1.0e-3);
    MyAdapt.adaptMesh(1.0e-5);
    i++;
  }while(1);
  std::cout << "\nThe CPU time for solving the whole problem is: " << omp_get_wtime() - time_start << std::endl;
  MyAdapt.post_con();
  
  Jerror.close();
  Ndof.close();
  return 0;
}
