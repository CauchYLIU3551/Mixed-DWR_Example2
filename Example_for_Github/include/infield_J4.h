#ifndef _infield_J4_h_
#define _infield_J4_h_
#include "Options.h"
//#define DIM 2
/////////////////////////////////////////
// For the specified field integral
// case 3:
// The field is: [-0.5,0]x[0.7,1.0]


// Compute the volume of the element by the four vertices of it.
double get_volume(const double * v0,
                const double * v1,
                const double * v2,
                const double * v3)
{
        return ((v1[0] - v0[0])*(v2[1] - v0[1])*(v3[2] - v0[2])
                + (v1[1] - v0[1])*(v2[2] - v0[2])*(v3[0] - v0[0])
                + (v1[2] - v0[2])*(v2[0] - v0[0])*(v3[1] - v0[1])
                - (v1[0] - v0[0])*(v2[2] - v0[2])*(v3[1] - v0[1])
                - (v1[1] - v0[1])*(v2[0] - v0[0])*(v3[2] - v0[2])
                - (v1[2] - v0[2])*(v2[1] - v0[1])*(v3[0] - v0[0]));
}


bool infield(const double * p)
{
  if(DIM==2)
    {
      // For fun 2:
      if(p[0]<0.0&&p[0]>-0.5&&p[1]<1.0&&p[1]>0.7)
      
      //if(p[0]>-1.0&&p[0]<1.0&&p[1]>-0.05&&p[1]<0.05)
	{
	  return true;
	}
      else
	{
	  return false;
	}
    }
    
  if(DIM==3)
    {
      if(p[0]<1&&p[0]>-1&&p[1]<1&&p[1]>-1&&p[2]<1&&p[2]>-1 )
	{
	  return true;
	}
      else
	{
	  return false;
	}
    }
 
}
bool notinfield(const double * p)
{
  if((p[0]>0||p[0]<-0.5) || (p[1]<0)||p[1]>0.5)
  {
    return true;
  }
  else
  {
    return false;
  }
}
#endif
