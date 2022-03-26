#!/bin/bash

make

export AFEPACK_PATH=$HOME/local/include/AFEPack:~/local/AFEPack
export AFEPACK_TEMPLATE_PATH=$AFEPACK_PATH/template/tetrahedron:$AFEPACK_PATH/template/twin_tetrahedron:$AFEPACK_PATH/template/four_tetrahedron:$AFEPACK_PATH/template/triangle:$AFEPACK_PATH/template/twin_triangle:$AFEPACK_PATH/template/interval

export LD_LIBRARY_PATH=$HOME/local/lib

#export OMP_DISPLAY_ENV=true
#export OMP_PROC_BIND=cores
#export OMP_PLACES=cores
#export OMP_NUM_THREADS=$3


#numactl --interleave=all 
./main $1 $2
