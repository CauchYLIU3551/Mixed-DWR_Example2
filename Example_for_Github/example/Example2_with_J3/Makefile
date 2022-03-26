CPPFLAGS := -I../../include -I$(HOME)/local/include -I/~/include -I~/myinclude

CXXFLAGS := -O0 

CXX=mpicxx


LDFLAGS := -L$(HOME)/local/lib -L~/mylib -L~/lib

LDLIBS :=  -lAFEPack -ldeal_II -ldl -lgomp 

main : main.o
	$(CXX) -o main main.o $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)

clean:
	rm -rf main.o main *~ u_PDE_* ele_geo.txt element_volume.txt res_and_dual.txt
