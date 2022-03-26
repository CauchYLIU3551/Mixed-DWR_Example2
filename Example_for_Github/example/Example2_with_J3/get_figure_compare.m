filename = "Jerror.txt";
filename2 = "Ndof.txt";

%filename = "Jerror.fun1.DWR.txt";
%filename2 = "Ndof.fun1.DWR.txt";

J=importdata(filename);
N=importdata(filename2);

filename3 = "2d_Jerror.txt";
filename4 = "2d_Ndof.txt";

filename3 = "Jerror.fun1.Residual.txt";
filename4 = "Ndof.fun1.Residual.txt";


J2d=importdata(filename3);
N2d=importdata(filename4);

xlabel('log(Dof number)','fontsize',20);
ylabel('log(J error)','fontsize',20);

hold on;

set(gca,'FontSize',20);
plot(log(N(1:7)),log(J(1:7)),"k");
hold on;

set(gca,'FontSize',20);
plot(log(N2d(1:9)),log(J2d(1:9)),"k--");
hold on;
legend('Mixed-DWR','Residual-based');

plot(log(N(1:7)),log(J(1:7)),["*","k"]);
hold on;
plot(log(N2d(1:9)),log(J2d(1:9)),["*","k"]);
hold on;




