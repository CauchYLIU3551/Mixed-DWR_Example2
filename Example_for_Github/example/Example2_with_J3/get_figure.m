filename = "Jerror.txt";
filename2 = "Ndof.txt";

J=importdata(filename);
N=importdata(filename2);

xlabel('log(Dof number)','fontsize',20);
ylabel('log(J error)','fontsize',20);
hold on;

set(gca,'FontSize',20);
plot(log(N),log(J),"k");
hold on;
plot(log(N),log(J),["*","k"]);
