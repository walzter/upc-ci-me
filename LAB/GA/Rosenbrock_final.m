%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                       Genetic Algorithm                           %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% defining the fitness function and the number of variables to use 
gen_alg  = @(x)(1-x(1))^2+100*(x(2)-x(1)^2)^2;
num_vars = 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                   Solutions for each section                      %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CROSSOVER FRACTION
crossover_fraction = 0.55;

% MAX GENERATIONS
max_gens = 100; 

% POPULATION SIZE 
pop_size = 50; 

% SELECTION FUNCTION
selection_function = 'selectionremainder';

% INITIAL RANGE 
initial_range = [1;2];


% global vectors to store information from the outputs 
val_vector =[];
sol_vector = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                  SOLVE WITH BETTER PARAMETERS                     %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                STEP 1: Edit the Config with best params           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

config = optimoptions('ga','InitialPopulationRange',initial_range,...
        "SelectionFcn",selection_function,'PopulationSize',pop_size,'MaxGenerations',max_gens,'CrossoverFraction',crossover_fraction,'PlotFcn',{'gaplotstopping','gaplotbestf','gaplotbestindiv'});
    [sol,vals,output,pop,scrs] = ga(gen_alg,num_vars,[],[],[],[],...
    [],[],[],[],config);

val_vector = [val_vector; vals];
sol_vector = [sol_vector; sol];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            STEP 2: 2-D SURFACE PLOT (CONTOUR / RESPONSE)          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 f = @(x,y) (1-x).^2 + 100*(y-x.^2).^2;
 x = linspace(-2,2); y = linspace(-1,3);
 [xx,yy] = meshgrid(x,y); ff = f(xx,yy);
 levels = 10:10:300;
 LW = 'linewidth'; FS = 'fontsize'; MS = 'markersize';
 % Defining the figure and the contour
 figure, contour(x,y,ff,levels,LW,1.2), colorbar
 axis([-2 2 -1 2]), axis square, hold on


 fminx0 = @(x0) min(chebfun(@(y) f(x0,y),[-1 3]));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                STEP 3: GLOBAL MINIMUM FOR THE 2-D PLOT            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Global minimun for the 2d surface

 global_minimum = plot3(XY_ga(:,1),XY_ga(:,2),val_vector,'p','Color','r','MarkerSize',10)
 legend(global_minimum,'Global minimum','Location','best');
 title('Graph 6. Rosenbrock - Global Minimum')
 ylabel('fval')
 xlabel('Crossover Fraction');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                STEP 4: PLOT GLOBAL MINIMUM FOR THE 3-D PLOT       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
x = -2.:0.05:2.;
y = -1.:0.05:3.;
n = length(x);
m = length(y);
[X,Y] = meshgrid(x,y);
Z = zeros(n,m);

for i = 1:n
    for j = 1:m
        Z(i,j) =  (1-X(i,j))^2 + 100*(Y(i,j)-X(i,j)^2)^2;
    end
end

figure
colormap (flipud(jet))
C = log10(Z);
s = surf(X,Y,Z,C);
s.EdgeColor = 'none';

hold on     

h1= plot3(XY_ga(:,1),XY_ga(:,2),val_vector,'*','Color','r','MarkerSize',15);
legend(h1,'Global minimum','Location','best');
strm = RandStream.getGlobalStream;
%strm.State = output.rngstate.State;
[x,vals,flags,final_output] = ga(gen_alg,num_vars);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              STEP 5: FINAL RESULTS WITH SELECTED PARAMETERS       %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n')
% The parameters found was:
fprintf('The parameters found were:')
fprintf('\n')
% -Crossover fraction
fprintf('Crossover fraction : %d\n', crossover_fraction)
% -MaxGenerations
fprintf('MaxGenerations : %d\n', max_gens)
% -Population size
fprintf('Population size : %g\n', pop_size)
% -Selection function
fprintf('Selection function : selectionremainder')
% -Initial range
fprintf('Initial range : %g\n', initial_range)

fprintf('\n')
fprintf('\n')
fprintf('The number of generations was : %d\n', final_output.generations)
fprintf('The number of function evaluations was : %d\n', final_output.funccount)
fprintf('The best function value found was : %g\n', fval)
% Clear variables
clearvars options