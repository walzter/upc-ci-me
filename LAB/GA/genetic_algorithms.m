% clearing everything 
close all; clear; clc
rng default 

% global vectors to store information from the outputs 
val_vector =[];
sol_vector = [];
% GLOBAL NUMBER OF ITERATIONS
NUMBER_OF_ITERATIONS = 100;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                      SELECTED PARAMETERS                          %
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
%                       CROSSOVER FRACTION                          %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%creating a vector with shape(1, NUMBER_OF_ITERATIONS) to preallocate the size -> Matlab
final_vector = zeros(1, NUMBER_OF_ITERATIONS);

% loop through the range, and then take the average for all of the values 
for k=0:0.5:1
	for l=1:NUMBER_OF_ITERATIONS
		config = optimoptions('ga','MaxGenerations',200);
		config = optimoptions(config,'CrossoverFraction',k);
		[sol,vals,falgs,output,pop,scrs] = ga(gen_alg,num_vars,[],[],[],[],...
        [],[],[],[],config);
		final_vector(l) = vals;
	end
	val_vector = [val_vector; mean(final_vector)];
	sol_vector = [sol_vecotr; sol];
end

%plotting 
%
% GRAPH 1
%
plot(0:0.05:1,val_vector)
title('Graph 1. Finding the ideal Crossover Fraction');
xlabel('Crossover Fraction');
ylabel('Number of Generations')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                    MAXIMUM # OF GENERATIONS                       %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gen_zero = 20; 
gen_init = 10; 
gen_final = 200;
%creating a vector with shape(1, num_loops) to preallocate the size -> Matlab
final_vector = zeros(1, NUMBER_OF_ITERATIONS);
% loop through the range, and then take the average for all of the values 
for k=gen_zero:gen_init:gen_final
	for l=1:NUMBER_OF_ITERATIONS
		config = optimoptions('ga','MaxGenerations',k);
		config = optimoptions(config,'CrossoverFraction',crossover_fraction);
		[sol,vals,falgs,output,pop,scrs] = ga(gen_alg,num_vars,[],[],[],[],...
        [],[],[],[],config);
		final_vector(l) = vals;
	end
	val_vector = [val_vector; mean(final_vector)];
	sol_vector = [sol_vecotr; sol];
end

%plotting 
%
% GRAPH 2
%
plot(gen_zero:gen_init:gen_final,val_vector)
title('Graph 2. Finding the ideal number of Generations');
xlabel('Crossover Fraction');
ylabel('Fval')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                          POPULATION SIZE                          %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pop_zero = 1; 
pop_init = 2; 
pop_final = 50;
%creating a vector with shape(1, NUMBER_OF_ITERATIONS) to preallocate the size -> Matlab
final_vector = zeros(1, NUMBER_OF_ITERATIONS);
% loop through the range, and then take the average for all of the values 
for k=gen_zero:gen_init:gen_final
	for l=1:NUMBER_OF_ITERATIONS
		config = optimoptions('ga','PopulationSize',k,...
             'MaxGenerations',gen,'CrossoverFraction',crossFraction);
		[sol,vals,falgs,output,pop,scrs] = ga(gen_alg,num_vars,[],[],[],[],...
        [],[],[],[],config);
		final_vector(l) = vals;
	end
	val_vector = [val_vector; mean(final_vector)];
	sol_vector = [sol_vecotr; sol];
end

%plotting
%
% GRAPH 3
% 
plot(pop_zero:pop_init:pop_final,val_vector)
title('Graph 3. Finding the ideal Population Size');
xlabel('Population Size');
ylabel('Fval')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                        SELECTION FUNCTION                         %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%dictionary-like structure for keeping all the selection functions
selection_function = {[],'selectionstochunif','selectionremainder','selectionuniform','selectionroulette'};
%creating a vector with shape(1, NUMBER_OF_ITERATIONS) to preallocate the size -> Matlab
final_vector = zeros(1, NUMBER_OF_ITERATIONS);
% loop through the selection functions to find the best one
for k=1:length(selection_function)
	display(selection_function{k})
	for l=1:NUMBER_OF_ITERATIONS
		config = optimoptions('ga',"SelectionFcn",selection_function{k},...
             'PopulationSize',pop_size,'MaxGenerations',max_gens,'CrossoverFraction',crossover_fraction);
		[sol,vals,falgs,output,pop,scrs] = ga(gen_alg,num_vars,[],[],[],[],...
        [],[],[],[],config);
		final_vector(l) = vals;
	end
	val_vector = [val_vector; mean(final_vector)];
	sol_vector = [sol_vecotr; sol];
end

%need to use stem here 
%
% GRAPH 4
%
stem(1:length(selection_function),val_vector);
title('Graph 4. Identifying the Selection Function')
ylabel('Fval')
% similar to sns -> need to get current axis and access the attribute
set(gca, 'xticklabel',selection_function)
xtickangle(45)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                   %
%                        INITIAL RANGE                              %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
r_zero = 1.1; 
r_init = 2; 
r_final = 50;
%creating a vector with shape(1, NUMBER_OF_ITERATIONS) to preallocate the size -> Matlab
final_vector = zeros(1, NUMBER_OF_ITERATIONS);
% loop through the range, and then take the average for all of the values 
for k=r_zero:r_init:r_final
	for l=1:NUMBER_OF_ITERATIONS
		config = optimoptions('ga','InitialPopulationRange',[1;k],...
             "SelectionFcn",selection_function,'PopulationSize',pop_size,'MaxGenerations',max_gens,'CrossoverFraction',crossover_fraction);
		[sol,vals,falgs,output,pop,scrs] = ga(gen_alg,num_vars,[],[],[],[],...
        [],[],[],[],config);
		final_vector(l) = vals;
	end
	val_vector = [val_vector; mean(final_vector)];
	sol_vector = [sol_vecotr; sol];
end

%plotting
%
% GRAPH 5
% 
plot(r_zero:r_init:r_final,val_vector)
title('Graph 5. Finding the ideal Initial Range');
xlabel('Initial Range');
ylabel('Fval')