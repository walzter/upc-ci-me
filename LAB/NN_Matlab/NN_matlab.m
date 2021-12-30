% >>>>>>>>>> PART 1: Defining weights, bias + f(x) <<<<<<<<
% we define the Neuron Weights 
w = [4 -2];
%Then the neuron bias 
b = -3; 
% the activation function which will be used 
func = 'tansig';
% Activation function: Logistic sigmoid transfer function 
% func = 'logsig'
% Activation function: Hard-limit transfer function (threshold) 
% func = 'hardlim'
% Activation function: Linear transfer function 
% func = 'purelin'

% >>>>>>>>>> PART 2: Initiate Input vectors <<<<<<<<
%define input vectors 
p = [2 3];

% >>>>>>>>>> PART 3: Neuron Input and Output <<<<<<<<
% calculate the neuron output 
net_input = p*w'+b;
% passing it through the activation function 
neuron_output = feval(func, net_input);

% >>>>>>>>>> PART 4: Plotting <<<<<<<<
[p1,p2] = meshgrid(-10:.25:10);
z = feval(func, [p1(:) p2(:)]*w'+b ); z = reshape(z,length(p1),length(p2)); plot3(p1,p2,z);
grid on;
xlabel('Input 1');
ylabel('Input 2');
zlabel('Neuron output');