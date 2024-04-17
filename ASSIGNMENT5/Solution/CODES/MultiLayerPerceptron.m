clc;clearvars;close all;
infile         = 'Perceptron_Adaline.xlsx';
neurons_scheme = [5,4,3,2];

datatable   = readtable(infile);
headers     = datatable.Properties.VariableNames; headers(:,end)=[];
inputs      = datatable.Variables; clear datatable;

no_of_totalLayers  = length(neurons_scheme);
no_of_hiddenLayers = no_of_totalLayers-2;
no_of_synaptics    = no_of_totalLayers-1;
no_of_features     = neurons_scheme(1);
no_of_targets      = neurons_scheme(2);
no_of_instance     = size(inputs,1);

inputs_normalized  = (inputs-mean(inputs))./std(inputs);
targets            = inputs(:,end-no_of_targets+1:end); 
inputs(:,end-no_of_targets+1:end)=[];
inputs = inputs'; targets=targets';

learning_rate   = 10^-2;
max_epoch       = 10000; 
termination_threshold = 10^-6;


weights         = zeros(no_of_features+1,no_of_targets);
cost            = zeros(max_epoch,1);

network_synaptics(no_of_synaptics).weights =zeros(5);
for i = 1:no_of_synaptics
    input_neuron_size  = neurons_scheme(i);
    output_neuron_size = neurons_scheme(i+1);
    network_synaptics(i).weights =zeros(output_neuron_size+1,input_neuron_size+1);
end

for epoch = 1:max_epoch
    
    layers = feedforward(network_synaptics,inputs);
    predictions = layers(end).acitvations;
    errors = targets-predictions;
    cost(epoch) = cost_function(errors);
    network_synaptics = backpropagate(network_synaptics,layers,errors);
    
end

plot(1:epoch,cost);

%------------------------------------------------------------------------
%Subroutines

function layers = feedforward(network_synaptics,inputs)
    no_of_synaptics = length(network_synaptics);
    layers(1:no_of_synaptics+1).acitvations = inputs;
    for synaptic_num = 1:no_of_synaptics
        k=synaptic_num+1;
        weights = network_synaptics(synaptic_num);
        layers(k).netinputs = weights*(layers(k-1).activations);
        layers(k).activations = sigmoid_activation(netinputs);
    end
end

function network_synaptics = backpropagate(network_synaptics,layers,errors)
    no_of_synaptics = length(network_synaptics);
    for synaptic_num = 1:no_of_synaptics
        activations = layers(synaptic_num).activations;
        netinputs   = layers(synaptic_num).netinputs;
        k = no_of_synaptics-synaptic_num+1;
        weights = network_synaptics(k).weights;
        if synaptic_num==1 
           delta = (errors.*sigmoid_derivative(netinputs));
        else     
           delta   = (tranpose(weights)*delta).*sigmoid_derivative(netinputs);
        end
        gradient = delta*transpose(acitvations);
        network_synaptics(k).weights = weights-learning_rate*gradient;
    end
end

function activations = sigmoid_activation(netinputs)
    acitvations = 1./(1+exp(-netinputs));
end

function activation_derivatives = sigmoid_derivative(net_inputs)
    activations = sigmoid_activation(net_inputs);
    activation_derivatives = activations.*(1-activations);
end

function cost = cost_function(errors)
    no_of_instances = size(errors,2);
    cost   = 0.5*errors*transpose(errors)/no_of_instances;
end
