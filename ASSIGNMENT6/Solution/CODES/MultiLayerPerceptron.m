function trainedNeuralNetwork = (network_architecture,data);

trainedNeuralNetwork.network_architecture=network_architecture;
learning_rate   = network_architecture.learning_rate;
max_epoch       = network_architecture.max_epoch;
act_fn          = network_architecture.activation_function;
neurons_scheme  = network_architecture.neurons_scheme;

no_of_totalLayers  = length(neurons_scheme);
no_of_synaptics    = no_of_totalLayers-1;
no_of_features     = neurons_scheme(1);
no_of_instances    = size(data,1);

data               = (data-mean(data))./std(data);
data               = transpose(data);
inputs             = data(1:no_of_features,:);
targets            = data(no_of_features+1:end,:); clear data;

cost            = zeros(max_epoch,1);
network_synaptics(no_of_synaptics).weights =zeros(5);

for i = 1:no_of_synaptics
    input_neuron_size  = neurons_scheme(i);
    output_neuron_size = neurons_scheme(i+1);
    network_synaptics(i).weights =ones(output_neuron_size,input_neuron_size)*0.001;
    network_synaptics(i).biases  =ones(output_neuron_size,1)*0.001;
end

for epoch = 1:max_epoch
    
    layers = feedforward(network_synaptics,inputs,act_fn);
    predictions = layers(end).activations;
    errors = targets-predictions;
    cost(epoch) = cost_function(errors);
    network_synaptics = backpropagate(learning_rate,act_fn,network_synaptics,layers,errors);
    
end

trainedNeuralNetwork.network_synaptics=network_synaptics;
trainedNeuralNetwork.cost =cost;

end
