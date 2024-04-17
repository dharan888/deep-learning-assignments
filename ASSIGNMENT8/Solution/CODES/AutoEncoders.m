function trainedNeuralNetwork = AutoEncoders(network_architecture,data)

trainedNeuralNetwork.network_architecture=network_architecture;
learning_rate   = network_architecture.learning_rate;
max_epoch       = network_architecture.max_epoch;
act_fn          = network_architecture.activation_function;
neurons_scheme  = network_architecture.neurons_scheme;

no_of_totalLayers  = length(neurons_scheme);
no_of_synaptics    = no_of_totalLayers-1;

inputs             = (data-mean(data))./std(data);
inputs             = transpose(inputs); clear data;

cost               = zeros(max_epoch,1);
network_synaptics(no_of_synaptics).weights =zeros(5);

for i = 1:no_of_synaptics
    input_neuron_size  = neurons_scheme(i);
    output_neuron_size = neurons_scheme(i+1);
    network_synaptics(i).weights =rand(output_neuron_size,input_neuron_size);
    network_synaptics(i).biases  =rand(output_neuron_size,1)*0.0001;
end

for epoch = 1:max_epoch
    
    layers = feedforward(network_synaptics,inputs,act_fn);
    predictions = layers(end).activations;
    errors = inputs-predictions;
    cost(epoch) = cost_function(errors);
    network_synaptics = backpropagate(learning_rate,act_fn,network_synaptics,layers,errors);
    
end

trainedNeuralNetwork.network_synaptics=network_synaptics;
trainedNeuralNetwork.cost =cost;

end
