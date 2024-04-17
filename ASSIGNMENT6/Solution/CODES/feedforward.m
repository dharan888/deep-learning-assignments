function layers = feedforward(network_synaptics,inputs,act_fn)
    no_of_synaptics = length(network_synaptics);
    layers(1).activations = inputs;
    layers(1).netinputs   = inputs;
    layers(no_of_synaptics+1).activations = 0;
    for synaptic_num = 1:no_of_synaptics
        k=synaptic_num+1;
        weights = network_synaptics(synaptic_num).weights;
        biases  = network_synaptics(synaptic_num).biases;
        layers(k).netinputs = weights*(layers(k-1).activations)+biases;
        if synaptic_num == no_of_synaptics
           layers(k).activations = layers(k).netinputs;
        else
           layers(k).activations = activation_function(layers(k).netinputs,act_fn); 
        end   
    end
end