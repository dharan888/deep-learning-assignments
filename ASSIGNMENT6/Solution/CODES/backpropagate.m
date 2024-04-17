function network_synaptics = backpropagate(learning_rate,act_fn,network_synaptics,layers,errors)
    no_of_synaptics = length(network_synaptics);
    for s = no_of_synaptics:-1:1
        k = s+1;
        if s==no_of_synaptics 
           delta = errors;
        else     
           delta = (transpose(network_synaptics(k).weights)*delta).*derivative_function(layers(k).netinputs,act_fn);
        end
        network_synaptics(s).weights = network_synaptics(s).weights+learning_rate*delta*transpose(layers(s).activations);
        network_synaptics(s).biases  = network_synaptics(s).biases+learning_rate*sum(delta,2);
    end
end