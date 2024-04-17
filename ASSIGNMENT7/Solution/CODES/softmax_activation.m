function activations = softmax_activation(netinputs)
    activations = exp(netinputs);
    activations = activations./sum(activations);
end