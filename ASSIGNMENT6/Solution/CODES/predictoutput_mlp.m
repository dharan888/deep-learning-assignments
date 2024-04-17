function networkpredictions = predictoutput_rbfnn(trainedNeuralNetwork,test_data)
    
    neurons_scheme  = trainedNeuralNetwork.network_architecture.neurons_scheme;
    act_fn          = trainedNeuralNetwork.network_architecture.activation_function;
    network_synaptics = trainedNeuralNetwork.network_synaptics;
    
    no_of_features     = neurons_scheme(1);
    no_of_instances    = size(test_data,1);
    
    
    test_data          = (test_data-mean(test_data))./std(test_data);
    test_data          = transpose(test_data);
    inputs             = test_data(1:no_of_features,:);
    targets            = test_data(no_of_features+1:end,:)';
    %mean_targets       = mean(targets);std_targets        = std(targets);
    networkpredictions.targets     = test_data(no_of_features+1:end,:); clear test_data

    layers             = feedforward(network_synaptics,inputs,act_fn);
    networkpredictions.predicteds  = layers(end).activations;
    networkpredictions.errors      = networkpredictions.targets-networkpredictions.predicteds;
    networkpredictions.cost        = cost_function(networkpredictions.errors);
    
end