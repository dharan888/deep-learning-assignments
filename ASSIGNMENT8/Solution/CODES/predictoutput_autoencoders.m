function networkpredictions = predictoutput_autoencoders(trainedNeuralNetwork,test_data)

    act_fn          = trainedNeuralNetwork.network_architecture.activation_function;
    network_synaptics = trainedNeuralNetwork.network_synaptics;
   
    test_data          = (test_data-mean(test_data))./std(test_data);
    test_data          = transpose(test_data);
    
    networkpredictions.layers      = feedforward(network_synaptics,test_data,act_fn);
    networkpredictions.targets     = test_data; clear test_data;
    networkpredictions.predicteds  = networkpredictions.layers(end).activations;
    networkpredictions.errors      = networkpredictions.targets-networkpredictions.predicteds;
    networkpredictions.cost        = cost_function(networkpredictions.errors);
    
end