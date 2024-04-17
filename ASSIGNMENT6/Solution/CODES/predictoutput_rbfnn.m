function predicteds = predictoutput_rbfnn(trainedNeuralNetwork,test_data)
    
    receptors       = trainedNeuralNetwork.network_architecture.receptors;
    weights         = trainedNeuralNetwork.weights;
    centroids       = trainedNeuralNetwork.receptors.centroids;
    variance        = trainedNeuralNetwork.receptors.variance;
    
    no_of_receptors    = sum(receptors);
    no_of_class        = length(receptors);
    [no_of_instances,no_of_features] = size(test_data);
    
    
    phi=zeros(no_of_instances,no_of_receptors);
    for r=1:no_of_receptors
        delta=inputs-centroids(r,:);
        phi(:,r)=sum(0.5*delta.*delta/variance(r));
    end
    phi         = transpose([ones(no_of_instances,1),phi]);
    netinputs   = weights*phi;
    [~,predicteds]  = transpose(max(softmax_activation(netinputs)));
    
end