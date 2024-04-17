function predicteds = predictoutput_rbfnn(trainedNeuralNetwork,inputs)
    
    receptors       = trainedNeuralNetwork.network_architecture.receptors;
    weights         = trainedNeuralNetwork.weights;
    centroids       = trainedNeuralNetwork.receptors.centroids;
    variance        = trainedNeuralNetwork.receptors.variance;
    
    no_of_receptors    = sum(receptors);
    no_of_instances    = size(inputs,1);
      
    phi=zeros(no_of_instances,no_of_receptors);
    for r=1:no_of_receptors
        diff=inputs-centroids(r,:);
        phi(:,r)=exp(-0.5*sum(diff.*diff,2)/variance(r));
    end
    phi         = transpose([ones(no_of_instances,1),phi]);
    netinputs   = weights*phi;
    activations = transpose(softmax_activation(netinputs));
    [~,predicteds]  = max(activations,[],2);
    
end