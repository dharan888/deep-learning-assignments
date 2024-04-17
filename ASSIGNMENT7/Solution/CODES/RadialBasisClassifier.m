function trainedNeuralNetwork = RadialBasisClassifier(network_architecture,data)

trainedNeuralNetwork.network_architecture=network_architecture;
learning_rate   = network_architecture.learning_rate;
max_epoch       = network_architecture.max_epoch;
receptors       = network_architecture.receptors;

no_of_receptors    = sum(receptors);
no_of_class        = length(receptors);

%%data               = transpose(data);
inputs             = data(:,1:end-1);
targets            = data(:,end); clear data;
[no_of_instances,no_of_features] = size(inputs);

cost               = zeros(max_epoch,1);
weights            = zeros(no_of_class,no_of_receptors+1);

n=1;
centroids=zeros(no_of_receptors,no_of_features);
variance=zeros(no_of_receptors,1);
for class_num = 1:no_of_class
    class_data      = inputs(targets==class_num,:);
    no_of_clusters  = receptors(class_num);
    m               = no_of_clusters+n-1;
    clusters        = KNNplus_clustering(class_data,no_of_clusters);
    centroids(n:m,:)= clusters.centroids;
    variance(n:m,:) = clusters.variance;
    n               = m+1;
end

phi=zeros(no_of_instances,no_of_receptors);
for r=1:no_of_receptors
    diff=inputs-centroids(r,:);
    phi(:,r)=exp(-0.5*sum(diff.*diff,2)/variance(r));
end
phi         = transpose([ones(no_of_instances,1),phi]);
indices     = sub2ind([no_of_class,no_of_instances],targets,(1:no_of_instances)');

for epoch = 1:max_epoch
    netinputs   = weights*phi;
    predicteds  = softmax_activation(netinputs);
    cost(epoch,1)= sum(-log(predicteds(indices)))/no_of_instances; %cross entropy
    
    delta       = predicteds; clear predicted;
    delta(indices)=delta(indices)-1;
    gradient    = delta*transpose(phi)/no_of_instances;
    weights     = weights - learning_rate*gradient;
end

trainedNeuralNetwork.weights=weights;
trainedNeuralNetwork.cost =cost;
trainedNeuralNetwork.receptors.centroids=centroids;
trainedNeuralNetwork.receptors.variance=variance;

end
