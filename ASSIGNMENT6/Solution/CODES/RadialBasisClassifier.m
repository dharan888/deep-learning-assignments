function trainedNeuralNetwork = RadialBasisClassifier(network_architecture,data)

trainedNeuralNetwork.network_architecture=network_architecture;
learning_rate   = network_architecture.learning_rate;
max_epoch       = network_architecture.max_epoch;
receptors       = network_architecture.receptors;

no_of_receptors    = sum(receptors);
no_of_class        = length(receptors);
[no_of_instances,no_of_features] = size(data);

%%data               = transpose(data);
inputs             = data(:,1:no_of_features);
targets            = data(:,end); clear data;

cost               = zeros(max_epoch,1);
weights            = zeros(no_of_class+1,no_of_receptors);

n=1;
centroids=zeros(no_of_receptors,no_of_features);
variance=zeros(no_of_receptors,1);
for class_num = 1:no_of_class
    class_data = inputs(targets==class_num,:);
    no_of_clusters = receptors(class_num);
    if no_of_clusters == 1
       centroids(class_num,:)=mean(class_data);
       delta = class_data-centroids;
       variance(class_num,:)= sum(delta.*delta,'all');
       n=n+1;
    else
       m=no_of_clusters+n;
       clusters = KNNplus_clustering(class_data,no_of_clusters);
       centroids(n:m,:)=clusters.centroids;
       variance(n:m,:)=clusters.variance;
    end
end

phi=zeros(no_of_instances,no_of_receptors);
for r=1:no_of_receptors
    delta=inputs-centroids(r,:);
    phi(:,r)=sum(0.5*delta.*delta/variance(r));
end
phi         = transpose([ones(no_of_instances,1),phi]);
indices     = sub2ind(size(no_of_class,no_of_instances),targets,1:no_of_instances);

cost=zeros(max_epoch,1);
for epoch = 1:max_epoch
    netinputs   = weights*phi;
    predicted   = softmax_activation(netinputs);
    cost(epoch,1)= sum(-log(predicted),'all')/no_of_instances; %cross entropy
    
    gradient    = (predicted(indices)-1)*transpose(phi)/no_of_instances;
    weights     = weights + learning_rate*gradient;
end

trainedNeuralNetwork.weights=weights;
trainedNeuralNetwork.cost =cost;
trainedNeuralNetwork.receptors.centroids=centroids;
trainedNeuralNetwork.receptors.variance=variance;

end


function activations = softmax_activation(netinputs)
    activations = exp(netinputs);
    activations = activations./sum(activations);
end