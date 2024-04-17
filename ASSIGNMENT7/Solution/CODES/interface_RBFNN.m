clc;clearvars;close all;
infile         = 'Clust_data_RBFNN.xlsx';

datatable   = readtable(infile);
headers     = datatable.Properties.VariableNames; headers(:,end)=[];
training_data = datatable.Variables; clear datatable;

network_architecture.learning_rate  = 5;
network_architecture.max_epoch      = 10000;
network_architecture.receptors      = [1,1]; %no. of receptors for each class

trainedNeuralNetwork = RadialBasisClassifier(network_architecture,training_data);
cost = trainedNeuralNetwork.cost;
max_epoch = network_architecture.max_epoch;
plot(1:max_epoch,cost);
title('Cost Vs. Epochs');xlabel('epochs');ylabel('cost');

no_of_class = length(network_architecture.receptors);
data=zeros(25*25,2);k=1;
for i=1:25
    for j=1:25
        data(k,:)= [i j];
        k=k+1;
    end
end
predicteds=predictoutput_rbfnn(trainedNeuralNetwork,data);
figure;
for k=1:no_of_class
    class_instances=data(predicteds==k,:);
    scatter(class_instances(:,1),class_instances(:,2));
    hold on
end
