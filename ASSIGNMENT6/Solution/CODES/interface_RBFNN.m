clc;clearvars;close all;
infile         = 'data_data.xlsx';

datatable   = readtable(infile);
headers     = datatable.Properties.VariableNames; headers(:,end)=[];
training_data = datatable.Variables; clear datatable;

network_architecture.learning_rate  = 10^-3;
network_architecture.max_epoch      = 10000;
network_architecture.receptors      = [2,2,1]; %no. of receptors for each class

trainedNeuralNetwork = RadialBasisClassifier(network_architecture,training_data);
cost = trainedNeuralNetwork.cost;
max_epoch = network_architecture.max_epoch;
plot(1:max_epoch,cost);
title('Cost Vs. Epochs');xlabel('epochs');ylabel('cost');
figure;

test_data = training_data(:,1:end-1);
[no_of_instances = size(test_data,1);
predicteds = predictoutput_rbfnn(trainedNeuralNetwork,test_data);
targets    = training_data(:,end);

scatter(predicteds,targets);
title('predictions Vs. targets');xlabel('predicteds');ylabel('targets');
figure;
hold on
view(3);
scatter3(test_data(:,1),test_data(:,2),networkpredictions.targets');
scatter3(test_data(:,1),test_data(:,2),networkpredictions.predicteds');
hold off

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
    scatter(class_instances(:,1),class_instances(:,2);;
    hold on
end

scatter(
