clc;clearvars;close all;
infile         = 'data_data.xlsx';

datatable   = readtable(infile);
headers     = datatable.Properties.VariableNames; headers(:,end)=[];
training_data = datatable.Variables; clear datatable;

network_architecture.learning_rate  = 10^-3;
network_architecture.max_epoch      = 10000;
network_architecture.neurons_scheme = [2,2,1]; %no. of neurons in [inputlayer firsthiddenlayer secondhiddenlayer outputlayer]
network_architecture.activation_function = 'sig';

trainedNeuralNetwork = MultiLayerPerceptron(network_architecture,training_data);
cost = trainedNeuralNetwork.cost;
max_epoch = network_architecture.max_epoch;
plot(1:max_epoch,cost);
title('Cost Vs. Epochs');xlabel('epochs');ylabel('cost');
figure;

test_data = training_data;
no_of_instances = size(test_data,1);
networkpredictions = predictoutput_mlp(trainedNeuralNetwork,test_data);
display(networkpredictions.cost);
scatter(networkpredictions.predicteds,networkpredictions.targets);
title('predictions Vs. targets');xlabel('predicteds');ylabel('targets');
figure;
bar(1:no_of_instances,networkpredictions.errors);
title('Error Bars');xlabel('instances');ylabel('errors');
figure;
test_data = (test_data-mean(test_data))./std(test_data);
hold on
view(3);
scatter3(test_data(:,1),test_data(:,2),networkpredictions.targets');
scatter3(test_data(:,1),test_data(:,2),networkpredictions.predicteds');
hold off


