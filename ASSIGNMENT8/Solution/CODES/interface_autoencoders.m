clc;clearvars;close all;
infile         = 'autoencoder_nonlinear.xlsx';

datatable   = readtable(infile);
headers     = datatable.Properties.VariableNames; headers(:,end)=[];
training_data = datatable.Variables; clear datatable;
encode_scheme = [2 2];
decode_scheme = [];

network_architecture.learning_rate  = 0.01;
network_architecture.max_epoch      = 100000;
network_architecture.activation_function = 'sig';

nF      = size(training_data,2);
network_architecture.neurons_scheme = [nF,encode_scheme,decode_scheme,nF];

trainedNeuralNetwork = AutoEncoders(network_architecture,training_data);
cost = trainedNeuralNetwork.cost;
max_epoch = network_architecture.max_epoch;
plot(1:max_epoch,cost);
title('Cost Vs. Epochs');xlabel('epochs');ylabel('cost');

test_data = training_data;
[no_of_instances,no_of_features] = size(test_data);
networkpredictions = predictoutput_autoencoders(trainedNeuralNetwork,test_data);
display(networkpredictions.cost);
error_percent = round((networkpredictions.errors./test_data')*100,2);

k=0;rem=2;
for ftr_num = 1:no_of_features
    figure;
    bar(1:no_of_instances,error_percent(ftr_num,:));
    title_text = strcat('X',string(ftr_num));
    title(title_text);
    title('Error Bars');ylabel('Error Percent');xlabel(title_text);
    k=k+2;rem=no_of_features-2;
end
