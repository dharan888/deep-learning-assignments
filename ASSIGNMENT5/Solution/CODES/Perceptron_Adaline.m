clc;clearvars;close all;
infile      = 'Perceptron_Adaline.xlsx';
no_of_targets = 1;

datatable   = readtable(infile);
headers     = datatable.Properties.VariableNames; headers(:,end)=[];
inputs      = datatable.Variables; clear datatable;
targets     = inputs(:,end-no_of_targets+1:end); 
inputs(:,end-no_of_targets+1:end)=[];
[no_of_instances,no_of_features]     = size(inputs);
inputs_normalized = (inputs-mean(inputs))./std(inputs);
inputs_normalized = [ones(no_of_instances,1),inputs_normalized];

learning_rate   = 10^-2;
max_epoch       = 10000; 
termination_threshold = 10^-6;

weights         = zeros(no_of_features+1,no_of_targets);
rms             = zeros(max_epoch,1);


for epoch = 1:max_epoch
    
    Y_predicted = inputs_normalized*weights;
    errors      = targets-Y_predicted;
    gradient    = transpose(inputs_normalized)*errors;
    delta_w     = learning_rate*gradient;
    improvement = norm(delta_w)/norm(weights);
    weights     = weights+delta_w;
    rms(epoch)  = sqrt(transpose(errors)*errors);
    if improvement<10^-6; break;end
end

epoch,rms(epoch)
rms(epoch+1:end)=[];
plot(1:length(rms),rms);figure;
scatter(Y_predicted,targets);figure;
bar(1:no_of_instances,errors);

