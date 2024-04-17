%% PERCEPTRON MODEL FITTING
%% Asssumptions:
%  1. multiple output features
%  2. binary classification (value either 0 or 1)
%  3. stochastic gradient for backpropogation

clc;clearvars;close all;
infile      = 'Perceptron_multi_target.xlsx';
no_of_targets = 2;

datatable   = readtable(infile);
headers     = datatable.Properties.VariableNames; headers(:,end)=[];
inputs      = datatable.Variables; clear datatable;
targets     = inputs(:,end-no_of_targets+1:end); 
inputs(:,end-no_of_targets+1:end)=[];
[no_of_instances,no_of_features]     = size(inputs);
inputs_normalized = (inputs-mean(inputs))./std(inputs);
inputs_normalized = [ones(no_of_instances,1),inputs_normalized];

weights     = zeros(no_of_features+1,no_of_targets);
learning_rate   = 0.01;
max_epoch       = 10000; 
target_accuracy = 90;
total_classifications = no_of_instances*no_of_targets;

for epoch = 1:max_epoch
    
    for instance_number = 1:no_of_instances
        instance_vector = inputs_normalized(instance_number,:);
        z           = instance_vector*weights;
        y_predicted = signum_activation(z);
        y_actual    = targets(instance_number,:);
        error       = y_actual - y_predicted;
        weights     = weights+learning_rate*(transpose(instance_vector)*error); 
    end
        Y_predicted = signum_activation(inputs_normalized*weights);
        error_all   = targets-Y_predicted;
        misclassifieds = 0.5*sum(sum(abs(error_all)));
        accuracy = (total_classifications-misclassifieds)/total_classifications*100;
        if accuracy >= target_accuracy
            break;
        end
end
 
 
disp(accuracy, epoch)

confusion_matrix = zeros(2);
for target_number = 1:no_of_targets
    for instance_number=1:no_of_instances
       y_predicted = Y_predicted(instance_number,target_number);
       y_actual = targets(instance_number,target_number);
       m = 0.5*(y_predicted+1)+1; n = 0.5*(y_actual+1)+1;
       confusion_matrix(m,n)=confusion_matrix(m,n)+1;
       x1=inputs(instance_number,1);x2=inputs(instance_number,2);
       if y_predicted==-1
           plot(x1,x2,'*r')
       else
           plot(x1,x2,'*g')
       end
       hold on
    end
    figure;
end

disp(confusion_matrix)

function neuron_outputs = signum_activation(neuron_inputs)
 
    [no_of_neurons,no_of_instances] = size(neuron_inputs);
    neuron_outputs  = ones(no_of_neurons,no_of_instances);
    neuron_outputs(neuron_inputs < 0)= -1;

end
