%% PERCEPTRON MODEL FITTING
%% Asssumptions:
%  1. only one output feature
%  2. binary classification (value either 0 or 1)
%  3. stochastic gradient for backpropogation

clc;clearvars;close all;
infile      = 'Perceptron_single_target.xlsx';

% reading the data as a datatable
datatable   = readtable(infile);
% getting the input feature column names
headers     = datatable.Properties.VariableNames; headers(:,end)=[];
% getting the input feature values from the table
inputs      = datatable.Variables; clear datatable;
% divinding input features and target features separately
target      = inputs(:,end); inputs(:,end)=[];
% find the no. of instances and no. of features 
[no_of_instances,no_of_features]     = size(inputs);
% adding 1's column to the input for adding bias
inputs      = [ones(no_of_instances,1),inputs];
% initiating the weight matrix 
weights     = zeros(no_of_features+1,1);
% defining the neural network training parameters
max_epoch       = 1000; 
target_accuracy = 95;
total_classifications = no_of_instances;
% looping through expochs
for epoch = 1:max_epoch
  %  model fitting
  for instance_number = 1:no_of_instances
      % feedforward operation 
      instance_vector = inputs(instance_number,:);
      z  = instance_vector*weights;
      y_predicted = signum_activation(z);
      y_actual    = target(instance_number);
      % backpropogation - stochastic gradient
      error       = y_actual - y_predicted;
      weights     = weights + error*instance_vector'; 
  end
      % model prediction
      Y_predicted = signum_activation(inputs*weights);
      error_all   = target-Y_predicted;
      % computing the accuracy
      misclassifieds = sum(abs(error_all));
      accuracy = (total_classifications-misclassifieds)/total_classifications*100;
      if accuracy >= target_accuracy
           break;
      end
end

% constructing the confusion matrix
confusion_matrix=zeros(2);
for instance_number = 1:no_of_instances
      y_predicted = Y_predicted(instance_number);
      y_actual    = target(instance_number);
      m = y_predicted+1; n = y_actual+1;  % predicted - rows , actual - columns
      confusion_matrix(m,n)=confusion_matrix(m,n)+1; % updation
end

% plotting the decision surface
for i=1:100
    for j=1:100
        instance_vector = [1 i j];
        z = dot(weights,instance_vector);
        if z<0
           plot(i,j,'*r')
        else
           plot(i,j,'*g')
        end
        hold on
    end
end

% applying signum activation to obtain value between 0 and 1
function neuron_outputs = signum_activation(neuron_inputs)
    [no_of_neurons,no_of_instances] = size(neuron_inputs);
    neuron_outputs  = ones(no_of_neurons,no_of_instances);
    neuron_outputs(neuron_inputs < 0)= 0;
end
