%MLP linear
clc
clear all
% data=xlsreaqd('data_data.xlsx');
data=xlsread('data_data.xlsx');

[m,n]=size(data);
for i=1:n
    data(:,i)=data(:,i)-mean(data(:,i))/sqrt(var(data(:,i)));
end

mu=10^-2;
epo_max=2000;
data1=data(:,1:n-1);
epo=0;
eps=0.0001;
b=1;
%weights for 1st neuron 1st layer
W_1_1=ones(2,1)*eps;
%bias for 1st neuron 1st layer
b_1_1=eps;
%weights for 2nd neuron 1st layer
W_1_2=ones(2,1)*eps;
%bias for 2nd neuron 1st layer
b_1_2=eps;
%weights for 1st neuron 2nd layer
W_2_1=ones(2,1)*eps;
%bias for 1st neuron 2nd layer
b_2=eps;


while true
    epo=epo+1;
    de1_2_1=zeros(m,1);
    de1_1_1=zeros(m,1);
    de1_1_2=zeros(m,1);
    
    for i=1:m
        %Evaluation the outputs - note that there are no activation
        %function
        z1_1(i,1)=b_1_1+data1(i,:)*W_1_1;
        z1_1(i,2)=b_1_2+data1(i,:)*W_1_2;
        
        z2_1(i)=z1_1(i,:)*W_2_1+b_2;
        % terms in the summation
        de1_2_1(i)=(z2_1(i)-data(i,n));
        de1_1_1(i)=de1_2_1(i)*W_2_1(1,1);
        de1_1_2(i)=de1_2_1(i)*W_2_1(2,1);
        
    end
    de1_W_1_1=zeros(2,1);
    de1_W_1_2=zeros(2,1);
    de1_W_2_1=zeros(2,1);
    de1_b_1_1=0;
    de1_b_1_2=0;
    de1_b_2_1=0;
    for i=1:m
        % update for the weights in the 1st layer
        de1_W_1_1=de1_W_1_1-mu*de1_1_1(i)*data1(i,:)';
        de1_W_1_2=de1_W_1_2-mu*de1_1_2(i)*data1(i,:)';
        % update for the weights in the 2nd layer
        de1_W_2_1=de1_W_2_1-mu*de1_2_1(i)*z1_1(i,:)';
        % update bias
        de1_b_1_1=de1_b_1_1-mu*de1_1_1(i);
        de1_b_1_2=de1_b_1_2-mu*de1_1_2(i);
        de1_b_2_1=de1_b_2_1-mu*de1_2_1(i);
    end
    % new weights and bias
    W_1_1=W_1_1+de1_W_1_1/m;
    W_1_2=W_1_2+de1_W_1_2/m;
    W_2_1=W_2_1+de1_W_2_1/m;
    b_1_1=b_1_1+de1_b_1_1/m;
    b_1_2=b_1_2+de1_b_1_2/m;
    b_2=b_2+de1_b_2_1/m;
    %square-root of sum of squares - error
    per(epo,1)=sqrt(dot((z2_1'-data(:,n)),(z2_1'-data(:,n))));
    per(epo,2)=epo;
    if norm(z2_1-data(:,n))/norm(data(:,n))<10^-3||epo>epo_max
        z2_1'
        epo
        break
    end
end
[a,b]=min(per)
% plot(per(:,2),per(:,1))
% hold on
scatter3(data(:,1),data(:,2),data(:,3))
%scatter3(data(:,1),data(:,2),z2_1)
% hold off


