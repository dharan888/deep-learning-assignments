clc;clearvars;close all;clear figure;
infile         = 'clust_data.xlsx';

%% Reading data 
datatable   = readtable(infile);
headers     = datatable.Properties.VariableNames; headers(:,end)=[];
data        = datatable.Variables; clear datatable;

%% Calling Clustering function
no_of_clusters = 3;
%intial_centroids=[5,5;8,8];
clusters = KNNplus_clustering(data,no_of_clusters);
%clusters  = KNNplus_clustering(data,no_of_clusters,intial_centroids);

%% Plot the final clusters
color = ['r','b','g','k','y'];
lgd_txt=strings(1,no_of_clusters*2);
plots=gobjects(no_of_clusters*2,1);
centroids=round(clusters.centroids,1);
K=no_of_clusters;
for i = 1:K
    Xc= centroids(i,1);
    Yc= centroids(i,2);
    X = data(clusters.cluster_nos==i,1);
    Y = data(clusters.cluster_nos==i,2);
    no= strcat('#',string(clusters.numbers(i)));
    pe= strcat(string(clusters.percentage(i)),'%');
    lgd_txt(i)=strcat('G',string(i),{'  '},no,{'  '},pe);
    tx= string(centroids(i,1));
    ty= string(centroids(i,2));
    lgd_txt(i+K)=strcat('C',string(i),':',{'  '},tx,{'  '},ty);
    hold on  
    plots(i)=scatter(X,Y,color(i));
    plots(i+K)=scatter(Xc,Yc,color(i),'filled');
end
title_text = strcat(string(no_of_clusters),{' '},'CLUSTERS');
xlabel('X1');ylabel('X2');title(title_text);
legend(plots,lgd_txt,'Location','northwest');
hold off