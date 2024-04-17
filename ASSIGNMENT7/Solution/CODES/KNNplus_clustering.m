function clusters = KNNplus_clustering(data,no_of_clusters,varargin)

    %data       = (data-mean(data))./std(data);
    [nI,nF]     = size(data);
    
    %% Parameters defination
    K           = no_of_clusters; 
    if K==1
           clusters.cluster_nos = ones(nI,1);
           clusters.numbers     = nI;
           clusters.percentage  = 100;
           clusters.centroids   = mean(data);
           diff                 = data-clusters.centroids;
           clusters.variance    = 0.5*sum(diff.*diff,'all')/nI;
           return;
    end
    dist_old    = 10^6;
    termination_threshold = 10^-8;

    %% Parameters Initiazation
    cluster_nos = zeros(nI,1);
    numbers     = zeros(K,1);
    percentage  = zeros(K,1);
    threshold   = 100;
    iterations  = 0;

    %% Initial centroids 
    if isempty(varargin) %for default, use KNN++ approach
        centroids       = zeros(K,nF);
        centroids(1,:)       = data(randi(nI),:);
        dist=zeros(nI,1);
        for i=2:K
            for m = 1:nI 
               instance = data(m,:);
               delta    = centroids(1:i-1,:)-instance;
               euclid   = sqrt(sum(delta.*delta,2));
               dist(m)  = min(euclid);
            end
            [~,n]=max(dist);
            centroids(i,:)=data(n,:);
        end
    elseif length(varargin)==1
        [nC,nDc]  = size(varargin{1});
        if or(nC~=K,nDc~=nF)
            disp('clustering unsuccessful:centroid dimensions do not match');
            return;
        else
            centroids = varargin{1};
        end
    else
        disp('clustering unsuccessful:too many inputs');
        return;
    end 
    %% Clustering
    while threshold > termination_threshold
        % Finding Euclidean distance of data points
        dist_new=0;
        if iterations~=0 
           for j=1:K
            instances     = data(cluster_nos==j,:);
            centroids(j,:)= mean(instances,1);
           end
        end
        for i=1:nI
           instance = data(i,:);
           delta    = (centroids-instance);
           euclid   = sqrt(sum(delta.*delta,2));
           [min_dist,cluster_nos(i)]=min(euclid);
           dist_new = dist_new+min_dist;
        end
        % Finding new centroids
        threshold = (abs(dist_old-dist_new)/dist_old);
        dist_old  = dist_new;
        iterations = iterations+1;
    end
    disp(iterations);
    clusters.cluster_nos = cluster_nos;
    clusters.numbers     = zeros(K,1);
    clusters.percentage  = zeros(K,1);
    clusters.centroids   = centroids;
    clusters.variance    = zeros(K,1);
    for j=1:K
        instances     = data(cluster_nos==j,:);
        delta         = instances-centroids(j,:);
        clusters.numbers(j)  = size(instances,1);
        clusters.percentage(j)= round(clusters.numbers(j)/nI*100,1);
        clusters.variance(j) = 0.5*sum(delta.*delta,'all')/clusters.numbers(j);
    end
end