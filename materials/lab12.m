% Google PageRank algorithm on the example of random network

%% Load the network data
clear;


%% Display a small amount of network
NumNetwork=500;
AdjMatrixSmall=AdjMatrix(1:NumNetwork,1:NumNetwork);

for j=1:NumNetwork
    coordinates(j,1)=NumNetwork*rand;
    coordinates(j,2)=NumNetwork*rand;
end;

gplot(AdjMatrixSmall,coordinates,'k-*');

%% Check the amount of links originating from each webpage
NumLinks=sum(AdjMatrixSmall,2);

%% Create a matrix of probabilities (Google matrix)
% Element (i,j) of the matrix shows the probability of moving from i-th
% page of the network to jth page. It is assumed that the user can follow
% any link on the page with a total probability of 85% (all hyperlinks are 
% equal), and jump (teleport) to any other page in the network with a total
% probability of 15% (again, all pages are equal).

alpha=0.15; 
GoogleMatrix=zeros(NumNetwork,NumNetwork);
for i=1:NumNetwork
    if NumLinks(i)~=0
       GoogleMatrix(i,:)=AdjMatrixSmall(i,:)./NumLinks(i);
    else
       GoogleMatrix(i,:)=1./NumNetwork;
    end;
end;

GoogleMatrix=(1-alpha)*GoogleMatrix+alpha*ones(NumNetwork,NumNetwork)./NumNetwork;

%% Check that all the rows in the GoogleMatrix matrix sum to 1
SumGoogleMatrix=sum(GoogleMatrix,2);

%% Finding an eigenvector corresponding to 1 (why is there sucj an eigenvector)?
w0=ones(1,NumNetwork)./sqrt(NumNetwork);

w1=w0*GoogleMatrix;
w2=w1*GoogleMatrix;
w3=w2*GoogleMatrix;

w100=w0*(GoogleMatrix)^100;
w90=w0*(GoogleMatrix)^90;

% Check the difference between v30 and v20. Observe that it is pretty
% small
deltaw=w100-w90;

%% Compute the eigenvalues and the right eigenvectors 
[VectRight,LamRight]=eig(GoogleMatrix);
% Explain the result
LamRight=diag(LamRight);

%% Compute the eigenvalues and the left eigenvectors
[VectLeft,LamLeft]=eig(GoogleMatrix');
LamLeft=diag(LamLeft);

%% Separate the eigenvector corresponding to the eigenvalue 1 and scale it
u1=VectLeft(:,1);

u1=abs(u1)/norm(u1);

%% Select the maximum element and the corresponding element. 
%Which page is the most important in the network?

[MaxRank,PageMaxRank]=max(u1);

%% Check if it's the most popular (most linked to page):

MostLinks=sum(AdjMatrixSmall);
[MaxLinks,PageMaxLinks]=max(MostLinks);


