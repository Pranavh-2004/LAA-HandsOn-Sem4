% Simple face recognition algorithm

%% Input database files into the Matlab
clear;
Database_Size=30;

%Reading images from the database. The image files should be located in the
%subfolder "database"
for j=1:Database_Size
    image_read=imread(['database\person' num2str(j) '.pgm']);
    [m,n]=size(image_read);
    P(:,j)=reshape(image_read,m*n,1);
end;

%% Computing and displaying the mean face


%% Subtract the mean face


%% Compute the covariance matrix of the set and its eigenvalues and eigenvectors


%% Display the set of eigenvaces
for j=2:Database_Size;
    if j==2
        EigenFaces=reshape(EigenVectors(:,j)+mean_face,m,n);
    else
        EigenFaces=[EigenFaces reshape(EigenVectors(:,j)+mean_face,m,n)];
    end;
end

EigenFaces=uint8(EigenFaces);

figure;
imshow(EigenFaces);

%% Orthogonality and symmetry

%% Recognition of an altered image (sunglasses)
image_read=imread(['database\person30altered1.pgm']);
U=reshape(image_read,m*n,1);

NormsEigenVectors=diag(Products);
W=(EigenVectors'*(double(U)-mean_face));
W=W./NormsEigenVectors;

U_approx=EigenVectors*W+mean_face;

image_approx=uint8(reshape(U_approx,m,n));
figure;
imshow([image_read,image_approx])

%% Recognition of an altered image (lower part of the face)


%% Recognition of a person not in the database







