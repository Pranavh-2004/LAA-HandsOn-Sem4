% Facebook and graph partitioning

%% Simple graph

%% Load the data
clear;
load('social.mat','Social');
spy(Social);


%% Laplacian


%% Spectral decomposition


%% Clusteer in two groups


%% Cluster in 4 groups

Diagonal(indsmall)=large+1000;
[small,indsmall]=min(Diagonal);

V3=V(:,indsmall);

pp=[];
pn=[];
np=[];
nn=[];

for j=1:m
    if (V2(j)>0)
        if (V3(j)>0)
            pp=[pp,j];
        else
            pn=[pn,j];
        end;
    else
        if (V3(j)>0)
            np=[np,j];
        else
            nn=[nn,j];
        end;
    end;
end;

order=[pp,pn,np,nn];

iden=eye(m,m);

for j=1:351;
    for k=1:351;
        P(j,k)=iden(order(j),k);
    end;
end;

SocialOrdered=P*Social*P'

figure;
spy(SocialOrdered)

%% Second order of Fiedler

SocialPos=Social(pos,pos);
SocialNeg=Social(neg,neg);

rowsumpos=sum(SocialPos,2);

DiagSocialPos=diag(rowsumpos);

LaplaceSocialPos=DiagSocialPos-SocialPos;

[VPos,DPos]=eig(LaplaceSocialPos);

V2Pos=VPos(:,2);

[mpos,npos]=size(V2Pos);

posp=[];
posn=[];

for j=1:mpos
    if (V2Pos(j)>0)
        posp=[posp,pos(j)];
    else
        posn=[posn,pos(j)];
    end;
end;


rowsumneg=sum(SocialNeg,2);

DiagSocialNeg=diag(rowsumneg);

LaplaceSocialNeg=DiagSocialNeg-SocialNeg;

[VNeg,DNeg]=eig(LaplaceSocialNeg);

V2Neg=VNeg(:,2);

[mneg,nneg]=size(V2Neg);

negp=[];
negn=[];

for j=1:mneg
    if (V2Neg(j)>0)
        negp=[negp,neg(j)];
    else
        negn=[negn,neg(j)];
    end;
end;



ordergen=[posp,posn,negp,negn];

iden=eye(m,m);

for j=1:351;
    for k=1:351;
        P(j,k)=iden(ordergen(j),k);
    end;
end;

SocialOrderedGen=P*Social*P'
figure;
spy(SocialOrderedGen)






