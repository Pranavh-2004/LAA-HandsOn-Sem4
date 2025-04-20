% Orthogonal matrices and 3d rotation

%%
clear;
Vertices=[1 1 1;...
    -1 1 1;...
    1 -1 1;...
    1 1 -1;...
    -1 -1 1;...
    -1 1 -1;...
    1 -1 -1;...
    -1 -1 -1];

Edges=zeros(8,8);
Edges(1,2)=1;
Edges(1,3)=1;
Edges(1,4)=1;
Edges(2,5)=1;
Edges(2,6)=1;
Edges(3,5)=1;
Edges(3,7)=1;
Edges(4,6)=1;
Edges(4,7)=1;
Edges(5,8)=1;
Edges(6,8)=1;
Edges(7,8)=1;

Edges=Edges+Edges';

theta1=pi/3;
theta2=pi/4;
theta3=pi/6;

rotmat=rotation(theta1,theta2,theta3);

VertRot=Vertices*rotmat;

figure;
axis equal;
hold on;

for j=1:8
    for k=j+1:8
        if (Edges(j,k)==1)
            line([VertRot(j,1),VertRot(k,1)],[VertRot(j,2),VertRot(k,2)]);%,[VertRot(j,3),VertRot(k,3)]);
        end;
    end;
end;

hold off;




%% Buckyball
[Edges,Vertices] = bucky;
figure;
axis equal;
hold on;

theta1=pi/3;
theta2=pi/4;
theta3=pi/6;

rotmat=rotation(theta1,theta2,theta3);

VertRot=Vertices*rotmat;

for j=1:60
    for k=j+1:60
        if (Edges(j,k)==1)
            line([VertRot(j,1),VertRot(k,1)],[VertRot(j,2),VertRot(k,2)]);%,[VertRot(j,3),VertRot(k,3)]);
        end;
    end;
end;

hold off;

figure;
axis equal;
hold on;

for j=1:60
    for k=j+1:60
        if (Edges(j,k)==1)
            line([Vertices(j,1),Vertices(k,1)],[Vertices(j,2),Vertices(k,2)],[Vertices(j,3),Vertices(k,3)]);
        end;
    end;
end;

hold off;


%% 3D fawn
clear;

load('v.mat','v');
load('f.mat','f');

[mVert,nVert]=size(v);

[mFace,nFace]=size(f);


figure;
axis equal;
hold on;
for j=1:mFace
    line([v(f(j,1),1),v(f(j,2),1)],[v(f(j,1),2),v(f(j,2),2)],[v(f(j,1),3),v(f(j,2),3)]);
    line([v(f(j,1),1),v(f(j,3),1)],[v(f(j,1),2),v(f(j,3),2)],[v(f(j,1),3),v(f(j,3),3)]);
    line([v(f(j,2),1),v(f(j,3),1)],[v(f(j,2),2),v(f(j,3),2)],[v(f(j,2),3),v(f(j,3),3)]);    
end;
hold off;

figure;
axis equal;
hold on;

theta1=pi/3;
theta2=pi/4;
theta3=pi/2;

rotmat=rotation(theta1,theta2,theta3);

VertRot=v*rotmat;

for j=1:mFace
    line([VertRot(f(j,1),1),VertRot(f(j,2),1)],[VertRot(f(j,1),2),VertRot(f(j,2),2)]);
    line([VertRot(f(j,1),1),VertRot(f(j,3),1)],[VertRot(f(j,1),2),VertRot(f(j,3),2)]);
    line([VertRot(f(j,2),1),VertRot(f(j,3),1)],[VertRot(f(j,2),2),VertRot(f(j,3),2)]);
end;


