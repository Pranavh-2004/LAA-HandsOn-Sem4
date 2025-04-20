% The Chaos Game

%% Sierpinski triangle

%% Sierpinski triangle with a change of ratio


%% Sierpinski triangle with rotation

%% Square


%% Square with different ratio


%% Square with vertex preference (new vertex cannot be the same as previous)
clear;

T=[1/2 0; 0 1/2];

t=linspace(0,2*pi,5);
t(5)=[];
v=[cos(t); sin(t)];

x(:,1)=[rand-0.5; rand-0.5];

plot(v(1,:),v(2,:),'k*',x(1,1),x(2,1),'b.');
axis equal;
hold on;

Num=5000;

k1=0;
for j=1:Num
    k=randi(3);
    if (k>=k1)
        k=k+1;
    end;
    x(:,j+1)=T*(x(:,j)-v(:,k))+v(:,k);
    k1=k;
end;

plot(x(1,:),x(2,:),'b.');
hold off;

%% Square with vertex preference (new vertex cannot be opposite of the previous)
clear;

T=[1/2 0; 0 1/2];

t=linspace(0,2*pi,5);
t(5)=[];
v=[cos(t); sin(t)];

x(:,1)=[rand-0.5; rand-0.5];

plot(v(1,:),v(2,:),'k*',x(1,1),x(2,1),'b.');
axis equal;
hold on;

Num=5000;

w=[real(v(1));imag(v(1))];
k1=0;

for j=1:Num
    k=randi(4)
    if (k~=k1+2)&&(k1~=k+2)
        x(:,j+1)=T*(x(:,j)-v(:,k))+v(:,k);
        k1=k;
    else
        x(:,j+1)=x(:,j);
    end;
end;

plot(x(1,:),x(2,:),'b.');
hold off;

%% Barnsley fern
clear;

T1=[0.85,0.04;-0.04,0.85];
T2=[-0.15,0.28;0.26,0.24];
T3=[0.2,-0.26;0.23,0.22];
T4=[0,0;0,0.16];
Q1=[0;1.64];
Q2=[-0.028;1.05];
Q3=[0;1.6];
Q4=[0,0];
P1=0.85;
P2=0.07;
P3=0.07;
P4=0.01;

Num=15000;
x(:,1)=rand(2,1);
plot(x(1,:),x(2,:),'b.')
axis equal;
hold on;
for j=1:Num
    r=rand;
    if r<=P1 
        x(:,j+1)=T1*x(:,j)+Q1;
    elseif r<=P1+P2 
        x(:,j+1)=T2*x(:,j)+Q2;
    elseif r<=P1+P2+P3 
        x(:,j+1)=T3*x(:,j)+Q3;
    else
        x(:,j+1)=T4*x(:,j);
    end
end

plot(x(1,:),x(2,:),'b.');
hold off;



%% Hexagon
clear;

T=[1/3 0; 0 1/3];

t=linspace(0,2*pi,7);
t(7)=[];
v=[cos(t); sin(t)];

x(:,1)=[rand-0.5; rand-0.5];

plot(v(1,:),v(2,:),'k*',x(1,1),x(2,1),'b.');
axis equal;
hold on;

Num=10000;

for j=1:Num
    k=randi(6);
    x(:,j+1)=T*(x(:,j)-v(:,k))+v(:,k);
end;
plot(x(1,:),x(2,:),'b.');
hold off;


%% Pentagon with a skipped vertex

clear;
T=[1/2.5 0; 0 1/2.5];

t=linspace(0,2*pi,6);
t(6)=[];
v=[cos(t); sin(t)];

x(:,1)=[rand-0.5; rand-0.5];

plot(v(1,:),v(2,:),'k*',x(1,1),x(2,1),'b.');
axis equal;
hold on;

Num=10000;

k1=0;

for j=1:Num
    k=randi(5);
    if k~=k1
        x(:,j+1)=T*(x(:,j)-v(:,k))+v(:,k);
        k1=k;
    else
        x(:,j+1)=x(:,j);
    end;
end;
plot(x(1,:),x(2,:),'b.');
hold off;

