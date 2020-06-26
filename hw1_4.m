clear;clc;
alpha = [10,2];
sigma = [2,10];
for i = 1:2
miu = [alpha(i);0];
gamma = [1,0;0,sigma(i)];
x1 = mvnrnd(miu,gamma,500);
x2 = mvnrnd(-miu,gamma,500);
subplot(1,2,i);
plot(x1(:,1),x1(:,2),'o');
hold on;
plot(x2(:,1),x2(:,2),'x');
hold on;
x = [x1;x2];

%% PCA
covariance = cov(x,1);
[eigenV,eigenD] = eig(covariance);
quiver(eigenV(1,2),eigenV(2,2),5,'black');

%% LDA
w = inv(2*gamma) * (2*miu);
quiver(w(1),w(2),'black');
end