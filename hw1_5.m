%% load data
clear;
close all;
clc;
imgDataPath1 = 'trainset/';
imgDataPath2 = 'testset/';
imgDataDir1  = dir(imgDataPath1);             % all folders
imgDataDir2  = dir(imgDataPath2); 
train = zeros(50,50,240);
test = zeros(50,50,60);
trainvector = zeros(240,2500);
testvector = zeros(60,2500);
for i = 3:length(imgDataDir1)
    if(isequal(imgDataDir1(i).name,'.')||... % hidden
       isequal(imgDataDir1(i).name,'..')||...
       ~imgDataDir1(i).isdir)                % non-folder
        continue;
    end
    imgDir1 = dir([imgDataPath1 imgDataDir1(i).name '/*.jpg']); 
    for j =1:length(imgDir1)                 % all pictures
        train(:,:,(i-3)*40+j) = imread([imgDataPath1 imgDataDir1(i).name '/' imgDir1(j).name]);
        trainvector((i-3)*40+j,:) = reshape(train(:,:,(i-3)*40+j),[1,2500]);
    end
end
for i = 3:length(imgDataDir2)
    if(isequal(imgDataDir2(i).name,'.')||... % hidden
       isequal(imgDataDir2(i).name,'..')||...
       ~imgDataDir2(i).isdir)                % non-folder
        continue;
    end
    imgDir2 = dir([imgDataPath2 imgDataDir2(i).name '/*.jpg']); 
    for j =1:length(imgDir2)                 % all pictures
        test(:,:,(i-3)*10+j) = imread([imgDataPath2 imgDataDir2(i).name '/' imgDir2(j).name]);
        testvector((i-3)*10+j,:) = reshape(test(:,:,(i-3)*10+j),[1,2500]);
    end
end

%% PCA
covariance = cov(trainvector);
[eigenV,eigenD] = eig(covariance);
PCmatrix = zeros(50,50,16);
figure;
for i = 1:16
    PCmatrix(:,:,i) = reshape(eigenV(:,2501-i),[50,50]);
    subplot(4,4,i);
    imshow(normalize(PCmatrix(:,:,i),'range'));
end

%% RDA
s=1;
w = zeros(2500,15);
figure;
for i = 1:5
    for j = i+1:6
        miu1 = mean( trainvector((i-1)*40+1:i*40,:) )';
        miu2 = mean( trainvector((j-1)*40+1:j*40,:) )';
        sig1 = cov( trainvector((i-1)*40+1:i*40,:) );
        sig2 = cov( trainvector((j-1)*40+1:j*40,:) );
        w(:,s) = (sig1+sig2+eye(2500)) \ (miu1-miu2);
        RDA_result = normalize( reshape(w(:,s),[50,50]), 'range' );
        subplot(4,4,s);
        imshow(RDA_result);
        s=s+1;
    end
end
w = normalize(w,'norm');

%% PCA gaussian
% learning
z_pcatrain = zeros(15,40,6);
miu3 = zeros(15,6);
sig3 = zeros(15,15,6);
average = mean( trainvector )';
for c = 1:6
    for i = 1:15
        z_pcatrain(i,:,c) = eigenV(:,2501-i)' * (trainvector((c-1)*40+1:c*40,:)' - average);
        miu3(i,c) = mean(z_pcatrain(i,:,c));
    end
    sig3(:,:,c) = cov(z_pcatrain(:,:,c)');
end
% test
z_pcatest = zeros(15,60);
for i = 1:15
    z_pcatest(i,:) = eigenV(:,2501-i)' * (testvector' - average);
end
prob_pca = zeros(1,6);
pcatest_result = zeros(1,60);
pca_error = zeros(1,60);
for n = 1:60
    for c = 1:6
       prob_pca(c) = (z_pcatest(:,n)-miu3(:,c))' / sig3(:,:,c) * (z_pcatest(:,n)-miu3(:,c)) + log(det(sig3(:,:,c)));
    end
	[~,pcatest_result(n)] = min(prob_pca);
    pca_error(n) = (pcatest_result(n) ~= fix(n/10)+1);
end

%% LDA gaussian
% learning
z_ldatrain = zeros(15,40,6);
miu4 = zeros(15,6);
sig4 = zeros(15,15,6);

for c = 1:6
    for i = 1:15
        z_ldatrain(i,:,c) = w(:,i)' * (trainvector((c-1)*40+1:c*40,:)' - average);
        miu4(i,c) = mean(z_ldatrain(i,:,c));
    end
    sig4(:,:,c) = cov(z_ldatrain(:,:,c)');
end
% test
z_ldatest = zeros(15,60);
for i = 1:15
    z_ldatest(i,:) = w(:,i)' * (testvector' - average);
end
prob_lda = zeros(1,6);
ldatest_result = zeros(1,60);
lda_error = zeros(1,60);
for n = 1:60
    for c = 1:6
       prob_lda(c) = (z_ldatest(:,n)-miu4(:,c))' / sig4(:,:,c) * (z_ldatest(:,n)-miu4(:,c)) + log(det(sig4(:,:,c)));
    end
	[~,ldatest_result(n)] = min(prob_lda);
    lda_error(n) = (ldatest_result(n) ~= fix(n/10)+1);
end

%% PCA + LDA
% learning
z_mid1 = zeros(240,30);
for c = 1:6
    for i = 1:30
        z_mid1((c-1)*40+1:c*40,i) = eigenV(:,2501-i)' * (trainvector((c-1)*40+1:c*40,:)' - average);
    end
end

s=1;
z_mid2 = zeros(30,15);
for m = 1:5
    for n = m+1:6
        miu1 = mean( z_mid1((m-1)*40+1:m*40,:) )';
        miu2 = mean( z_mid1((n-1)*40+1:n*40,:) )';
        sig1 = cov( z_mid1((m-1)*40+1:m*40,:) );
        sig2 = cov( z_mid1((n-1)*40+1:n*40,:) );
        z_mid2(:,s) = (sig1+sig2) \ (miu1-miu2);
        s=s+1;
    end
end
z_mid2 = normalize(z_mid2,'norm');
z_train = zeros(15,40,6);
miu5 = zeros(15,6);
sig5 = zeros(15,15,6);
for c = 1:6
    for i = 1:15
        z_train(i,:,c) = z_mid2(:,i)' * z_mid1((c-1)*40+1:c*40,:)';
        miu5(i,c) = mean(z_train(i,:,c));
    end
    sig5(:,:,c) = cov(z_train(:,:,c)');
end
% test
z_testmid = zeros(30,60);
z_testfin = zeros(15,60);
for i = 1:30
    z_testmid(i,:) = eigenV(:,2501-i)' * (testvector' - average);
end
    z_testfin = z_mid2' * z_testmid;
prob = zeros(1,6);
test_result = zeros(1,60);
error = zeros(1,60);
for n = 1:60
    for c = 1:6
       prob(c) = (z_testfin(:,n)-miu5(:,c))' / sig5(:,:,c) * (z_testfin(:,n)-miu5(:,c)) + log(det(sig5(:,:,c)));
    end
	[~,test_result(n)] = min(prob);
    error(n) = (test_result(n) ~= fix(n/10)+1);
end