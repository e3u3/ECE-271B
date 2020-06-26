clear;clc;close all;
TrainNum = 20000; TestNum = 10000; LabelNum = 10; 
[TrainImg, TrainLbl] = readMNIST('training set/train-images-idx3-ubyte/train-images.idx3-ubyte', 'training set/train-labels-idx1-ubyte/train-labels.idx1-ubyte', TrainNum, 0); 
[TestImg, TestLbl] = readMNIST('test set/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte', 'test set/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte', TestNum, 0);   

target_train = - ones(TrainNum, LabelNum); 
for h = 1:TrainNum     
    target_train(h, TrainLbl(h)+1) = 1; 
end
target_test = - ones(TestNum, LabelNum); 
for h = 1:TestNum     
    target_test(h, TestLbl(h)+1) = 1; 
end

%% a b
TestPoe = zeros(3, 10);
numSV = zeros(3, 10);
pos = cell(3, 10);
neg = cell(3, 10);
C = [2,4,8];
for j = 1:3
    figure;
    for i = 1:10
        tic;
        model = svmtrain(target_train(:,i), TrainImg, ['-t 0 -c ', int2str(C(j))]);
        [predicted_label, accuracy, decision_values] = svmpredict(target_test(:,i), TestImg, model);
        TestPoe(j, i) = accuracy(1);
        numSV(j, i) = model.totalSV;
        [~, ind_max] = maxk(model.sv_coef, 3);
        [~, ind_min] = mink(model.sv_coef, 3);
        max3 = model.sv_indices(ind_max);
        min3 = model.sv_indices(ind_min);
        pos{j, i} = zeros(28, 28*3);
        neg{j, i} = zeros(28, 28*3);
        for k = 1:3
            pos{j, i}(:,k*28-27:k*28) = reshape(TrainImg(max3(k), :), [28, 28])';
            neg{j, i}(:,k*28-27:k*28) = reshape(TrainImg(min3(k), :), [28, 28])';
        end
        [pred, acc, dec] = svmpredict(target_train(:,i), TrainImg, model);
        subplot(5,2,i);
        cdfplot(dec .* target_train(:,i));
        xlabel('margin');
        title(['digit ', num2str(i-1)]);
        toc;
    end
end

%% c
TestPoe2 = zeros(1, 10);
numSV2 = zeros(1, 10);
pos2 = cell(1,10);
neg2 = cell(1,10);
figure;
for i=1:10
    model2 = svmtrain(target_train(:,i), TrainImg, '-c 2 -g 0.0625');
    [predicted_label2, accuracy2, decision_values2] = svmpredict(target_test(:,i), TestImg, model2);
    TestPoe2(i) = accuracy2(1); 
    numSV2(i) = model2.totalSV;
    [~, ind_max] = maxk(model2.sv_coef, 3);
    [~, ind_min] = mink(model2.sv_coef, 3);
    max3 = model2.sv_indices(ind_max);
    min3 = model2.sv_indices(ind_min);
    pos2{i} = zeros(28, 28*3);
    neg2{i} = zeros(28, 28*3);
    for k = 1:3
        pos2{i}(:,k*28-27:k*28) = reshape(TrainImg(max3(k), :), [28, 28])';
        neg2{i}(:,k*28-27:k*28) = reshape(TrainImg(min3(k), :), [28, 28])';
    end
    [pred2, acc2, dec2] = svmpredict(target_train(:,i), TrainImg, model2);
    subplot(5,2,i);
    cdfplot(target_train(:,i).*dec2);
    xlabel('margin');
    title(['digit ', num2str(i-1)]);
end