clear;clc;close all;
TrainNum = 60000; TestNum = 10000; LabelNum = 10;
[TrainImg, TrainLbl] = readMNIST('training set/train-images-idx3-ubyte/train-images.idx3-ubyte', 'training set/train-labels-idx1-ubyte/train-labels.idx1-ubyte', TrainNum, 0);
[TestImg, TestLbl] = readMNIST('test set/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte', 'test set/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte', TestNum, 0);
TrainImg_biased = [TrainImg, ones(TrainNum, 1)];
TestImg_biased = [TestImg, ones(TestNum, 1)];

t = zeros(LabelNum, TrainNum);
for h=1:TrainNum
    t(TrainLbl(h)+1,h) = 1;
end

%% a2
w = randn(size(TrainImg_biased, 2), LabelNum);
rate = 10^-5;
iteration = 4000;
POE_train = zeros(iteration,1); 
POE_test = zeros(iteration,1);
for i = 1:iteration
    y = softmax((TrainImg_biased * w)');
    gradient = (t - y) * TrainImg_biased;
    w = w + rate * gradient';

    train_est = y;
    [~, train_result] = max(train_est);
    error_train = sum(train_result - 1 ~= TrainLbl');
    POE_train(i) = error_train / TrainNum;
    
    test_est = softmax((TestImg_biased * w)');
    [~, test_result] = max(test_est);
    error_test = sum(test_result - 1 ~= TestLbl');
    POE_test(i) = error_test / TestNum;
end

figure();
plot(POE_train);
hold on;
plot(POE_test); 
legend('train set', 'test set');
xlabel('Iteration'); ylabel('Probability of error');
title('Single Layer');

% b
H = [10 20 50];
w = cell(1,3); v = cell(1,3);
for c=1:3
    w{c} = randn(size(TrainImg_biased, 2), H(c));
    v{c} = randn(H(c), LabelNum);
end

% b2
rate = 10^-5;
iteration = 4000;
w{h} = w{h} + rate * gradient_w';
v{h} = v{h} + rate * gradient_v';

b3
lambda = 0.001;

b4
lambda = 0.0001;
rate = 2*10^-6;
iteration = 4000; 
POE_train = {zeros(iteration,1), zeros(iteration,1),zeros(iteration,1)};
POE_test = {zeros(iteration,1), zeros(iteration,1),zeros(iteration,1)};
for h=1:3
    for i=1:iteration
        % sigmoid
        g = TrainImg_biased * w{h};
        y = logsig(g);
        u = y * v{h};
        z = softmax(u');
        gradient_w = (y .* (1 - y))' .* (v{h} * (t-z)) * TrainImg_biased;
        gradient_v = (t - z) * y;
        w{h} = w{h} + rate * gradient_w' - lambda * 2 * w{h};
        v{h} = v{h} + rate * gradient_v' - lambda * 2 * v{h};
        
%         %% ReLU
%         g = TrainImg_biased * w{h};
%         y = max(g,0);
%         u = y * v{h};
%         z = softmax(u');
%         gradient_w = sign(y)' .* (v{h} * (t-z)) * TrainImg_biased;
%         gradient_v = (t - z) * y;
%         w{h} = w{h} + rate * gradient_w' - lambda * 2 * w{h};
%         v{h} = v{h} + rate * gradient_v' - lambda * 2 * v{h};   
        
        train_est = z;
        [~, train_result] = max(train_est);
        error_train = sum(train_result - 1 ~= TrainLbl');
        POE_train{h}(i) = error_train / TrainNum;

        test_est = softmax( (max(TestImg_biased * w{h}, 0) * v{h})' );
        [~, test_result] = max(test_est);
        error_test = sum(test_result - 1 ~= TestLbl');
        POE_test{h}(i) = error_test / TestNum;     
    end
end
for k=1:3
    figure();
    plot(POE_train{k}); hold on;
    plot(POE_test{k}); 
    legend('train set', 'test set');
    xlabel('Iteration'); ylabel('Probability of error');
    title(['ReLU with H=',num2str(H(k)),' lambda=',num2str(lambda)]);
end

% c
% Single layer SGD        
w = randn(size(TrainImg_biased, 2), LabelNum);
rate = 10^-2; epoch = 20; step=20;
POE_test = zeros(epoch * TrainNum / step, 1); 
POE_train = zeros(epoch * TrainNum / step, 1);
for s = 1:epoch
    for i = 1:TrainNum
        y = softmax((TrainImg_biased(i,:) * w)');
        gradient = (t(:,i) - y) * TrainImg_biased(i,:);
        w = w + rate * gradient';
        if mod(i, step) == 0
            train_est = softmax((TrainImg_biased * w)');
            [~, train_result] = max(train_est);
            error_train = sum(train_result - 1 ~= TrainLbl');
            POE_train((s - 1) * TrainNum / step + fix(i / step)) = error_train / TrainNum;

            test_est = softmax((TestImg_biased * w)');
            [~, test_result] = max(test_est);
            error_test = sum(test_result - 1 ~= TestLbl');
            POE_test((s - 1) * TrainNum / step + fix(i / step)) = error_test / TestNum;
        end
    end
end
figure();
plot(POE_train); hold on;
plot(POE_test); 
xticks(0:epoch * TrainNum / step / 4:epoch * TrainNum / step);
xticklabels({'0','5','10','15','20'});
legend('train set', 'test set');
xlabel('Epoch'); ylabel('Probability of error');
title('Single layer SGD');

H = [10 20 50];
w = cell(1,3); v = cell(1,3);
for c=1:3
    w{c} = randn(size(TrainImg_biased, 2), H(c));
    v{c} = randn(H(c), LabelNum);
end

rate = 10^-2; epoch = 20; step=20;
POE_train = {zeros(epoch * TrainNum / step,1), zeros(epoch * TrainNum / step,1), zeros(epoch * TrainNum / step,1)};
POE_test = {zeros(epoch * TrainNum / step,1), zeros(epoch * TrainNum / step,1), zeros(epoch * TrainNum / step,1)};
for h=1:3
    for s = 1:epoch
        for i = 1:TrainNum
            % sigmoid    rate \ title
            g = TrainImg_biased(i,:) * w{h};
            y = logsig(g);
            u = y * v{h};
            z = softmax(u');
            gradient_w = (y .* (1 - y))' .* (v{h} * (t(:,i) - z)) * TrainImg_biased(i,:);
            gradient_v = (t(:,i) - z) * y;
            w{h} = w{h} + rate * gradient_w';
            v{h} = v{h} + rate * gradient_v';

            if mod(i, step) == 0
            train_est = softmax( (logsig(TrainImg_biased * w{h}) * v{h})' );
            [~, train_result] = max(train_est);
            error_train = sum(train_result - 1 ~= TrainLbl');
            POE_train{h}((s - 1) * TrainNum / step + fix(i / step)) = error_train / TrainNum;

            test_est = softmax( (logsig(TestImg_biased * w{h}) * v{h})' );
            [~, test_result] = max(test_est);
            error_test = sum(test_result - 1 ~= TestLbl');
            POE_test{h}((s - 1) * TrainNum / step + fix(i / step)) = error_test / TestNum;
            end

            % ReLU
            g = TrainImg_biased(i,:) * w{h};
            y = max(g,0);
            u = y * v{h};
            z = softmax(u');
            gradient_w = sign(y)' .* (v{h} * (t(:,i) - z)) * TrainImg_biased(i,:);
            gradient_v = (t(:,i) - z) * y;
            w{h} = w{h} + rate * gradient_w';
            v{h} = v{h} + rate * gradient_v';   

            if mod(i, step) == 0
            train_est = softmax( (max(TrainImg_biased * w{h}, 0) * v{h})' );
            [~, train_result] = max(train_est);
            error_train = sum(train_result - 1 ~= TrainLbl');
            POE_train{h}((s - 1) * TrainNum / step + fix(i / step)) = error_train / TrainNum;

            test_est = softmax( (max(TestImg_biased * w{h}, 0) * v{h})' );
            [~, test_result] = max(test_est);
            error_test = sum(test_result - 1 ~= TestLbl');
            POE_test{h}((s - 1) * TrainNum / step + fix(i / step)) = error_test / TestNum;
            end
        end
    end
end
for k=1:3
    figure();
    plot(POE_train{k}); hold on;
    plot(POE_test{k}); 
    xticks(0:epoch * TrainNum / step / 4:epoch * TrainNum / step);
    xticklabels({'0','5','10','15','20'});
    legend('train set', 'test set');
    xlabel('Epoch'); ylabel('Probability of error');
    title(['Sigmoid SGD with H=',num2str(H(k))]);
end