function [Eval, y_predict] = LEAD(X_train, y_train, X_test, y_test, theta, alpha, gamma, numK)
%LEAD Implements the LEAD algorithm for Multi-Dimensional Classification.
%   This is the main implementation including the kernel-based method.
%
%   Inputs:
%       X_train, y_train, X_test, y_test: Training and testing data.
%       theta, alpha, gamma: Model hyperparameters (\lambda, \alpha, \gamma).
%       numK: Number of nearest neighbors for LLE.
%   Outputs:
%       Eval: Struct with evaluation metrics (HS, EM).
%       y_predict: Predicted labels for the test set.

    head_str = '      ';

    % Obtain parameters of data sets
    num_training = size(X_train,1);%number of training examples
    num_features = size(X_train,2);%number of input features
    num_dim = size(y_train,2);     %number of dimensions(class variables)
    num_testing = size(X_test,1);  %number of testing examples
    C_per_dim = cell(num_dim,1);   %class labels in each dimension
    num_per_dim = zeros(num_dim,1);%number of class labels in each dimension
    for dd=1:num_dim
        temp = y_train(:,dd);
        C_per_dim{dd} = unique(temp);
        num_per_dim(dd) = length(C_per_dim{dd});
    end
    
    %%
    % (S1) one-hot Label Encoding
    temp_str = [head_str,'Step 1: One-Hot Label Encoding (',disp_time(clock,0),')...\n'];
    fprintf(temp_str);
    
    R_encoded = zeros(num_training,sum(num_per_dim));  % Encoded label matrix 
    for d1 = 1:num_dim
        col_begin = 1 + sum(num_per_dim(1:d1-1));
        col_end = sum(num_per_dim(1:d1));
        dim_per_col = num_per_dim(d1);
        [~,~,y_pair_cp] = unique(y_train(:,d1),'rows');
        tmp_y = zeros(size(y_pair_cp,1),dim_per_col);
        for i = 1:num_training
            tmp_y(i,y_pair_cp(i)) = 1;
        end
        R_encoded(:,col_begin:col_end) = tmp_y;
    end
    
    %%
    temp_str = [head_str,'Step 2: LLE (',disp_time(clock,0),')...\n'];
    fprintf(temp_str);
    % Obtain the linear combination coefficients matrix S
    [idx_train, dist_train] = knnsearch(X_train,X_train,'K',numK+1);
    S = zeros(num_training);
    for itrain=1:num_training
        dist0_idx = (dist_train(itrain,2:end)==0);%excluding the instance itself
        if sum(dist0_idx)>0%if there are other instances who are exactly identical with the current one
            si = zeros(numK,1);
            si(dist0_idx) = 1/sum(dist0_idx);
        else
            XK = transpose(X_train(idx_train(itrain,2:end),:));
            Xi = repmat(transpose(X_train(itrain,:)),1,numK);
            Di = (Xi-XK)'*(Xi-XK);
            si = pinv(Di)*ones(numK,1)/(ones(1,numK)*pinv(Di)*ones(numK,1));
        end
        cc = idx_train(itrain,2:end);
        S(idx_train(itrain,2:end),itrain) = si;%si/sum(si);
    end
    % Obtain the Embedded label matrix
    St = (eye(num_training)-S)*(eye(num_training)-S)'+theta*eye(num_training);
    V_train = theta*pinv(St)*R_encoded;
    
    %%
    temp_str = [head_str,'Step 3: Predictive Model Induction (',disp_time(clock,0),')...\n'];
    fprintf(temp_str);
    Kpara.type = 'RBF';%RBF kernel
    if num_training<2000
        Kpara.gamma  = 1/2/std(pdist(X_train))^2; %parameter of kernel function
    else
        Kpara.gamma  = 1/2/std(pdist(X_train(1:2000,:)))^2; %parameter of kernel function
    end
    % label correlation matrix (knn)
    C2 = 1 - pdist(V_train','cosine');
    C = squareform(C2);  % label correlation matrix (consin)
    LC = calculateL(C);
    K = Kernel(X_train,X_train,Kpara);%Kernel matrix
    S_A = K + gamma*eye(num_training);
    S_B = alpha*(LC + LC');
    S_C = V_train;
    Theta = sylvester(S_A,S_B,S_C);clear S_A S_B S_C;
    
    %%
    temp_str = [head_str,'Step 4: Testing phase (',disp_time(clock,0),')...\n'];
    fprintf(temp_str);
    V_test = Kernel(X_test,X_train,Kpara)*Theta;
    y_predict = zeros(size(y_test)); % predicted original label vector
    
    % one-hot label decoding
    for ii=1:num_testing
        dim_begin = 1;
        for dd=1:num_dim
            y_test_one_dim = V_test(ii,(dim_begin:dim_begin+num_per_dim(dd)-1));
            [~,one_dim_max_index] = max(y_test_one_dim);
            y_predict(ii,dd) = one_dim_max_index;
            dim_begin = dim_begin + num_per_dim(dd);
        end
    end
    
    %Hamming Score(or Class Accuracy)
    Eval.HS = sum(sum(y_predict==y_test))/(size(y_test,1)*size(y_test,2));
    %Exact Match(or Example Accuracy or Subset Accuracy)
    Eval.EM = sum(sum((y_predict==y_test),2)==size(y_test,2))/size(y_test,1);
    %Sub-ExactMatch
    Eval.SEM = sum(sum((y_predict==y_test),2)>=(size(y_test,2)-1))/size(y_test,1);
end