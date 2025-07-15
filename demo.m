%%
clear;clc;close all;fclose('all');

load('dataset/Edm.mat');
% load('dataset/Flare1.mat');
% load('dataset/Oes97.mat');
% load('dataset/Jura.mat');
% load('dataset/Oes10.mat');
% load('dataset/Enb.mat');
% load('dataset/Song.mat');
% load('dataset/WQplants.mat');
% load('dataset/WQanimals.mat');
% load('dataset/WaterQuality.mat');

% load('dataset/BeLaE.mat');
% load('dataset/Voice.mat');
% load('dataset/Scm20d.mat');
% load('dataset/Rf1.mat');
% load('dataset/Thyroid.mat');
% load('dataset/Pain.mat');
% load('dataset/Scm1d.mat');
% load('dataset/CoIL2000.mat');
% load('dataset/TIC2000.mat');
% load('dataset/Flickr.mat');
% load('dataset/Disfa.mat');
% load('dataset/Fera.mat');
% load('dataset/Adult.mat');
% load('dataset/Default.mat');





%%
% main
numFolds = 10;
numAlgos = 1;
fildID = 1;%for standard output, the screen
HammingScore = zeros(numFolds,numAlgos);
ExactMatch = zeros(numFolds,numAlgos);
SubExactMatch = zeros(numFolds,numAlgos);
for numFold=1:numFolds
    %% (1)Base classifier: support vector machine
    X_train = data.norm(idx_folds{numFold}.train,:);
    X_test = data.norm(idx_folds{numFold}.test,:);
    y_train = target(idx_folds{numFold}.train,:);
    y_test = target(idx_folds{numFold}.test,:);
    %parameters
    theta = 0.1;
    alpha = 0.001;
    gamma = 0.1;
    numK = 10;

    % [ Eval,y_predict ] = LEAD(X_train,y_train,X_test,y_test,theta,alpha,gamma,numK);
    [ Eval,y_predict ] = LEAD_NoKernel(X_train,y_train,X_test,y_test,theta,alpha,gamma,numK);
    disp(['HammingScore=',num2str(Eval.HS,'%4.3f'),', ExactMatch=',num2str(Eval.EM,'%4.3f'),', SubExactMatch=',num2str(Eval.SEM,'%4.3f')]);
    
    HammingScore(numFold) = Eval.HS;
    ExactMatch(numFold) = Eval.EM;
    SubExactMatch(numFold) = Eval.SEM;
    
end
%% disp
meanHS = mean(HammingScore);stdHS = std(HammingScore);
meanEM = mean(ExactMatch);stdEM = std(ExactMatch);
meanSEM = mean(SubExactMatch);stdSEM = std(SubExactMatch);
for iAlgo=1:numAlgos
    temp_str = ['Algo.',num2str(iAlgo),': HS=',num2str(meanHS(iAlgo),'%4.3f'),'±',num2str(stdHS(iAlgo),'%4.3f'),...
        ', EM=',num2str(meanEM(iAlgo),'%4.3f'),'±',num2str(stdEM(iAlgo),'%4.3f'),...
        ', SEM=',num2str(meanSEM(iAlgo),'%4.3f'),'±',num2str(stdSEM(iAlgo),'%4.3f'),'\n'];
    fprintf(fildID,temp_str);
end
Final = [num2str(meanHS(iAlgo),'%4.3f'),'±',num2str(stdHS(iAlgo),'%4.3f')
         num2str(meanEM(iAlgo),'%4.3f'),'±',num2str(stdEM(iAlgo),'%4.3f')
         num2str(meanSEM(iAlgo),'%4.3f'),'±',num2str(stdSEM(iAlgo),'%4.3f')];
fprintf('%s\n',Final(1,:));
fprintf('%s\n',Final(2,:));
fprintf('%s\n',Final(3,:));

