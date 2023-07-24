%% Synthetic Minority Over-sampling Technique (SMOTE) for Synthetic Data Generation (SDG)
clear;
tic
load('fisheriris.mat');
Input=meas;
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target';
% Define amount of synthetic data to be generate
% 1.5 means Synthesizes 150% new observations - 1 means double the size of original
SyntheticSamples=5;
% Run SMOTE function
[Synthetic,Labels] = smote(Input, SyntheticSamples, 'Class', Target);

%% Plot data and classes
Feature1=2;
Feature2=4;
% Original
f1=Input(:,Feature1); % feature1
f2=Input(:,Feature2); % feature 2
% lbl=datatemp(:,end); % labels
% Synthetic 
ff1=Synthetic(:,Feature1); % feature1
ff2=Synthetic(:,Feature2); % feature 2
lbl2=Labels; % labels
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
plot(Input, 'linewidth',1); title('Original Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,2)
plot(Synthetic(:,1:end-1), 'linewidth',1); title('Synthetic Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,3)
gscatter(f1,f2,Target,'rkgb','.',20); title('Original');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,4)
gscatter(ff1,ff2,Labels,'rkgb','.',20); title('Synthetic');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;

%% Train and Test
% Training Synthetic dataset by SVM
Mdlsvm  = fitcecoc(Synthetic,Labels); CVMdlsvm = crossval(Mdlsvm); 
SVMError = kfoldLoss(CVMdlsvm); SVMAccOrgTrain = (1 - SVMError)*100;
% Predict for Synthetic dataset - on whole original dataset SVM
[label5,score5,cost5] = predict(Mdlsvm,Input);
% For Synthetic data svm
sizlbl=size(Target); sizlbl=sizlbl(1,1);
countersvm=0; % Misclassifications places
misindexsvm=0; % Misclassifications indexes
for i=1:sizlbl
if Target(i)~=label5(i)
misindex(i)=i; countersvm=countersvm+1; end; end
% Test accuracy Synthetic svm
TestErrAugsvm = countersvm*100/sizlbl; SVMAccAugTest = 100 - TestErrAugsvm;
% Result SVM
AugRessvm = [' Synthetic Train SVM "',num2str(SVMAccOrgTrain),'" Synthetic Test SVM"', num2str(SVMAccAugTest),'"'];
disp(AugRessvm);
toc
