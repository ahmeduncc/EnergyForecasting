%% Feedforward neural network forecasting example: 
% ISO New England dataset from years 2004 to 2008

%% Initialization
clear; close all; clc;

%% Load ISO New England dataset

load '..\..\..\Examples\Electricity Load & Price Forecasting\Load\Data\DBLoadData.mat';
% addpath 'G:\Daten\Programming\Matlab\Examples\Electricity Load & Price Forecasting\Util'

dates = datetime(data.NumDate,'ConvertFrom','datenum');
target = data.SYSLoad;

%% create feature matrix

% weather data
temperature = data.DryBulb; 
dewPoint = data.DewPnt; 

% date predictors
hourOfDay = hour(dates);
dayOfWeek = weekday(dates);
monthOfYear = month(dates);

% lagged load inputs
prevDaySameHourLoad = [NaN(24,1); target(1:end-24)];
prevWeekSameHourLoad = [NaN(168,1); target(1:end-168)];
prev24HrAveLoad = filter(ones(1,24)/24, 1, target);

% feature matrix
% X = [temperature dewPoint hourOfDay dayOfWeek prevDaySameHourLoad ...
%     prevWeekSameHourLoad prev24HrAveLoad];
X = [temperature dewPoint hourOfDay dayOfWeek monthOfYear prevDaySameHourLoad ...
    prevWeekSameHourLoad prev24HrAveLoad];

%% Split the dataset to create a Training and Test set

% Create training set
trainInd = datenum(dates) < datenum('2008-01-01');
trainX = X(trainInd,:);
trainY = target(trainInd);
trainDates = dates(trainInd);

% Create test set and save for later
testInd = datenum(dates) >= datenum('2008-01-01');
testX = X(testInd,:);
testY = target(testInd);
testDates = dates(testInd);

%% Initialize and train ANN 

% Create a Fitting Network
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
hiddenLayerSize = 30;
net = fitnet(hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error
% net.performFcn = 'mae';

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

% Train the Network
[net,tr] = train(net, trainX', trainY');

%% Test the model

y = net(trainX')';
e = gsubtract(trainY,y);
performance = perform(net,trainY,y)

% Recalculate Training, Validation and Test Performance
trainTargets = trainY.* tr.trainMask{1}';
valTargets = trainY.* tr.valMask{1}';
testTargets = trainY.* tr.testMask{1}';
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

maePerformance = mean(abs(e(169:end)))

figure;
fitPlot(trainDates, [trainY y], e);

% View the Network
view(net);

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotfit(net,x,t)

forecastLoad = net(testX')';
err = testY-forecastLoad;
figure;
fitPlot(testDates, [testY forecastLoad], err);

errpct = abs(err)./testY*100;

MAE = mean(abs(err));
MAPE = mean(errpct(~isinf(errpct)));
MSE = mse(net, testY, forecastLoad);

fprintf('Trained ANN, test set: \nMean Absolute Percent Error (MAPE): %0.3f%% \nMean Absolute Error (MAE): %0.4f Wh\nMean Squared Error (MSE): %0.4f Wh\n',...
    MAPE, MAE, MSE)
fprintf('\n');
