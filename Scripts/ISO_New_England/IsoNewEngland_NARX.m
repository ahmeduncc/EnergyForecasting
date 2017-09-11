%% NARX forecasting example: 
% ISO New England dataset from years 2004 to 2009


%% Initialization
clear; close all; clc;


%% Import ISO New England load data, weather data and holidays

load ('BFH_Projekt2_17\Scripts\ISO_New_England\Data\DBLoadData.mat');
[~, xlsTable] = xlsread('BFH_Projekt2_17\Data\ISO_New_England\Holidays.xls'); 


%% Create feature matrix

dates = datetime(data.NumDate, 'ConvertFrom', 'datenum');
targets = data.SYSLoad;

% weather data
dryBulb = data.DryBulb; 
dewPoint = data.DewPnt; 

% date predictors
hourOfDay = hour(dates);
dayOfWeek = weekday(dates);
% dayOfMonth = day(dates);
monthOfYear = month(dates);

% holidays
holidays = datenum(xlsTable(2:end, 1), 'dd.mm.yyyy');
isWorkingDay = ~ismember(floor(data.NumDate), holidays) & ...
    ~ismember(dayOfWeek,[1 7]);

% feature matrix
X = [dryBulb dewPoint hourOfDay dayOfWeek monthOfYear isWorkingDay];

% showIsoNewEnglandFeaturePlots(dates, targets, dryBulb, dewPoint);


%% Split the dataset to create training, validation and test data

%create training, validation and test indexes
trainInd = datenum(dates) < datenum('2008-01-01');
valInd = datenum(dates) >= datenum('2008-01-01') & ...
    datenum(dates) < datenum('2009-01-01');
testInd = datenum(dates) >= datenum('2009-01-01');

% Create training set
trainX = X(trainInd,:);
trainY = targets(trainInd);
trainDates = dates(trainInd);

% Create validation set
valX = X(valInd,:);
valY = targets(valInd);
valDates = dates(valInd);

% Create test set
testX = X(testInd,:);
testY = targets(testInd);
testDates = dates(testInd);

% Save training and test data
save BFH_Projekt2_17\Scripts\ISO_New_England\Data\IsoNewEngland_NARX_TrainSet.mat ...
    trainX trainY trainDates;
save BFH_Projekt2_17\Scripts\ISO_New_England\Data\IsoNewEngland_NARX_ValSet.mat ...
    valX valY valDates;
save BFH_Projekt2_17\Scripts\ISO_New_England\Data\IsoNewEngland_NARX_TestSet.mat ...
    testX testY testDates;
clear xlsTable data holidays;


%% Initialize and train NARX network

Xcell = tonndata(X, false, false);
Ycell = tonndata(targets, false, false);

% Choose a Training Function
% For a list of all training functions type: help nntrain
% trainFcn = 'trainbr';     % takes longer but may be better for challenging problems.
% trainFcn = 'trainscg'; % uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Nonlinear Autoregressive Network with External Input
maxInputDelay = 24;
maxFeedbackDelay = 24;
inputDelays = 1:maxInputDelay;
feedbackDelays = 1:maxFeedbackDelay;
hiddenLayerSize = 10;
net = narxnet(inputDelays, feedbackDelays, hiddenLayerSize, 'open', ...
    trainFcn);

% Choose Input and Feedback Pre/Post-Processing Functions
% Settings for feedback input are automatically applied to feedback output
% For a list of all processing functions type: help nnprocess
% Customize input parameters at: net.inputs{i}.processParam
% Customize output parameters at: net.outputs{i}.processParam
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

% Prepare the Data for Training and Simulation
% The function PREPARETS prepares timeseries data for a particular network,
% shifting time by the minimum amount to fill input states and layer
% states. Using PREPARETS allows you to keep your original time series data
% unchanged, while easily customizing it for networks with differing
% numbers of delays, with open loop or closed loop feedback modes.
[delayedInputs, inputStates, layerStates, delayedTargets] = preparets(net, ...
    Xcell, {}, Ycell);

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideMode = 'time';  
% net.divideMode = 'sampletime';  
% net.divideFcn = 'dividerand';  % Divide data randomly
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
net.divideFcn = 'divideind';  % Divide the data by index
net.divideParam.trainInd = find(trainInd);
net.divideParam.valInd = find(valInd);
net.divideParam.testInd = find(testInd);

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
    'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr' ...
    , 'plotwb'};

fprintf('\nTrain NARX network (ISO New England data) ...\n');

% Train the Network
[net, tr] = train(net, delayedInputs, delayedTargets, inputStates, ...
    layerStates, 'useParallel','yes', 'useGPU','no');

fprintf('... training completed.\n\n');

save BFH_Projekt2_17\Scripts\ISO_New_England\Models\IsoNewEngland_NARX_Model.mat ... 
    net tr maxInputDelay maxFeedbackDelay delayedInputs delayedTargets ...
    inputStates layerStates
clear Xcell Ycell;


%% Evaluate the model in open loop

delayedForecastMat = cell2mat(net(delayedInputs, inputStates, layerStates))';
delayedTargetsMat = cell2mat(delayedTargets)';

fprintf('\nPerformance Metrics Open Loop (ISO New England)\n');
fprintf('-----------------------------------------------\n');

% energy consumption in GWh
gwhFactor = 10^(-3);

% entire dataset
showNARXPerformanceMetrics(net, delayedForecastMat, delayedTargetsMat, ...
    'Entire dataset open loop', dates((maxFeedbackDelay + 1):end) , 0, ...
    gwhFactor);

% training set
showNARXPerformanceMetrics(net, delayedForecastMat(tr.trainInd), ...
    delayedTargetsMat(tr.trainInd), ...
    'Training set open loop', dates(tr.trainInd + maxFeedbackDelay), 0, ...
    gwhFactor);

% validation set 
showNARXPerformanceMetrics(net, delayedForecastMat(tr.valInd), ...
    delayedTargetsMat(tr.valInd), ...
    'Validation set open loop', dates(tr.valInd + maxFeedbackDelay), 0, ...
    gwhFactor);

% test set 
[errPctOpen, errOpen] = showNARXPerformanceMetrics(net, ...
    delayedForecastMat(tr.testInd), ...
    delayedTargetsMat(tr.testInd), ...
    'Test set open loop', dates(tr.testInd + maxFeedbackDelay), 1, ...
    gwhFactor);


%% Evaluate the model in closed loop

% The function CLOSELOOP replaces the feedback input with a direct
% connection from the outout layer.
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
% view(netc)

forecastDuration = 24;
currentHour = 1;
forecastIndex = (currentHour):(currentHour + ...
    maxFeedbackDelay + forecastDuration - 1);
testDatesIndex = (currentHour + maxFeedbackDelay):(currentHour + ...
    maxFeedbackDelay + forecastDuration - 1);

testXcell = tonndata(testX(forecastIndex, :), false, false);
testYcell = tonndata(testY(forecastIndex), false, false);
[inputsClosed, inputStatesClosed, layerStatesClosed, delayedTargetsClosed] ...
    = preparets(netc, testXcell, {}, testYcell);
closedForecastMat = cell2mat(netc(inputsClosed, inputStatesClosed, ...
    layerStatesClosed))';
closedTargetsMat = cell2mat(delayedTargetsClosed)';

fprintf('\nPerformance Metrics Closed Loop (ISO New England)\n');
fprintf('-------------------------------------------------\n');

[errPctClosed, errClosed] = showNARXPerformanceMetrics(net, closedForecastMat, ...
    closedTargetsMat, ...
    'Test set closed loop', testDates(testDatesIndex), 1, gwhFactor);

