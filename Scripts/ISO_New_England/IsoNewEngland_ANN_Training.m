%% Feedforward neural network forecasting example: 
% ISO New England dataset from years 2004 to 2009


%% Initialization
clear; close all; clc;


%% Import ISO New England load data, weather data and holidays

load ('BFH_Projekt2_17\Scripts\ISO_New_England\Data\DBLoadData.mat');
xlsPath = 'BFH_Projekt2_17\Scripts\ISO_New_England\Data\Holidays.xls';
[~, xlsTable] = xlsread(xlsPath); 


%% Create feature matrix

dates = datetime(data.NumDate, 'ConvertFrom', 'datenum');
targets = data.SYSLoad;

% date predictors
hourOfDay = hour(dates);
dayOfWeek = weekday(dates);
monthOfYear = month(dates);

% holidays
holidays = datenum(xlsTable(2:end, 1), 'dd.mm.yyyy');
isWorkingDay = ~ismember(floor(data.NumDate), holidays) & ...
    ~ismember(dayOfWeek,[1 7]);

% lagged load inputs
prevDaySameHourLoad = [NaN(24, 1); targets(1:end - 24)];
prevWeekSameHourLoad = [NaN(168, 1); targets(1:end - 168)];
prev24HrAveLoad = filter(ones(1, 24)/24, 1, targets);

% feature matrix
X = [data.DryBulb data.DewPnt hourOfDay dayOfWeek monthOfYear ...
    isWorkingDay prevDaySameHourLoad prevWeekSameHourLoad prev24HrAveLoad];


%% Plot ISO New England dataset from years 2004 to 2009

figure;
plot(dates, targets, 'b');
title('ISO New England Daten von 2004 bis 2009 ');
xlabel('Zeit');
ylabel('MWh');
axis tight;


%% Split the dataset to create a training, validation and test set

% Create training set
trainInd = datenum(dates) < datenum('2008-01-01');
trainX = X(trainInd,:);
trainY = targets(trainInd);
trainDates = dates(trainInd);

% Create validation set
valInd = datenum(dates) >= datenum('2008-01-01') & ...
    datenum(dates) < datenum('2009-01-01');
valX = X(valInd,:);
valY = targets(valInd);
valDates = dates(valInd);

% Create test set
testInd = datenum(dates) >= datenum('2009-01-01');
testX = X(testInd,:);
testY = targets(testInd);
testDates = dates(testInd);

% Save training, validation and test sets
save BFH_Projekt2_17\Scripts\ISO_New_England\Data\IsoNewEngland_TrainSet.mat ...
    trainDates trainX trainY;
save BFH_Projekt2_17\Scripts\ISO_New_England\Data\IsoNewEngland_ValSet.mat ...
    valDates valX valY;
save BFH_Projekt2_17\Scripts\ISO_New_England\Data\IsoNewEngland_TestSet.mat ...
    testDates testX testY;
clear xlsPath xlsTable  


%% Initialize and train feedforward neural network 

% Create a Fitting Network
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
hiddenLayerSize = 25;
net = fitnet(hiddenLayerSize,trainFcn);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% net.divideFcn = 'dividerand';  % Divide data randomly
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
net.divideMode = 'sample';  % Divide up every sample
net.divideFcn = 'divideind';  % Divide the data by index
net.divideParam.trainInd = find(trainInd)';
net.divideParam.valInd = find(valInd');
net.divideParam.testInd = find(testInd');

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';   % Mean Squared Error
% net.performFcn = 'mae'; % Mean absolute error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit'};

fprintf('\n');
fprintf('Train feedforward neural network ...');
fprintf('\n');

% For training results dialog: type nntraintool
[net,tr] = train(net, X', targets');

fprintf('... training completed.\n');
fprintf('\n');

save BFH_Projekt2_17\Scripts\ISO_New_England\Models\IsoNewEngland_FitNet_Model.mat ... 
    net

%% Evaluate the model

% training set 

trainForecast = net(trainX')';
trainError = trainY - trainForecast;
trainErrPct = abs(trainError./trainY)*100;

trainPerformance = perform(net, trainY, trainForecast);
trainMSE = mse(net, trainY, trainForecast);
trainRMSE = sqrt(trainMSE);
trainMAE = mae(trainError);
trainMAPE = mean(trainErrPct, 'omitnan');

figure;
fitPlot(trainDates, [trainY trainForecast], trainError);

fprintf('\n');
fprintf('Training set\n');
fprintf('Network performance: %0.4f \n', trainPerformance);
fprintf('Training MSE: %0.4f \n', trainMSE);
fprintf('Training RMSE: %0.4f \n', trainRMSE);
fprintf('Training MAE: %0.4f \n', trainMAE);
fprintf('Training MAPE: %0.4f%% \n', trainMAPE);

% validation set

valForecast = net(valX')';
valError = valY - valForecast;
valErrPct = abs(valError./valY)*100;

valPerformance = perform(net, valY, valForecast);
valMSE = mse(net, valY, valForecast);
valRMSE = sqrt(valMSE);
valMAE = mae(valError);
valMAPE = mean(valErrPct, 'omitnan');

figure;
fitPlot(valDates, [valY valForecast], valError);

fprintf('\n');
fprintf('Validation set\n');
fprintf('Network performance: %0.4f \n', valPerformance);
fprintf('Validation MSE: %0.4f \n', valMSE);
fprintf('Validation RMSE: %0.4f \n', valRMSE);
fprintf('Validation MAE: %0.4f \n', valMAE);
fprintf('Validation MAPE: %0.4f%% \n', valMAPE);

% test set

testForecast = net(testX')';
testError = testY - testForecast;
testErrPct = abs(testError./testY)*100;

testPerformance = perform(net, testY, testForecast);
testMSE = mse(net, testY, testForecast);
testRMSE = sqrt(testMSE);
testMAE = mae(testError);
testMAPE = mean(testErrPct, 'omitnan');

figure;
fitPlot(testDates, [testY testForecast], testError);

fprintf('\n');
fprintf('Test set\n');
fprintf('Network performance: %0.4f \n', testPerformance);
fprintf('Test MSE: %0.4f \n', testMSE);
fprintf('Test RMSE: %0.4f \n', testRMSE);
fprintf('Test MAE: %0.4f \n', testMAE);
fprintf('Test MAPE: %0.4f%% \n', testMAPE);
fprintf('\n');
