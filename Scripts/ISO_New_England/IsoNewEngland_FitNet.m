%% Feedforward neural network forecasting example: 
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

% lagged load inputs
prevDaySameHourLoad = [NaN(24, 1); targets(1:end - 24)];
prevWeekSameHourLoad = [NaN(168, 1); targets(1:end - 168)];
prev24HrAveLoad = filter(ones(1, 24)/24, 1, targets);

% feature matrix
X = [dryBulb dewPoint hourOfDay dayOfWeek monthOfYear isWorkingDay ...
    prevDaySameHourLoad prevWeekSameHourLoad prev24HrAveLoad];

% targets in GWh
% showFeaturePlots(dates, targets./10^3, dryBulb, dewPoint, ' (F)');

save BFH_Projekt2_17\Scripts\ISO_New_England\Data\IsoNewEngland_Data.mat ...
    X targets dates;


%% Plot ISO New England dataset from years 2004 to 2009

% figure;
% plot(dates, targets, 'b');
% title('ISO New England Daten von 2004 bis 2009');
% xlabel('Zeit');
% ylabel('MW');
% axis tight;


%% Create training, validation and test set indexes

trainValInd = find(datenum(dates) < datenum('2008-01-01'));
[trainInd, valInd, ~] = dividerand(numel(trainValInd), 0.85, 0.15, 0);
testInd = find(datenum(dates) >= datenum('2008-01-01'));

clear xlsTable data holidays trainValInd; 


%% Initialize and train feedforward neural network 

trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
% trainFcn = 'trainbfg';     % BFGS quasi-Newton backpropagation.
% trainFcn = 'trainbr';    % Bayesian Regulation backpropagation.
% trainFcn = 'trainscg';   % Scaled conjugate gradient backpropagation.
hiddenLayerSize = 25;

% Create a Fitting Network
net = fitnet(hiddenLayerSize, trainFcn);
% net.trainParam.max_fail = 20;

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
net.divideMode = 'sample';  % Divide up every sample
% net.divideFcn = 'dividerand';  % Divide data randomly
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
net.divideFcn = 'divideind';  % Divide the data by index
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';   % Mean Squared Error
% net.performParam.regularization = 0.7;
% net.performFcn = 'mae'; % Mean absolute error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotregression', 'plotfit', 'plotwb'};

fprintf('\nTrain feedforward neural network (ISO New England data) ...\n');

% For training results dialog: type nntraintool
[net,tr] = train(net, X', targets', 'useParallel','yes', ...
    'useGPU','no');

fprintf('... training completed.\n\n');

save BFH_Projekt2_17\Scripts\ISO_New_England\Models\IsoNewEngland_FitNet_Model.mat ... 
    net tr;


%% Evaluate the model

% Create training, validation and test set from tr indexes
trainX = X(tr.trainInd, :);
trainY = targets(tr.trainInd);
valX = X(tr.valInd, :);
valY = targets(tr.valInd);
testX = X(tr.testInd, :);
testY = targets(tr.testInd);

fprintf('\nPerformance Metrics FitNet (ISO New England)');
fprintf('\n--------------------------------------------\n');

% entire dataset
showPerformanceMetrics(net, X, targets, 'Entire dataset');

% training set 
showPerformanceMetrics(net, trainX, trainY, 'Training set');

% validation set 
showPerformanceMetrics(net, valX, valY, 'Validation set');

% test set 
showPerformanceMetrics(net, testX, testY, 'Test set', dates(tr.testInd)');
% showPerformanceMetrics(net, testX, testY, 'Test set');


%% Group Analysis of Errors

% [~, ~, ~, hr] = datevec(dates(tr.testInd));
% 
% % By Hour
% figure;
% boxplot(errPct, hr+1);
% xlabel('Hour'); ylabel('Percent Error Statistics');
% title('Breakdown of forecast error statistics by hour');
% 
% % By Weekday
% figure;
% boxplot(errPct, weekday(dates(tr.testInd)), 'labels', ...
%     {'Sun','Mon','Tue','Wed','Thu','Fri','Sat'});
% ylabel('Percent Error Statistics');
% title('Breakdown of forecast error statistics by weekday');
% 
% % By Month
% figure;
% boxplot(errPct, datestr(dates(tr.testInd),'mmm'));
% ylabel('Percent Error Statistics');
% title('Breakdown of forecast error statistics by month');
% 
% clear hr;