%% Feedforward neural network forecasting example: 
% Swissgrid data from years 2009 to 2017


%% Initialization
clear; close all; clc;


%% Import Meteomatics data and Swissgrid data

% version = 'matlab_v_1.1';
% user   = 'bfh';
% pwd    = 'Ulupulumo621';
% server = 'api.meteomatics.com';
% lat   = 47.36667;
% lon   = 8.55;
% start_datum = '2009-01-01T01:00+00:00';
% % start_datum = '2017-01-01T01:00+00:00';
% % period      = 'P1461DT';
% % period      = 'P2922DT';
% period      = 'P3134DT';
% % period      = 'P212DT';
% resolution  = 'PT1H'; 
% parameters = 't_2m:C,dew_point_2m:C'; 
% [dn, meteoData] = time_series_query(user, pwd, server, start_datum, period, ...
%     resolution, parameters, lat, lon);
% % remove 2 last element
% meteoData(end,:) = [];
% meteoData(end,:) = [];
% save BFH_Projekt2_17\Scripts\Swissgrid\Data\MeteoData.mat meteoData;

load 'BFH_Projekt2_17\Scripts\Swissgrid\Data\MeteoData.mat';
% Endverbrauchte Energie Regelblock Schweiz (15 Min Auflösung)
load('BFH_Projekt2_17\Scripts\Swissgrid\Data\ConsumedEnergy.mat');


%% Create feature matrix

% create date vector
startDate = datetime(2009,1,1,1,0,0);
% ignore summer daylight saving
endDate = datetime(2017,7,31,23,0,0);
dates = (startDate:hours(1):endDate)';

% compute energy consumption per hour
targets = sum(reshape(ConsumedEnergy, 4, []))';
% % smooth data with Savitzky-Golay filter
% targets = sgolayfilt(rawTargets, 22, 25);

% weather data
temperature = meteoData(:,1); 
dewPoint = meteoData(:,2); 

% date predictors
hourOfDay = hour(dates);
dayOfWeek = weekday(dates);
% dayOfMonth = day(dates);
monthOfYear = month(dates);

% lagged load inputs
prevDaySameHourLoad = [NaN(24, 1); targets(1:end - 24)];
prevWeekSameHourLoad = [NaN(168, 1); targets(1:end - 168)];
prev24HrAveLoad = filter(ones(1, 24)/24, 1, targets);

% feature matrix
X = [temperature dewPoint hourOfDay dayOfWeek monthOfYear ...
    prevDaySameHourLoad prevWeekSameHourLoad prev24HrAveLoad];
% X = [temperature hourOfDay dayOfWeek monthOfYear ...
%     prevDaySameHourLoad prevWeekSameHourLoad prev24HrAveLoad];

% targets in GWh
% showFeaturePlots(dates, targets./10^6, temperature, dewPoint, ' (C)');

save BFH_Projekt2_17\Scripts\Swissgrid\Data\Swissgrid_Data.mat ...
    X targets dates;


%% Plot Swissgrid data from years 2009 to 2016

% figure;
% plot(dates, targets, 'b');
% title('Swissgrid: Endverbrauch Endergie von 2009 bis 2016');
% xlabel('Zeit');
% ylabel('kWh');
% axis tight;

% % plot first weeek
% shortIndex = 1:168;
% figure;
% plot(dates(shortIndex), rawTargets(shortIndex), 'b', dates(shortIndex), ...
%     targets(shortIndex), 'r');
% title('Swissgrid: Endverbrauch Endergie von 2009 bis 2016');
% xlabel('Zeit');
% ylabel('kWh');
% legend('raw data', 'smoothed data');
% axis tight;


%% Create training, validation and test set indexes

trainValInd = find(datenum(dates) <= datenum('2016-01-01'));
[trainInd, valInd, ~] = dividerand(numel(trainValInd), 0.85, 0.15, 0);
testInd = find(datenum(dates) > datenum('2016-01-01'));

clear startDate endDate TotalLoad meteoData trainValInd;


%% Initialize and train feedforward neural network 

trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
% trainFcn = 'trainbfg';     % BFGS quasi-Newton backpropagation.
% trainFcn = 'trainbr';    % Bayesian Regulation backpropagation.
% trainFcn = 'trainscg';   % Scaled conjugate gradient backpropagation.
hiddenLayerSize = 25;

% Create a Fitting Network
net = fitnet(hiddenLayerSize, trainFcn);
% net.trainParam.max_fail = 30;

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
net.divideMode = 'sample';  % Divide up every sample
% net.divideFcn = 'dividerand';  % Divide data randomly
% net.divideParam.trainRatio = 85/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 0/100;
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

fprintf('\nTrain feedforward neural network (Swissgrid data) ...\n');

% For training results dialog: type nntraintool
[net, tr] = train(net, X', targets', 'useParallel','yes', ...
    'useGPU','no');

fprintf('... training completed.\n\n');

save BFH_Projekt2_17\Scripts\Swissgrid\Models\Swissgrid_FitNet_Model.mat ... 
    net tr;


%% Evaluate the model

% Create training, validation and test set from tr indexes
trainX = X(tr.trainInd, :);
trainY = targets(tr.trainInd);
valX = X(tr.valInd, :);
valY = targets(tr.valInd);
testX = X(tr.testInd, :);
testY = targets(tr.testInd);

fprintf('\nPerformance Metrics FitNet (Swissgrid)');
fprintf('\n--------------------------------------\n');

% entire dataset
showPerformanceMetrics(net, X, targets, 'Entire dataset');

% training set 
showPerformanceMetrics(net, trainX, trainY, 'Training set');

% validation set 
showPerformanceMetrics(net, valX, valY, 'Validation set');

% test set 
[errPct, err] = showPerformanceMetrics(net, testX, testY, 'Test set', ...
    dates(tr.testInd)');
% showPerformanceMetrics(net, testX, testY, 'Test set');

