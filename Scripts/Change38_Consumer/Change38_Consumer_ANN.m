%% Feedforward neural net forecasting example: Change38 consumers

%% Initialization
clear; close all; clc;

%% load meteomatics data

% version = 'matlab_v_1.1';
% user   = 'bfh';
% pwd    = 'Ulupulumo621';
% server = 'api.meteomatics.com';
% lat   = 47.54177;
% lon   = 7.620086;
% start_datum = '2017-01-31T23:00+00:00';
% period      = 'P143DT13H';
% resolution  = 'PT1H'; 
% parameters = 't_2m:C,dew_point_2m:C'; 
% [dn, meteoData] = time_series_query(user, pwd, server, start_datum, period, ...
%     resolution, parameters, lat, lon);
load '..\Meteomatics\Data\MeteomaticsData_20170624.mat';

%%  load COPCO consumer data

% userId = 'auth0|5819e838bb13c98632bcb858';
% userId = 'auth0|5819e952bb13c98632bcb88c';
% userId = 'auth0|5804daf9c74919636e09273d';
% userId = 'auth0|5809e58b4f8a40514ba9310b';
% userId = 'auth0|580e06d713660ba533eb283a';
% userId = 'auth0|58518ab943d6ad4e31ca35f1';
% userId = 'auth0|584e570fc641b4f41364b7d9';
% userId = 'auth0|5824fb73944239c04eb414d1';
% userId = 'auth0|588afb6eb52f602be1ce1ab6';
% userId = 'auth0|585138fb65a32d5ea65af0f7';
% userId = 'auth0|587e829a88dc3369e720d4de';
% userId = 'auth0|589af2e5841735494d320f9c';
% userId = 'auth0|5890bb4cd3fa39022d7a0d8d';
% userId = 'auth0|5819f1a8bb13c98632bcb9f8';
% userId = 'auth0|585134f1b95c657144e74bad';
% userId = 'auth0|58935004d3f66f4e6c62610c';
% userId = 'auth0|587638bff94ee950379a86b2';
% userId = 'auth0|5819f01dbb13c98632bcb9a7';
% userId = 'auth0|588afca9b52f602be1ce1ac2';
%- userId = 'auth0|5819eb12bb13c98632bcb8d2';
userId = 'auth0|5819e9ee2602198a6e5483c1';
%- userId = 'auth0|588afd37b52f602be1ce1ac9';
% userId = 'auth0|580104daf6c72a7451e060d9';
% userId = 'auth0|58a0332ed9e31b0776cd0665';
% userId = 'auth0|5890bc119f50df24b07b3773';
%- userId = 'auth0|58af24181624d6057382d0a1';
% userId = 'auth0|580e13abb1d5eae0595368c4';
% userId = 'auth0|58ac4255ed51fc08e61f0d42';
% userId = 'auth0|58ab0e9ded51fc08e61efada';
%userId = 'auth0|5887de343856aa3546ca0037';

load 'Data\HOUR_CONSUMPTION_20170624.mat';

rows = HOUR_CONSUMPTION_20170624.USER_ID == userId;
vars = {'START_MILIS', 'CONSUMED_ENERGY'};
consumerData = HOUR_CONSUMPTION_20170624{rows, vars};

%% add missing values

i = 1;
j = 1;
currentDate = consumerData(1,1);
while currentDate < consumerData(end, 1)
    inputData(j, 1) = consumerData(i, 1);
    inputData(j, 2) = consumerData(i, 2);

    while currentDate < consumerData(end, 1) && ...
            consumerData(i + 1, 1) ~= currentDate + 1          
        j = j + 1;
        currentDate = currentDate + 1;
        inputData(j, 1) = currentDate;
        inputData(j, 2) = 0;
    end
    
    i = i + 1;
    j = j + 1;
    currentDate = currentDate + 1;
end
inputData(j, 1) = consumerData(i, 1);
inputData(j, 2) = consumerData(i, 2);

%% convert dates and extract full days

fullDateList = datetime(inputData(:, 1)*60*60, 'ConvertFrom', 'posixtime');
dateIndex = datenum(fullDateList) >= datenum('2017-02-01') & ...
    datenum(fullDateList) < datenum('2017-06-24');
dates = fullDateList(dateIndex);
target = inputData(dateIndex, 2);

%% smooth curve and calculate avg per day

% target = filter(ones(1,4)/4, 1, target);
targetsPerDay = reshape(target, [24,143])';
avgPerDay = mean(targetsPerDay);
testAvg = repmat(avgPerDay,1,14)';

%% replace anomalies with default (avg) values

anomalyRange = 1177:(1177 + 9*24 - 1);
target(anomalyRange) = repmat(avgPerDay,1,9)';

%% create feature matrix

% weather data
temperature = meteoData(dateIndex,1); 
dewPoint = meteoData(dateIndex,2); 

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
% prevWeekSameHourLoad prev24HrAveLoad];
X = [temperature dewPoint hourOfDay dayOfWeek prevDaySameHourLoad ...
    prevWeekSameHourLoad prev24HrAveLoad];

%% plot current energy consumption

figure;
plot(dates, target, 'b');
% stem(dates, target, 'b', 'marker', 'none');
title(['Energieverbrauch Kunde ' userId]);
xlabel('Tage');
ylabel('Wh');
axis tight;

minDate = dates(1);
maxDate = dates(end);
display(minDate);
display(maxDate);

%% plot weather data

figure;
hold on;
plot(dates, temperature, 'b');
plot(dates, dewPoint, 'r');
title('Wetterdaten Meteomatics');
xlabel('Tag');
ylabel('Temperatur und Taupunkt [C]');
axis tight;
hold off;

%% split the dataset to create a Training and Test set

% create training set
trainInd = datenum(dates) < datenum('2017-06-01');
trainX = X(trainInd,:);
trainY = target(trainInd);
trainDates = dates(trainInd);

% create test set 
testInd = datenum(dates) >= datenum('2017-06-01') & datenum(dates) < ...
    datenum('2017-06-15');
testX = X(testInd,:);
testY = target(testInd);
testDates = dates(testInd);

%% initialize and train network

% Create a Fitting Network
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
hiddenLayerSize = 20;
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

%% test the Network

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

%% plot avg default values

predictionsPerDay = reshape(forecastLoad, [24,14])';
predictAvgPerDay = mean(predictionsPerDay);
dayRange = 1:24;

figure;
hold on;
plot(dayRange, avgPerDay, 'b');
plot(dayRange, predictAvgPerDay, 'r');
title('Durchschnittswerte im Tagesverlauf');
xlabel('Tagesstunden');
ylabel('Erwartete (b) und vorhergesagte (r) Werte [Wh]');
axis tight;
hold off;

err = testY-testAvg;

figure();
fitPlot(testDates, [testY testAvg], err);

errpct = abs(err)./testY*100;

MAE = mean(abs(err));
MAPE = mean(errpct(~isinf(errpct)));
MSE = mse(net, testY, testAvg);

fprintf('Avg values: Mean Absolute Percent Error (MAPE): %0.3f%% \nMean Absolute Error (MAE): %0.4f Wh\nMean Squared Error (MSE): %0.4f Wh\n',...
    MAPE, MAE, MSE)
fprintf('\n');

