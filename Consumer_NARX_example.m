% Energy consumer NARX forecast example

%% load meteomatics data

% version = 'matlab_v_1.1';
% user   = 'bfh';
% pwd    = 'Ulupulumo621';
% server = 'api.meteomatics.com';
% lat   = 47.54177;
% lon   = 7.620086;
% start_datum = '2016-12-13T11:30+00:00';
% period      = 'P185DT';
% resolution  = 'PT15M'; 
% parameters = 't_2m:C,dew_point_2m:C'; 
% [dn, data] = time_series_query(user, pwd, server, start_datum, period, ...
%     resolution, parameters, lat, lon);
% temperature = data(:,1); 
% dewPoint = data(:,2); 

%% load COPCO consumer data and add missing values

consumerData = load('Data/Consumer1_Andreas_Aeberhard_20170608.txt');
% consumerData = load('Data/Consumer4_Niklaus_Freuler_20170419.txt');

i = 1;
j = 1;
currentDate = consumerData(1,1);
while currentDate < consumerData(end, 1)
    inputData(j, 1) = consumerData(i, 1);
    inputData(j, 2) = consumerData(i, 2);

    while currentDate < consumerData(end, 1) && ...
            consumerData(i + 1, 1) ~= currentDate + 900000          
        j = j + 1;
        currentDate = currentDate + 900000;
        inputData(j, 1) = currentDate;
        inputData(j, 2) = 0;
    end
    
    i = i + 1;
    j = j + 1;
    currentDate = currentDate + 900000;
end
inputData(j, 1) = consumerData(i, 1);
inputData(j, 2) = consumerData(i, 2);

% moving avg
inputWindowSize = 4;
filteredData = filter(ones(1, inputWindowSize)./inputWindowSize, 1, inputData(:, 2));

forecastPeriod = (7*96 + 1):length(inputData(:, 1));
dates = datetime(inputData(forecastPeriod, 1)./1000, 'ConvertFrom', 'posixtime');
y = filteredData(forecastPeriod);
% y = inputData(forecastPeriod, 2);
y_no_filter = inputData(forecastPeriod, 2);
dataRange = 1:length(y);

%% create feature matrix

avgWindowSize = 96;
movingAvg = filter(ones(1, avgWindowSize)./avgWindowSize, 1, filteredData);
[h, m] = hms(dates);
x(:, 1) = 60*h + m;         % Hour of day
x(:, 2) = weekday(dates);   % Day of the week
x(:, 3) = filteredData((6*96 + 1):(end - 96)); % Load from the same hour the previous day
x(:, 4) = filteredData(1:(end - 7*96)); % Load from the same hour and same day from the previous week
x(:, 5) = movingAvg(forecastPeriod); % avg. load from day before
x(:, 6) = temperature(dataRange);
x(:, 7) = dewPoint(dataRange);

x1 = x(:, 1);
x2 = x(:, 2);
x3 = x(:, 3);
x4 = x(:, 4);
x5 = x(:, 5);
x6 = x(:, 6);
x7 = x(:, 7);

%% plot weather data

figure(1);
hold on;
plot(dates, x(:, 6), 'b');
plot(dates, x(:, 7), 'r');
title('Wetterdaten Meteomatics');
xlabel('Tag');
ylabel('Temperatur und Taupunkt [C]');
axis tight;
hold off;

%% train model

%   x - input time series.
%   y - feedback time series.

X = tonndata(x,false,false);
T = tonndata(y,false,false);

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Nonlinear Autoregressive Network with External Input
nbOfInputDelays = 8;
nbOfFeedbackDelays = 8;
inputDelays = 1:nbOfInputDelays;
feedbackDelays = 1:nbOfFeedbackDelays;
hiddenLayerSize = 25;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);

% Choose Input and Feedback Pre/Post-Processing Functions
% Settings for feedback input are automatically applied to feedback output
% For a list of all processing functions type: help nnprocess
% Customize input parameters at: net.inputs{i}.processParam
% Customize output parameters at: net.outputs{i}.processParam
% net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
% net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

% Prepare the Data for Training and Simulation
% The function PREPARETS prepares timeseries data for a particular network,
% shifting time by the minimum amount to fill input states and layer
% states. Using PREPARETS allows you to keep your original time series data
% unchanged, while easily customizing it for networks with differing
% numbers of delays, with open loop or closed loop feedback modes.
[xPreparets,xi,ai,t] = preparets(net,X,{},T);

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'time';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean Squared Error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
    'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr'};

% Train the Network
[net,tr] = train(net,xPreparets,t,xi,ai);

% Test the Network
forecasts = net(xPreparets,xi,ai);
e = gsubtract(t,forecasts);
performance = perform(net,t,forecasts)

% Recalculate Training, Validation and Test Performance
trainTargets = gmultiply(t,tr.trainMask);
valTargets = gmultiply(t,tr.valMask);
testTargets = gmultiply(t,tr.testMask);
trainPerformance = perform(net,trainTargets,forecasts)
valPerformance = perform(net,valTargets,forecasts)
testPerformance = perform(net,testTargets,forecasts)

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,forecasts)
%figure, plotresponse(t,forecasts)
%figure, ploterrcorr(e)
%figure, plotinerrcorr(xPreparets,e)

%% closed Loop Network

% Use this network to do multi-step prediction.
% The function CLOSELOOP replaces the feedback input with a direct
% connection from the outout layer.
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
% view(netc)
[xc,xic,aic,tc] = preparets(netc,X,{},T);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net,tc,yc)

% Multi-step Prediction
% Sometimes it is useful to simulate a network in open-loop form for as
% long as there is known output data, and then switch to closed-loop form
% to perform multistep prediction while providing only the external input.
% Here all but 5 timesteps of the input series and target series are used
% to simulate the network in open-loop form, taking advantage of the higher
% accuracy that providing the target series produces:

% numTimesteps = size(xPreparets,2);
% knownOutputTimesteps = 1:(numTimesteps-5);
% predictOutputTimesteps = (numTimesteps-4):numTimesteps;
% X1 = X(:,knownOutputTimesteps);
% T1 = T(:,knownOutputTimesteps);

[xtemp,xio,aio] = preparets(net,X,{},T);
[y1,xfo,afo] = net(xtemp,xio,aio);
% x2 = X(1,predictOutputTimesteps);

predictSteps = 96;
[netc,xic,aic] = closeloop(net,xfo,afo);
[y2,xfc,afc] = netc(cell(0, predictSteps), xic, aic);
% multiStepPerformance = perform(net,T((end - 95):end),y2)

%% plot actual and predicted values

plotRange = (nbOfInputDelays + 1):length(y);

figure(2);
hold on;
stem(dates(plotRange), y(plotRange), 'b', 'marker', 'none');
plot(dates(plotRange), cell2mat(forecasts)', 'r');
title('Energie-Konsum Bsp. Kunde mit Moving Avg Filter');
xlabel('Tage');
ylabel('Wh');
legend('Tatsächliche Werte', 'Vorhergesagte Werte')
axis tight;
hold off;

figure(3);
hold on;
stem(dates(plotRange), inputData((7*96 + nbOfInputDelays + 1):end, 2), ...
    'b', 'marker', 'none');
plot(dates(plotRange), cell2mat(forecasts)', 'r');
title('Energie-Konsum Bsp. Kunde');
xlabel('Tage');
ylabel('Wh');
legend('Tatsächliche Werte', 'Vorhergesagte Werte')
axis tight;
hold off;

figure(4);
plot(1:(length(T) + predictSteps), [cell2mat(T) cell2mat(y2)], 'r');

mse_filter = mean((y(plotRange) - cell2mat(forecasts)').^2)
mse_no_filter = mean((y_no_filter(plotRange) - cell2mat(forecasts)').^2)
