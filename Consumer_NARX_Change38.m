%% NARX forecasting example: Change38 consumer

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

rows = Production_HOUR_CONSUMPTION_20170624.USER_ID == userId;
vars = {'START_MILIS', 'CONSUMED_ENERGY'};
consumerData = Production_HOUR_CONSUMPTION_20170624{rows, vars};

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

%% create feature matrix

% weather data
temperature = meteoData(dateIndex,1); 
dewPoint = meteoData(dateIndex,2); 

% date predictors
hourOfDay = hour(dates);
dayOfWeek = weekday(dates);
dayOfMonth = day(dates);
monthOfYear = month(dates);

% lagged load inputs
prevDaySameHourLoad = [NaN(24,1); target(1:end-24)];
prevWeekSameHourLoad = [NaN(168,1); target(1:end-168)];
prev24HrAveLoad = filter(ones(1,24)/24, 1, target);

% feature matrix
X = [temperature dewPoint hourOfDay dayOfWeek];

%% Split the dataset to create a Training and Test set

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

%% Initialize and Train NARX Network

trainXcell = tonndata(trainX,false,false);
trainYcell = tonndata(trainY,false,false);

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.

% Create a Nonlinear Autoregressive Network with External Input
maxDelay = 14;
inputDelays = 1:maxDelay;
feedbackDelays = 1:maxDelay;
hiddenLayerSize = 20;
net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);

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
[Xs,Xi,Ai,Ts] = preparets(net,trainXcell,{},trainYcell);

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
[net,tr] = train(net,Xs,Ts,Xi,Ai);

%% Test Network on trainX dataset

y = net(Xs,Xi,Ai);
performance = perform(net,Ts,y)

% Recalculate Training, Validation and Test Performance
trainTargets = gmultiply(Ts,tr.trainMask);
valTargets = gmultiply(Ts,tr.valMask);
testTargets = gmultiply(Ts,tr.testMask);
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

e = cell2mat(gsubtract(Ts,y))';
errpct = abs(e)./cell2mat(Ts)'*100;
maePerformance = mean(abs(e))
mapePerformance = mean(errpct(~isinf(errpct)))

% View the Network
% view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotregression(t,y)
%figure, plotresponse(t,y)
%figure, ploterrcorr(e)
%figure, plotinerrcorr(x,e)

figure;
fitPlot(trainDates((maxDelay+1):end), [cell2mat(Ts)' cell2mat(y)'], e);

%% Test Network on testX dataset

testXcell = tonndata(testX,false,false);
testYcell = tonndata(testY,false,false);

[Xst,Xit,Ait,Tst] = preparets(net,testXcell,{},testYcell);

forecastLoad = net(Xst,Xit,Ait);
err = cell2mat(Tst)'-cell2mat(forecastLoad)';

figure;
fitPlot(testDates((maxDelay+1):end), [cell2mat(Tst)' cell2mat(forecastLoad)'], err);

errpct = abs(err)./cell2mat(Tst)'*100;

MAE = mean(abs(err));
MAPE = mean(errpct(~isinf(errpct)));
MSE = mse(net, Tst, forecastLoad);

fprintf('Trained ANN, test set: Mean Absolute Percent Error (MAPE): %0.3f%% \nMean Absolute Error (MAE): %0.4f Wh\nMean Squared Error (MSE): %0.4f Wh\n',...
    MAPE, MAE, MSE)
fprintf('\n');

%% Closed Loop Network

% Closed Loop Network
% Use this network to do multi-step prediction.
% The function CLOSELOOP replaces the feedback input with a direct
% connection from the outout layer.
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
% view(netc)
[Xs_cl,Xi_cl,Ai_cl,Ts_cl] = preparets(netc,testXcell,{},testYcell);
Yc = netc(Xs_cl,Xi_cl,Ai_cl);

numPredictions = 24;
for i = 1:numPredictions
    closedLoopRange = 1:i;
    errcl = cell2mat(Ts_cl(closedLoopRange))' - cell2mat(Yc(closedLoopRange))';
    errpct = abs(errcl)./cell2mat(Ts_cl(closedLoopRange))'*100;
    closedLoopMAPE = mean(errpct(~isinf(errpct)));
    fprintf('Closed loop prediction: hour: %i, MAPE: %0.3f%%', i, closedLoopMAPE);
    fprintf('\n');
end

closedLoopMSE = perform(net,Ts_cl(1:numPredictions),Yc(1:numPredictions))
closedLoopMAE = mean(abs(cell2mat(Ts_cl(1:numPredictions))'-cell2mat(Yc(1:numPredictions))'))

figure;
hold on;
timeDisplayRange = (maxDelay + 1):(maxDelay + numPredictions*2);
valueDisplayRange = 1:numPredictions*2;
plot(testDates(timeDisplayRange), cell2mat(Ts_cl(valueDisplayRange))', 'b');
plot(testDates(timeDisplayRange), cell2mat(Yc(valueDisplayRange))', 'r');
title('Closed Loop: erwarteter und vorhergesagter Energieverbrauch');
xlabel('Stunden');
ylabel('Energieverbrauch [MWh]');
legend('Erwartete Werte', 'Vorhergesagte Werte')
axis tight;
hold off;
