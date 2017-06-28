%% NARX forecasting example: ISO New England dataset for the years 2004 to 2008

%% Load New England ISO dataset for the years 2004 to 2008

load 'G:\Daten\Programming\Matlab\Examples\Electricity Load & Price Forecasting\Load\Data\DBLoadData.mat'
addpath 'G:\Daten\Programming\Matlab\Examples\Electricity Load & Price Forecasting\Util'

dates = datetime(data.NumDate,'ConvertFrom','datenum');
target = data.SYSLoad;

%% create feature matrix

% weather data
temperature = data.DryBulb; 
dewPoint = data.DewPnt; 

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
X = [temperature dewPoint hourOfDay dayOfWeek dayOfMonth monthOfYear];

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

fprintf('Trained ANN, test set: \nMean Absolute Percent Error (MAPE): %0.3f%% \nMean Absolute Error (MAE): %0.4f Wh\nMean Squared Error (MSE): %0.4f Wh\n',...
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

numPredictions = 12;
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
