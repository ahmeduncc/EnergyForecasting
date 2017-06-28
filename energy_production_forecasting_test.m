% Energy production forecasting test

data_range = (23 + 85*96):(23 + 85*96 + 10*96 - 1);
plot_range = 1:length(data_range);
% plot_range = 1:96;

producerData = load('Data/Producer_StJakobstadium.txt');
x = datetime(producerData(data_range,1)./1000, 'ConvertFrom', 'posixtime');
% x = datetime(1480963500, 'ConvertFrom', 'posixtime')
% datestr(x);
y = producerData(data_range, 3)/10^3;
% length(x)
% length(y)

version = 'matlab_v_1.1';
user   = 'bfh';
pwd    = 'Ulupulumo621';
server = 'api.meteomatics.com';
% Koordinaten St. Jakobpark Basel: 47.541770, 7.620086
% lat   = 46.934104;
% lon   = 7.440885;
lat   = 47.54177;
lon   = 7.620086;
start_datum = '2017-03-01T00:00:00+00:00';
period      = 'P9DT23H45M';
resolution  = 'PT15M'; 
parameters = 'solar_power_installed_capacity_2.4:MW'; 
% parameters = 'solar_power_installed_capacity_2.5_tracking_type_tilted-north-south-tracking_orientation_180_tilt_25:MW'; 
% parameters = 'solar_power_installed_capacity_2.5_tracking_type_fixed_orientation_180_tilt_25:MW'; 

[dn, data] = time_series_query(user, pwd, server, start_datum, period, resolution, ...
    parameters, lat, lon);
z = data(:,1)*10^3/4;
data_length = length(z)

% moving avg
hoursPerDay = 4;
coeff24hMA = ones(1, hoursPerDay)/hoursPerDay;
y = filter(coeff24hMA, 1, y);

figure(1);
hold on;
stem(x(plot_range), y(plot_range), 'b', 'marker', 'none');
plot(x(plot_range), z(plot_range), 'r')
title('Energieproduktion: St. Jakob-Park');
xlabel('Tage');
ylabel('kWh');
axis tight;
hold off;

figure(2);
plot(y, z, 'rx');

% mse_ = 1/data_length * sum((y - z).^2)
mse_ = mean((y - z).^2)
rmse_ = sqrt(mse_)
mae_ = mae(y - z)
mape_ = nanmean(abs((y - z)./y*100))
corr_coef = corrcoef(y, z)
sum_sensoren = sum(y)
sum_meteomatics = sum(z)

T = tonndata(z,false,false);
trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
% Create a Nonlinear Autoregressive Network
feedbackDelays = 1:8;
hiddenLayerSize = 10;
net = narnet(feedbackDelays,hiddenLayerSize,'open',trainFcn);
net.input.processFcns = {'mapminmax'};
[x,xi,ai,t] = preparets(net,{},{},T);
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'time';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.performFcn = 'mse';  % Mean Squared Error
net.plotFcns = {'plotperform','plottrainstate', 'ploterrhist', ...
    'plotregression', 'plotresponse', 'ploterrcorr', 'plotinerrcorr'};

% Train the Network
[net,tr] = train(net,x,t,xi,ai);

% Test the Network
y = net(x,xi,ai);
e = gsubtract(t,y);
performance = perform(net,t,y)

% Recalculate Training, Validation and Test Performance
trainTargets = gmultiply(t,tr.trainMask);
valTargets = gmultiply(t,tr.valMask);
testTargets = gmultiply(t,tr.testMask);
trainPerformance = perform(net,trainTargets,y)
valPerformance = perform(net,valTargets,y)
testPerformance = perform(net,testTargets,y)

% Closed Loop Network
% Use this network to do multi-step prediction.
% The function CLOSELOOP replaces the feedback input with a direct
% connection from the outout layer.
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
% view(netc)
[xc,xic,aic,tc] = preparets(netc,{},{},T);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net,tc,yc)
% figure, plotresponse(t,yc)

% Multi-step Prediction
% Sometimes it is useful to simulate a network in open-loop form for as
% long as there is known data T, and then switch to closed-loop to perform
% multistep prediction. Here The open-loop network is simulated on the
% known output series, then the network and its final delay states are
% converted to closed-loop form to produce predictions.
[x1,xio,aio,t] = preparets(net,{},{},T);
[y1,xfo,afo] = net(x1,xio,aio);
[netc,xic,aic] = closeloop(net,xfo,afo);
plot_index = 96;
[y2,xfc,afc] = netc(cell(0,plot_index),xic,aic);
% Further predictions can be made by continuing simulation starting with
% the final input and layer delay states, xfc and afc.
figure(3);
plot(1:(length(z) + plot_index), [z' cell2mat(y2)], 'r');
