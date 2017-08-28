%% Meteomatics forecasting example: Change38 producer St. Jakobstadium

%% Initialization
clear; close all; clc;

%% load metomatics forecast data

% version = 'matlab_v_1.1';
% user   = 'bfh';
% pwd    = 'Ulupulumo621';
% server = 'api.meteomatics.com';
% lat   = 47.54177;
% lon   = 7.620086;
% start_datum = '2016-12-05T18:45+00:00';
% period      = 'P185DT';
% resolution  = 'PT15M'; 
% parameters = 'solar_power_installed_capacity_2.2:MW'; 
% [dn, data] = time_series_query(user, pwd, server, start_datum, period, ...
%     resolution, parameters, lat, lon);
% z = data(:,1)*10^3/4; % transform from MW zu kWh
load('Data\StJakobstadium_Metomatics.mat');

%% load copco data

producerData = load('..\..\Data\Change38_Producer\Producer_StJakobstadium_20170608.txt');
copco_data_length = length(producerData(:,1));
data_range = 1:copco_data_length;
empty_y1 = zeros(length(z) - copco_data_length,1);
last_known_date = producerData(copco_data_length,1)/10^3;
future_dates = (last_known_date + 900):900:(last_known_date + 900 + ...
    900*(length(empty_y1) - 1));
x = [producerData(data_range,1)./10^3; future_dates'];
x = datetime(x, 'ConvertFrom', 'posixtime');
y = [producerData(data_range, 3)/10^3; empty_y1]; % kWh

% moving avg
% hoursPerDay = 4;
% coeff24hMA = ones(1, hoursPerDay)/hoursPerDay;
% y = filter(coeff24hMA, 1, y);


%% plot time series: full period

plot_range = 1:length(y);
figure(1);
hold on;
stem(x(plot_range), y(plot_range), 'b', 'marker', 'none');
plot(x(plot_range), z(plot_range), 'r')
legend('COPCO Sensordaten', 'Meteomatics Prognosen')
title('Energieproduktion: St. Jakob-Park');
xlabel('Tage');
ylabel('kWh');
axis tight;
hold off;

%% calculate metrics: full peroid

sum_copco_kWh = sum(y(data_range))/10^3
sum_meteomatics_kWh = sum(z(data_range))/10^3
corr_coef = corrcoef(y(data_range), z(data_range))
mse_ = mean((y(data_range) - z(data_range)).^2)
rmse_ = sqrt(mse_)
mae_ = mae(y(data_range) - z(data_range))

%% Data range: Feb Mar

feb_mar_range = 5494:11157;
x_feb_mar = x(feb_mar_range);
y_feb_mar = y(feb_mar_range);
z_feb_mar = z(feb_mar_range);

% moving avg
% hoursPerDay = 4;
% coeff24hMA = ones(1, hoursPerDay)/hoursPerDay;
% y_feb_mar = filter(coeff24hMA, 1, y_feb_mar);

%% plot time series: Feb Mar

figure(2);
hold on;
stem(x_feb_mar, y_feb_mar, 'b', 'marker', 'none');
plot(x_feb_mar, z_feb_mar, 'r')
title('Energieproduktion: St. Jakob-Park Februar und März 2017');
xlabel('Tage');
ylabel('kWh');
legend('COPCO Sensordaten', 'Meteomatics Prognosen')
axis tight;
hold off;

%% calculate metrics: Feb, March 2017

sum_copco_Feb_Mar_kWh = sum(y_feb_mar)/10^3
sum_meteomatics_Feb_Mar_kWh = sum(z_feb_mar)/10^3
corr_coef_Feb_Mar = corrcoef(y_feb_mar, z_feb_mar)
mse_Feb_Mar = mean((y_feb_mar - z_feb_mar).^2)
rmse_Feb_Mar = sqrt(mse_)
mae_Feb_Mar = mae(y_feb_mar - z_feb_mar)


%% Data range: April, May, June 2017

apr_may_index = 11158;
x_apr_may = x(apr_may_index:end);
y_apr_may = y(apr_may_index:end);
z_apr_may = z(apr_may_index:end);

%% plot time series: April, May, June 2017

figure(3);
hold on;
stem(x_apr_may, y_apr_may, 'b', 'marker', 'none');
plot(x_apr_may, z_apr_may, 'r')
title('Energieproduktion: St. Jakob-Park April, Mai, Juni 2017');
xlabel('Tage');
ylabel('kWh');
legend('COPCO Sensordaten', 'Meteomatics Prognosen')
axis tight;
hold off;

%% calculate metrics: April, May, June 2017

sum_copco_Feb_Mar_kWh = sum(y_apr_may)/10^3
sum_meteomatics_Feb_Mar_kWh = sum(z_apr_may)/10^3
corr_coef_Feb_Mar = corrcoef(y_apr_may, z_apr_may)
mse_Feb_Mar = mean((y_apr_may - z_apr_may).^2)
rmse_Feb_Mar = sqrt(mse_)
mae_Feb_Mar = mae(y_apr_may - z_apr_may)
