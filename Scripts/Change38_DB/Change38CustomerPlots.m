% Plots some Change38 producers and consumers from COPCO database

%% Initialization
clear; close all; clc;

%% plot data

producerData = load('..\..\Data\Change38_Producer\Producer_StJakobstadium.txt');
consumerData = load('..\..\Data\Change38_Consumer\Consumer1_Andreas_Aeberhard.txt');

start = producerData(1,1);
x = (producerData(:,1) - start)/(24*60*60*1000);
x1 = x(10*96:end);
y1 = producerData(10*96:end,3);
y2 = producerData(11*96:25*96,3);
yc1 = consumerData(:,3);
% z = producerData(10*96:end,3)/10^3;
z = producerData(:,3)/10^3;

figure(1);
stem(x1(1*96:25*96), y1(1*96:25*96), 'r', 'marker','none');
axis tight;
title('Energieproduktion: St. Jakob-Park');
xlabel('Tage');
ylabel('Wh');

figure(2);
stem(x1, y1, 'marker','none');
axis tight;
title('Energieproduktion: St. Jakob-Park');
xlabel('Tage');
ylabel('Wh');

x = (consumerData(:,1) - consumerData(1,1))/(24*60*60*1000);
y = consumerData(:,3);

figure(3);
stem(x(1:15*96), y(1:15*96), 'r', 'marker','none');
axis tight;
title('Energiekonsum: Beispiel-Kunde 1');
xlabel('Tage');
ylabel('Wh');

figure(4);
stem(x, y, 'marker','none');
axis tight;
title('Energiekonsum: Beispiel-Kunde 1');
xlabel('Tage');
ylabel('Wh');

consumerData = load('..\..\Data\Change38_Consumer\Consumer2_Romano_Zgraggen.txt');
s = 1/96;
m = double(size(consumerData,1));
e = m/96.0;
% x = 1:s:e+1-s;
x = (consumerData(:,1) - consumerData(1,1))/(24*60*60*1000);
% x = consumerData(:,1);
% z = 1:m;
y = consumerData(:,3);
yc21 = consumerData(:,3);
yc22 = consumerData(7*96:end,3);
yc23 = consumerData(7*96:21*96,3);

figure(5);
stem(x(7*96:21*96), y(7*96:21*96), 'r', 'marker','none');
axis tight;
title('Energiekonsum: Beispiel-Kunde 2');
xlabel('Tage');
ylabel('Wh');

figure(6);
stem(x, y, 'marker','none');
axis tight;
title('Energiekonsum: Beispiel-Kunde 2');
xlabel('Tage');
ylabel('Wh');

% figure(7);
% stem(z, y, 'g', 'marker','none');
% axis tight;
% title('Energiekonsum: Beispiel-Kunde 2');
% xlabel('Tage');
% ylabel('Wh');

consumerData = load('..\..\Data\Change38_Consumer\Consumer3_Peter_Reiser.txt');
x = (consumerData(:,1) - consumerData(1,1))/(24*60*60*1000);
y = consumerData(:,3);

figure(8);
stem(x(1*96:14*96), y(1*96:14*96), 'r', 'marker','none');
axis tight;
title('Energiekonsum: Beispiel-Kunde 3');
xlabel('Tage');
ylabel('Wh');

figure(9);
stem(x, y, 'marker','none');
axis tight;
title('Energiekonsum: Beispiel-Kunde 3');
xlabel('Tage');
ylabel('Wh');


