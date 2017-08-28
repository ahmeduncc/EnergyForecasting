% Smooth energy production curve St. Jakobstatium

producerData = load('..\..\Data\Change38_Producer\Producer_StJakobstadium.txt');
start = producerData(1,1);
x = (producerData(:,1) - start)/(24*60*60*1000);
y = producerData(:,3)/10^3;
sum_1 = sum(y)

hoursPerDay = 8;
coeff24hMA = ones(1, hoursPerDay)/hoursPerDay;
z = filter(coeff24hMA, 1, y);
sum_2 = sum(z)

figure(1);
stem(x, y, 'r', 'marker','none');
axis tight;
title('Energieproduktion: St. Jakob-Park');
xlabel('Tage');
ylabel('kWh');

figure(2);
stem(x, z, 'b', 'marker','none');
axis tight;
title('Energieproduktion: St. Jakob-Park (smoothing)');
xlabel('Tage');
ylabel('kWh');
