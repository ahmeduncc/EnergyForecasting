function showFeaturePlots(dates, targets, temperature, dewPoint, ...
    temperatureUnit)
%SHOWFEATUREPLOTS Show plots of feature vectors (targets: GWh)
%   (hour of day) boxplot, (day of week) boxplot, (month of year) boxplot,
%   temperature-load correlation, (dew point)-load correlation

[~, ~, ~, hr] = datevec(dates);

% By Hour
figure;
boxplot(targets, hr+1);
xlabel('Hour'); 
ylabel('Energy Consumption (GWh)');
title('Actual Energy Consumption per Hour');

% By Weekday
figure;
boxplot(targets, weekday(dates), 'labels', ...
    {'Sun','Mon','Tue','Wed','Thu','Fri','Sat'});
ylabel('Energy Consumption (GWh)');
title('Actual Energy Consumption per Weekday');

% By Month
figure;
boxplot(targets, datestr(dates,'mmm'));
ylabel('Energy Consumption (GWh)');
title('Actual Energy Consumption per Month');

if exist('temperature', 'var')
    figure;
    scatter(temperature, targets);
    xlabel(strcat('Temperature', temperatureUnit));
    ylabel('Energy Consumption (GWh)');    
    title('Temperature and Energy Consumption');
end

if exist('dewPoint', 'var')
    figure;
    scatter(dewPoint, targets);
    xlabel(strcat('Dew Point', temperatureUnit));
    ylabel('Energy Consumption (GWh)');    
    title('Dew Point and Energy Consumption');
end

end