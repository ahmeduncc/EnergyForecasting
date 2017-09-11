function [errPct, err] = showPerformanceMetrics(net, features, targets, ...
    title, dates, gwhFactor)
%SHOWPERFORMANCEMETRICS  calculates and prints performance metrics for
% a given neural network 
%   calculated metrics: MSE, RMSE, MAE, MAPE

forecast = net(features')';
err = targets - forecast;
errPct = abs(err./targets)*100;
performance = perform(net, targets, forecast);
MSE = mse(net, targets, forecast);
RMSE = sqrt(MSE);
MAE = mae(err);
MAPE = mean(errPct, 'omitnan');

if exist('dates', 'var')
    figure('Name', title);
    if exist('gwhFactor', 'var')
        fitPlot(dates, [targets*gwhFactor  forecast*gwhFactor], ...
            err*gwhFactor);
    else
        fitPlot(dates, [targets  forecast], err);
    end
end

fprintf('\n');
fprintf(title);
fprintf('\nNetwork performance: %0.4f \n', performance);
fprintf('Mean Squared Error (MSE): %0.4f \n', MSE);
fprintf('Root Mean Squared Error (RMSE): %0.4f \n', RMSE);
fprintf('Mean Absolute Error (MAE): %0.4f \n', MAE);
fprintf('Mean Absolute Percent Error (MAPE): %0.6f%% \n', MAPE);

end

