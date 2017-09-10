%% Matlab neural network toolbox examples

%% Initialization
clear; close all; clc;

%% Simulation with concurrent inputs in a Static Network

net = linearlayer;
net.inputs{1}.size = 2;
net.layers{1}.dimensions = 1;
% net.layers{1}
net.IW{1,1} = [1 2];
net.b{1} = 0;
P = [1 2 2 3; 2 1 3 1];
A = net(P)

%% Simulation with sequential inputs in a dynamic network

net = linearlayer([0 1]);
net.inputs{1}.size = 1;
net.layers{1}.dimensions = 1;
net.biasConnect = 0;
net.IW{1,1} = [1 2];
P = {1 2 3 4};
A = net(P)
view(net);

%%  simple time series problem

x = {0 -1 1 1 0 -1 1 0 0 1};
t = {0 -1 0 2 1 -1 0 1 0 1};
net = linearlayer(1:2,0.01);
% net.trainParam.showCommandLine = true;
% [x,xi,ai,t] = preparets(net,{},{},t);
[Xs,Xi,Ai,Ts] = preparets(net,x,t)
net = train(net,Xs,Ts,Xi,Ai);
% biases_ = net.biases{1}
% inputWeights_ = net.inputWeights{1}
bias_w = net.b{1}
input_w = net.IW{1,1}
view(net);
Y = net(Xs,Xi)
perf = perform(net,Ts,Y)

%% simple NAR

T = simplenar_dataset;
net = narnet(1:2,10);
[Xs,Xi,Ai,Ts] = preparets(net,{},{},T);
net = train(net,Xs,Ts,Xi,Ai);
% view(net);
[y1,xfo,afo] = net(Xs,Xi,Ai);
perf = perform(net,Ts,y1)
plotresponse(Ts,y1)
% plotwb(net);
[netc,xic,aic] = closeloop(net,xfo,afo);
plot_index = 100;
[y2,xfc,afc] = netc(cell(0,plot_index),xic,aic);
figure, plot(1:(plot_index), cell2mat(y2), 'b');

