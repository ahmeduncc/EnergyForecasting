function matlab_sample_weather_api()
% ---------------------------------------------------------------
% Sample Program to access Meteomatics Weather API through Matlab
% ---------------------------------------------------------------

% Query time series

global user;
global pwd;
global server;

version = 'matlab_v_1.1'

% user   = 'insert-your-username-here';
% pwd    = 'insert-your-password-here';
user   = 'bfh';
pwd    = 'Ulupulumo621';
server = 'api.meteomatics.com';

% ---------------------------------------------------------------
% Sample to query a time series
% ---------------------------------------------------------------

% 46.934104, 7.440885
% lat   = 50;
% lon   = 10;
lat   = 46.934104;
lon   = 7.440885;

% start_datum = '2016-12-24T07:00:00Z';
start_datum = '2017-03-01T00:00:00+00:00';
% period      = 'P5DT3H15M'; % period of 5 days, 3 hours, 15 min 
period      = 'P9DT23H45M';
% resolution  = 'PT15M'; % 15 min resolution
resolution  = 'PT15M'; 

% parameters = 't_2m:C,d_2m:C';  % Temperature and Dew Point at 2m
parameters = 'solar_power_installed_capacity_10_tracking_type_tilted-north-south-tracking_orientation_180_tilt_25:MW,solar_power_installed_capacity_10_tracking_type_fixed_orientation_200_tilt_25:MW'; 

[dn,data] = time_series_query_meteocache(start_datum,period,resolution,parameters,lat,lon);

% data_length = length(data(:,1))
% dn

figure

plot(dn,data(:,1),'b')
axis tight;
hold on
plot(dn,data(:,2),'r')
axis tight;

datetick('x','mm/dd/yy HH:MM');


% ---------------------------------------------------------------
% Sample to Query a domain
% ---------------------------------------------------------------

% lat_oben   = 52;
% lon_links  = 4;
% lat_unten  = 46;
% lon_rechts =15;
% 
% lat_px = 300;
% lon_px = 600;
% 
% termin = datenum(2016,12,24,15,35,0);
% 
% parameter = 't_2m:C';
% 
% data=domain_query_meteocache(termin,parameter,lat_oben,lon_links,lat_unten,lon_rechts,lat_px,lon_px);
% 
% 
% figure,imagesc(data),colorbar


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function data=domain_query_meteocache(dn,parameter,lat_oben,lon_links,lat_unten,lon_rechts,lat_px,lon_px)

global user;
global pwd;
global server;

time=datestr(dn,'YYYY-mm-ddTHH:MM:SSZ')

    
url=['http://' server '/' time '/' parameter '/' num2str(lat_oben) ',' ...
     num2str(lon_links) '_' num2str(lat_unten) ',' num2str(lon_rechts) ':' num2str(lat_px) 'x' num2str(lon_px) '/csv?connector=' version];

[s_,info]=urlread_auth(url,user,pwd);
info
s =char(s_);

ss=regexp(s, '[\f\n\r]', 'split');
data=[];

for i=4:length(ss)-1
  z=regexp(ss{i}, ';', 'split');
  d=cell2mat([cellfun(@str2num,z(1:end),'un',0).']);
  data(i-3,:)=d(2:end);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [dn,data]=time_series_query_meteocache(start_datum,period,resolution,parameters,lat,lon)

global user;
global pwd;
global server;

url = ['http://' server '/' start_datum period ':' resolution '/' parameters ...
    '/' num2str(lat) ',' num2str(lon) '/csv?connector=' version];
url

[s_,info]=urlread_auth(url,user,pwd);
info
s =char(s_);

data=[];
dn=[];

ss=regexp(s, '[\f\n\r]', 'split');
for i=2:length(ss)-1
  z=regexp(ss{i}, ';', 'split');
  dn(i-1)=datenum(z{1},'YYYY-mm-ddTHH:MM:SS');
  for j=2:length(z)
      tmp(j-1)=str2num(z{j});
  end
  data = [data;tmp];
end
