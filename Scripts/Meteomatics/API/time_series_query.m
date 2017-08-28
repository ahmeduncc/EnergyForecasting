function [dn, data] = time_series_query(user, pwd, server, start_datum, period, ...
    resolution, parameters, lat, lon)

url = ['http://' server '/' start_datum period ':' resolution '/' parameters ...
    '/' num2str(lat) ',' num2str(lon) '/csv?connector=' version]

[s_,info]=urlread_auth(url,user,pwd);
info
s =char(s_);

data=[];
dn={};

ss=regexp(s, '[\f\n\r]', 'split');
for i = 2:length(ss) - 1
  z = regexp(ss{i}, ';', 'split');
%   dn(i-1)=datenum(z{1},'YYYY-mm-ddTHH:MM:SS');
  dn{i-1} = z{1};
  for j = 2:length(z)
      tmp(j-1) = str2num(z{j});
  end
  data = [data;tmp];
end

end

