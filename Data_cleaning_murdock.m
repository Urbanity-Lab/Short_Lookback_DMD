clear all;
close all;
clc
%% User Input
MA = 10;
%% Input Data 
queue_data = readtable('que_matrix_murdock.csv');
StateData = queue_data(1:end,1:end);
Y_raw = StateData{:,:};
[m n] = size(Y_raw);
Y = movmean(Y_raw,MA);


% subplot(2,2,4)
plot(Y(1,2000:2800),'linewidth',2)

Y_data = [Y_raw(1,:); Y_raw(7,:); Y_raw(8,:);  Y(1,:)];
for i = 1:4
    figure(1)
    subplot(4,1,i)
    plot(Y_data(i,2000:3000))
end
%% 

for i = 1:4
    for j = 1:n-1
        if Y_data(i,j+1)-Y_data(i,j)>=8;
            Y_data(i,j+1) = Y_data(i,j);
        end
    end
end

for i = 1:4
    figure(2)
    subplot(4,1,i)
    plot(Y_data(i,1:1000))
end

writematrix(Y_data,'Modifed_1,7,8_murdock.csv')


for i = 1:4
    for j = 1:n-50
        if Y_data(i,j)-Y_data(i,j+1)>=12;
            for k = 1:2: round((Y_data(i,j)-Y_data(i,j+1))/3)
            Y_data(i,j+k) = Y_data(i,j+k-1)-6;
            Y_data(i,j+k+1) = Y_data(i,j+k);
            end             
        end
    end
end

Y_data = max(Y_data,0)
for i = 1:4
    figure(3)
    subplot(4,1,i)
    plot(Y_data(i,1:1000))
end
%writematrix(Y_data,'Modifed_1,5,7,9_murdock.csv')

x = 1:1000;
for i = 1:4 
    y = Y_data(i,1:1000);
    xx = 1:.25:1000;
    yy = spline(x,y,xx);
    figure(4)
    subplot(4,1,i)
    plot(x,y,xx,yy)
end

