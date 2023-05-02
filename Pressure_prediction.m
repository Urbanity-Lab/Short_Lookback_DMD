
clear all;
close all;
clf
clc
%% User Input
window = 25000;
ob_start = window+10;
ob_end = window+600;
observation = ob_end-ob_start;
predictionwindow = 300;
MA = 20;
em = 21;
w = 0;
link = 1;


%% Input Data 
data_que = readtable('que_matrix_murdock.csv');
StateData = data_que(1:end-1,1:end);
Y_raw = StateData{:,:}; % All data points
[m n] = size(Y_raw);
Y = movmean(Y_raw,MA);
%writematrix(Y,'Training.csv')
InputData = readtable('murdock_tran.csv');
signal_data = InputData(1:end-1,1:end);
U_sig = signal_data{:,:};
%% Creating X,X',U and U' matrices
X = Y(:,ob_start:ob_end-1); %Points for creating dynamics
%writematrix(Y,'Training.csv')

%% %Noise 
% k = linspace(0,0.1,observation);                               % Time (Index) Vector
%     for i = 1:m
%         varn = rand(size(k));                       % Variance Vector
%         noise(i,:) = sqrt(varn).*randn(size(k))+1;      % Signal + Noise
%     end
%      mean(noise,2)
%      figure(6)
%      plot(noise(1,:))
%      figure(7)
%      plot(X(1,:)+noise(1,:),"linewidth",1.5)
%      hold on
%      plot(X(1,:),"linewidth",1.5)
%      xlabel('Time (min)')
%      ylabel('Queue Length')
%      xticks([0 60 120 180 240 300 360 420 480 540 600 660 720 780 840 900 960 1020 1080 1140 1159])
%      xticklabels({'0','1','2','3','4','5','6','7','8','9','10','11','12,13','14','15','16','17','18','19','20'})
%      legend('Data+Noise','Data')
%      X = X + noise;

%X = movmean(X,MA+40);
X = myhenkel(X,em);
Xp  = Y(:,ob_start+1:ob_end);
Xp = myhenkel(Xp,em);
Ups = w*U_sig(:,ob_start:ob_end-1);
Ups = myhenkel(Ups,em);
U_1 = U_sig(:,1:end);
U_1 = myhenkel(U_1,em);
%% SVD Analysis 
Omega = [X;Ups];
[U,Sig,V] = svd(Omega,'econ');
thresh = 1e-9;
rtil = length(find(diag(Sig)>thresh));
U    = U(:,1:rtil); 
Sig  = Sig(1:rtil,1:rtil);
V    = V(:,1:rtil);
%% A and B matrix calculation 
A = Xp(1:end,:)*V*inv(Sig)*U(1:end/2,1:end)';
B = Xp(1:end,:)*V*inv(Sig)*U(end/2+1:end,1:end)';
%% 
X_K = Xp(:,end);
Pred = zeros(m,predictionwindow);
Update = Y(:,ob_start:ob_end);
%% 
%X_K = Xp(:,end); % First Input for Prediction
for i= 1:predictionwindow
     x_k = A(end-(m-1):end,:)*X_K + B(end-(m-1):end,:)*U_1(:,observation+1+i);
     Pred(:,i) = x_k; % 
     Update = [Update(:,2:end) x_k]; % Updating 
     Update_1 = myhenkel(Update,em);
     X_K = Update_1(:,end);     
 end
%% Plot
[m_pred n_pred] = size(Pred)
%Diff = abs((Pred(link,:)-Y(link,ob_end+1:ob_end+predictionwindow)).^2; 
Diff = abs((Pred(link,:)-Y(link,ob_end+1:ob_end+predictionwindow)))/Y(link,ob_end+1:ob_end+predictionwindow);
MAPE = mean(sum(Diff(:)))*100
  

figure(2)
plot(Pred(link,:),'r','linewidth',2.5)
hold on
plot(Y(link,ob_end+1:ob_end+predictionwindow),'b','linewidth',2)
hold on
plot(15*U_sig(link,ob_end+1:ob_end+predictionwindow),'--g','linewidth',2)
title ("Moving Average: " + MA + ", Training Time: " + observation/60 + ", MAPE: " + MAPE + "%, Embedding: " + em)
xlabel('Time (min)')
ylabel('Queue Length')
xticks([0 60 120 180 240 300 360 420 480 540 600 660 720 780 840 900 960 1020 1080 1140 1159])
xticklabels({'0','1','2','3','4','5','6','7','8','9','10','11','12,13','14','15','16','17','18','19','20'})
legend('Predicted Queue Length','Actual queue length after Moving Average','signal')

A=[Pred(link,:)' Y(link,ob_end+1:ob_end+predictionwindow)'];
T = array2table(A);
T.Properties.VariableNames(1:2) = {'Prediction','Actual'};
writetable(T,'Murdock_2210_2570_2870.csv')



%% Hankel Function.
function hm = myhenkel(A,l)
    [m,n] = size(A);
    hm = zeros(m*(l+1), n-l);
    for k = 1:l+1
        hm(m*(k-1)+1:m*k,1:n-l) = A(1:m,k:k+n-l-1);
    end
end
