clear all;
close all;
clc
tic
%% User Input
window = 22060;
ob_start = window+10;
ob_end = window+310;
%ob_end = window+200;
observation = ob_end-ob_start;
predictionwindow = 270;
MA = 1;
em = 21;
w = 1.3;
link = 3;

%% Input Data 
data_que = readtable('Modifed_1,5,7,9_nospline.csv');
StateData = data_que(1:end,1:end);
Y_raw = StateData{1:4,:};
Y = movmean(Y_raw,MA,2);
Y_mean = mean(Y,2);
[Y_r Y_c]= size(Y_mean);
Y = Y-Y_mean;
[m n] = size(Y)

InputData = readtable('rouse_tran.csv');
signal_data = InputData(1:end,1:end);
U_sig = signal_data{:,2000:end};
U_sig = [U_sig(1,:); U_sig(5,:); U_sig(7,:); U_sig(2,:)];
%U_sig = [U_sig(1,:); U_sig(5,:); U_sig(7,:)];
%U_sig = [U_sig; U_sig; U_sig; U_sig]; 
size(U_sig)

%% Creating X,X',U and U' matrices
X = Y(:,ob_start:ob_end-1); %Points for creating dynamics
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
Diff = ((Pred(link,:)-Y(link,ob_end+1:ob_end+predictionwindow))).^2;
MAPE = sqrt(mean(sum(Diff(:))))
  
figure(1)
plot(Y_raw(link,ob_end+1:ob_end+predictionwindow)+Y_mean(link),'k','linewidth',2)
hold on
signal = Pred(link,:);
nbits = 4 % 16 qantization levels
qLevels = 2^nbits;
signalMin = min(Pred(link,:))  
signalMax = max(Pred(link,:))  
scalingFactor = (signalMax-signalMin)/qLevels;
signal = signal / scalingFactor;
signal = round(signal);
Pred(link,:) = signal * scalingFactor;
%plot(signal,'r','linewidth',1)
plot(Pred(link,:),'--r','linewidth',2.5)
hold on
plot(Y(link,ob_end+1:ob_end+predictionwindow),'b','linewidth',2)
hold on
plot(15*U_sig(link,ob_end+1:ob_end+predictionwindow),'--g','linewidth',2)
xlabel('Time (min)')
ylabel('Queue Length')
xticks([0 60 120 180 240 300 360 420 480 540 600 660 720 780 840 900 960 1020 1080 1140 1159])
xticklabels({'0','1','2','3','4','5','6','7','8','9','10','11','12,13','14','15','16','17','18','19','20'})
legend('Raw data','Predicted Queue Length','Actual queue length after Moving Average','signal')

figure(2)

Pred(link,:) = Pred(link,:)+Y_mean(link);
Pred(link,:) = max(Pred(link,:),0);
plot(Pred(link,:),'r','linewidth',2.5)
hold on
plot(Y(link,ob_end+1:ob_end+predictionwindow)+Y_mean(link),'b','linewidth',2)
hold on
plot(15*U_sig(link,ob_end+1:ob_end+predictionwindow),'--g','linewidth',2)
title ("Moving Average: " + MA + ", Training Time: " + observation/60 + ", RMSE " + MAPE + "%, Embedding: " + em)
xlabel('Time (min)')
ylabel('Queue Length')
xticks([0 60 120 180 240 300 360 420 480 540 600 660 720 780 840 900 960 1020 1080 1140 1159])
xticklabels({'0','1','2','3','4','5','6','7','8','9','10','11','12,13','14','15','16','17','18','19','20'})
legend('Predicted Queue Length','Actual queue length after Moving Average','signal')

A=[Pred(link,:)' Y(link,ob_end+1:ob_end+predictionwindow)' U_sig(link,ob_end+1:ob_end+predictionwindow)'];
T = array2table(A);
T.Properties.VariableNames(1:3) = {'Prediction','Actual','Signal'};
writetable(T,'file1.csv')
toc
%% Hankel Function.
function hm = myhenkel(A,l)
    [m,n] = size(A);
    hm = zeros(m*(l+1), n-l);
    for k = 1:l+1
        hm(m*(k-1)+1:m*k,1:n-l) = A(1:m,k:k+n-l-1);
    end
end
