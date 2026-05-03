%% adaptive anc system with deep anc lite
clear; clc; close all; rng(42);

%% paths
toolbox_path = fullfile(pwd,'toolbox');
pandar_path = fullfile(pwd,'PANDAR_database_1.0');
music_file = 'Its Over.mp3';
noise_folder = fullfile(pwd,'MS-SNSD','noise_test');

noise_names = {'AirConditioner','Babble','Neighbor','ShuttingDoor','AirportAnnouncements'};
nNoise = length(noise_names);

%% parameters
fs = 16000;
dur = 20;
N = dur*fs;
t = (0:N-1)'/fs;

SNR_mix = 0;
NFFT = 512;
wiener_len = 256;
fx_len = 256;

frame_len = round(0.020*fs);
hop = round(0.010*fs);
overlap = frame_len - hop;
fft_len = 320;
win = hann(frame_len,'periodic');

fprintf('ANC PROJECT\n');
fprintf('fs = %d Hz\n',fs);
fprintf('duration = %d sec\n',dur);
fprintf('input snr = %d dB\n\n',SNR_mix);

%% load music
music = load_audio_loop(music_file,fs,N);
music = music/(rms(music)+eps);
fprintf('music samples = %d\n\n',length(music));

%% paths
P_ir = make_primary_path(fs);
P_ir = P_ir/(norm(P_ir)+eps);

addpath(genpath(toolbox_path))

pf = fullfile(pandar_path,'BoseQC20','acoustic_booth','persons', ...
    'PANDAR_TF_001_person_BoseQC20.ita');

tf = ita_read(pf);
S_ir = resample(tf.timeData(:,1),fs,tf.samplingRate);
S_ir = S_ir(1:min(512,length(S_ir)));
S_ir = S_ir/(norm(S_ir)+eps);

fprintf('primary path taps = %d\n',length(P_ir));
fprintf('secondary path taps = %d\n\n',length(S_ir));

%% load noise
fprintf('loading real noise files\n');

ref_noise = cell(nNoise,1);
primary_noise = cell(nNoise,1);
noisy_sigs = cell(nNoise,1);

alpha = 10^(-SNR_mix/20);

for k = 1:nNoise
    fileList = dir(fullfile(noise_folder,[noise_names{k} '_*.wav']));
    testFile = fullfile(noise_folder,fileList(1).name);

    ref_noise{k} = load_audio_loop(testFile,fs,N);
    ref_noise{k} = ref_noise{k}/(rms(ref_noise{k})+eps);

    primary_noise{k} = filter(P_ir,1,ref_noise{k});
    primary_noise{k} = primary_noise{k}/(rms(primary_noise{k})+eps);
    primary_noise{k} = alpha*primary_noise{k};

    noisy_sigs{k} = music + primary_noise{k};

    fprintf('%s loaded from %s\n',noise_names{k},fileList(1).name);
end

fprintf('\n');

%% psd
[Rmusic,f_psd] = pwelch(music,hann(NFFT),NFFT/2,NFFT,fs);
Rnoise = cell(nNoise,1);

for k = 1:nNoise
    Rnoise{k} = pwelch(primary_noise{k},hann(NFFT),NFFT/2,NFFT,fs);
end

fprintf('psd bins = %d\n\n',length(Rmusic));

%% wiener anc
fprintf('wiener anc\n');

wiener_out = cell(nNoise,1);
wiener_fir = cell(nNoise,1);
wiener_cancel = cell(nNoise,1);
wiener_H = cell(nNoise,1);

for k = 1:nNoise
    sec_ref = filter(S_ir,1,ref_noise{k});

    w = design_wiener_anc(sec_ref,primary_noise{k},wiener_len);

    y = filter(w,1,ref_noise{k});
    anti = filter(S_ir,1,y);
    out = noisy_sigs{k} - anti;

    wiener_fir{k} = w;
    wiener_cancel{k} = y;
    wiener_out{k} = out;

    [H_show,f_show] = freqz(w,1,NFFT,fs);
    wiener_H{k} = abs(H_show);

    snr_w = snr_imp_aligned(music,primary_noise{k},out);

    fprintf('%s wiener snr = %.1f dB\n',noise_names{k},snr_w);
end

fprintf('\n');

%% fxlms and nfxlms
fprintf('fxlms and nfxlms\n');

fx_out = cell(nNoise,1);
nfx_out = cell(nNoise,1);
fx_weights = cell(nNoise,1);
nfx_weights = cell(nNoise,1);
fx_mu = zeros(nNoise,1);
nfx_mu = zeros(nNoise,1);

for k = 1:nNoise
    [fx_out{k},fx_weights{k},fx_mu(k),fx_snr] = pick_fxlms( ...
        ref_noise{k},noisy_sigs{k},primary_noise{k},music,S_ir,fx_len,false);

    [nfx_out{k},nfx_weights{k},nfx_mu(k),nfx_snr] = pick_fxlms( ...
        ref_noise{k},noisy_sigs{k},primary_noise{k},music,S_ir,fx_len,true);

    fprintf('%s fxlms snr = %.1f dB, mu = %.4f\n',noise_names{k},fx_snr,fx_mu(k));
    fprintf('%s nfxlms snr = %.1f dB, mu = %.4f\n',noise_names{k},nfx_snr,nfx_mu(k));
end

fprintf('\n');

%% deep anc lite
fprintf('deep anc lite\n');

Xtrain = [];
Ytrain = [];

for k = 1:nNoise
    [Xin,Yout] = make_deep_anc_training(ref_noise{k},wiener_cancel{k},fs,win,overlap,fft_len);
    Xtrain = [Xtrain; Xin];
    Ytrain = [Ytrain; Yout];
end

Xmean = mean(Xtrain,1);
Xstd = std(Xtrain,[],1) + eps;
Ymean = mean(Ytrain,1);
Ystd = std(Ytrain,[],1) + eps;

Xn = (Xtrain - Xmean)./Xstd;
Yn = (Ytrain - Ymean)./Ystd;

layers = [
    featureInputLayer(size(Xn,2))
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(256)
    reluLayer
    fullyConnectedLayer(size(Yn,2))
    regressionLayer
];

opts = trainingOptions('adam', ...
    'MaxEpochs',20, ...
    'MiniBatchSize',512, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','none');

deep_model = trainNetwork(Xn,Yn,layers,opts);

deep_out = cell(nNoise,1);
deep_cancel = cell(nNoise,1);

for k = 1:nNoise
    yhat = predict_deep_anc(deep_model,ref_noise{k},fs,win,overlap,fft_len,Xmean,Xstd,Ymean,Ystd,N);
    anti = filter(S_ir,1,yhat);
    deep_cancel{k} = yhat;
    deep_out{k} = noisy_sigs{k} - anti;

    snr_d = snr_imp_aligned(music,primary_noise{k},deep_out{k});
    fprintf('%s deep anc lite snr = %.1f dB\n',noise_names{k},snr_d);
end

fprintf('\n');

%% evaluation
fprintf('evaluation\n');

methods = {'Wiener ANC','FxLMS','NFxLMS','Deep ANC lite'};
all_outs = {wiener_out,fx_out,nfx_out,deep_out};

snr_results = zeros(nNoise,4);
stoi_noisy = zeros(nNoise,1);
stoi_results = zeros(nNoise,4);

for k = 1:nNoise
    stoi_noisy(k) = stoi(music,noisy_sigs{k},fs);

    for m = 1:4
        out = all_outs{m}{k};
        snr_results(k,m) = snr_imp_aligned(music,primary_noise{k},out);
        stoi_results(k,m) = stoi(music,out,fs);
    end
end

fprintf('\n%-22s | %12s %12s %12s %16s\n','Noise',methods{:});
fprintf('%s\n',repmat('-',1,105));

for k = 1:nNoise
    fprintf('%-22s | %11.1f dB %11.1f dB %11.1f dB %15.1f dB\n', ...
        noise_names{k},snr_results(k,1),snr_results(k,2),snr_results(k,3),snr_results(k,4));
end

fprintf('%s\n',repmat('-',1,105));
fprintf('%-22s | %11.1f dB %11.1f dB %11.1f dB %15.1f dB\n','Average', ...
    mean(snr_results(:,1)),mean(snr_results(:,2)),mean(snr_results(:,3)),mean(snr_results(:,4)));

fprintf('\n%-22s | %12s %12s %12s %16s %12s\n','Noise',methods{:},'Noisy');
fprintf('%s\n',repmat('-',1,120));

for k = 1:nNoise
    fprintf('%-22s | %12.3f %12.3f %12.3f %16.3f %12.3f\n', ...
        noise_names{k},stoi_results(k,1),stoi_results(k,2),stoi_results(k,3), ...
        stoi_results(k,4),stoi_noisy(k));
end

fprintf('%s\n',repmat('-',1,120));
fprintf('%-22s | %12.3f %12.3f %12.3f %16.3f %12.3f\n','Average', ...
    mean(stoi_results(:,1)),mean(stoi_results(:,2)),mean(stoi_results(:,3)), ...
    mean(stoi_results(:,4)),mean(stoi_noisy));

%% plots
colors5 = lines(nNoise);

figure('Name','Wiener ANC filters','Color','w','Position',[30 30 1100 550]);

for k = 1:nNoise
    subplot(2,3,k)
    plot(f_show/1000,wiener_H{k},'Color',colors5(k,:),'LineWidth',1.5)
    grid on
    xlabel('Frequency kHz')
    ylabel('Gain')
    title([noise_names{k} ' Wiener'])
end

subplot(2,3,6)
hold on
for k = 1:nNoise
    plot(f_show/1000,wiener_H{k},'Color',colors5(k,:),'LineWidth',1.3)
end
grid on
xlabel('Frequency kHz')
ylabel('Gain')
title('All Wiener filters')
legend(noise_names,'FontSize',7,'Location','southwest')

figure('Name','ANC filter coefficients','Color','w','Position',[30 30 1100 650]);

for k = 1:nNoise
    subplot(3,5,k)
    stem(wiener_fir{k},'Color',colors5(k,:),'MarkerSize',3)
    grid on
    title([noise_names{k} ' Wiener'])
end

for k = 1:nNoise
    subplot(3,5,k+5)
    stem(fx_weights{k},'Color',colors5(k,:),'MarkerSize',3)
    grid on
    title([noise_names{k} ' FxLMS'])
end

for k = 1:nNoise
    subplot(3,5,k+10)
    stem(nfx_weights{k},'Color',colors5(k,:),'MarkerSize',3)
    grid on
    title([noise_names{k} ' NFxLMS'])
end

sgtitle('Controller filter coefficients')

figure('Name','PSD comparison','Color','w','Position',[30 30 1100 450]);

for k = 1:nNoise
    subplot(2,3,k)
    semilogx(f_psd(2:end),10*log10(Rmusic(2:end)),'k--','LineWidth',1.5)
    hold on
    semilogx(f_psd(2:end),10*log10(Rnoise{k}(2:end)),'Color',colors5(k,:),'LineWidth',1.5)
    grid on
    xlim([50 fs/2])
    xlabel('Frequency Hz')
    ylabel('PSD dB')
    title([noise_names{k} ' PSD'])
    legend('Music','Noise','FontSize',7,'Location','southwest')
end

subplot(2,3,6)
semilogx(f_psd(2:end),10*log10(Rmusic(2:end)),'k--','LineWidth',2)
hold on

for k = 1:nNoise
    semilogx(f_psd(2:end),10*log10(Rnoise{k}(2:end)),'Color',colors5(k,:),'LineWidth',1.2)
end

grid on
xlim([50 fs/2])
xlabel('Frequency Hz')
ylabel('PSD dB')
title('All PSDs')
legend(['Music',noise_names],'FontSize',7,'Location','southwest')

k_d = 1;
n4 = 4*fs;

figure('Name','Time domain example','Color','w','Position',[30 30 1100 900]);

sigs_td = {music,noisy_sigs{k_d},wiener_out{k_d},fx_out{k_d},nfx_out{k_d},deep_out{k_d}};
labs_td = {'Clean music','Noisy input','Wiener ANC','FxLMS','NFxLMS','Deep ANC lite'};

for p = 1:length(sigs_td)
    subplot(length(sigs_td),1,p)
    plot(t(1:n4),sigs_td{p}(1:n4),'LineWidth',0.6)
    grid on
    xlabel('Time s')
    ylabel('Amp')
    title(labs_td{p})
    xlim([0 4])
end

figure('Name','SNR improvement','Color','w','Position',[30 30 1000 500]);

bar(snr_results)
grid on
set(gca,'XTickLabel',noise_names,'XTickLabelRotation',15)
legend(methods,'Location','northeast')
ylabel('SNR improvement dB')
title('SNR improvement')
yline(0,'--','No improvement','HandleVisibility','off')

figure('Name','STOI scores','Color','w','Position',[30 30 1000 450]);

bar([stoi_noisy,stoi_results])
grid on
set(gca,'XTickLabel',noise_names,'XTickLabelRotation',15)
legend(['Noisy input',methods],'Location','southeast')
ylabel('STOI')
ylim([0 1.05])
title('STOI scores')
yline(1.0,':','Perfect','HandleVisibility','off')

figure('Name','PSD before and after','Color','w','Position',[30 30 1000 500]);

[Pm,fp] = pwelch(music,512,256,512,fs);
[Pi,~] = pwelch(noisy_sigs{k_d},512,256,512,fs);
[Pw,~] = pwelch(wiener_out{k_d},512,256,512,fs);
[Pfx,~] = pwelch(fx_out{k_d},512,256,512,fs);
[Pnfx,~] = pwelch(nfx_out{k_d},512,256,512,fs);
[Pd,~] = pwelch(deep_out{k_d},512,256,512,fs);

semilogx(fp,10*log10(Pm),'k--','LineWidth',2)
hold on
semilogx(fp,10*log10(Pi),'LineWidth',1.5)
semilogx(fp,10*log10(Pw),'LineWidth',1.5)
semilogx(fp,10*log10(Pfx),'LineWidth',1.5)
semilogx(fp,10*log10(Pnfx),'LineWidth',1.5)
semilogx(fp,10*log10(Pd),'LineWidth',1.5)

grid on
xlim([50 fs/2])
xlabel('Frequency Hz')
ylabel('PSD dB per Hz')
title(['PSD before and after ' noise_names{k_d}])
legend('Clean music','Noisy input','Wiener ANC','FxLMS','NFxLMS','Deep ANC lite','Location','southwest')

figure('Name','NFxLMS convergence','Color','w','Position',[30 30 900 450]);

win_len = round(fs*0.5);
n_win = floor(N/win_len);
time_win = (0:n_win-1)'*win_len/fs;

hold on

for k = 1:nNoise
    nr = zeros(n_win,1);

    for w = 1:n_win
        keep = (w-1)*win_len + 1:w*win_len;
        res = nfx_out{k}(keep) - music(keep);
        nr(w) = 20*log10(rms(primary_noise{k}(keep))/(rms(res)+eps));
    end

    plot(time_win,nr,'Color',colors5(k,:),'LineWidth',1.5)
end

grid on
yline(0,'--','No reduction','HandleVisibility','off')
legend(noise_names,'Location','southeast')
xlabel('Time s')
ylabel('Noise reduction dB')
title('NFxLMS convergence')

figure('Name','Summary','Color','w','Position',[30 30 750 450]);

avg_snr = mean(snr_results,1);

bar(avg_snr)
grid on
set(gca,'XTickLabel',methods,'XTickLabelRotation',10)
ylabel('Average SNR improvement dB')
title('Average performance')
yline(0,'--','HandleVisibility','off')

for m = 1:length(methods)
    text(m,avg_snr(m)+0.1,[num2str(avg_snr(m),'%.1f') ' dB'], ...
        'HorizontalAlignment','center','FontWeight','bold','FontSize',11)
end

%% save figures and tables
fig_folder = fullfile(pwd,'project_figures');
result_folder = fullfile(pwd,'project_results');
if ~exist(fig_folder,'dir')
    mkdir(fig_folder);
end
if ~exist(result_folder,'dir')
    mkdir(result_folder);
end
fig_list = findall(0,'Type','figure');
[~,order] = sort([fig_list.Number]);
fig_list = fig_list(order);

for i = 1:length(fig_list)
    figure(fig_list(i))
    name = ['fig_' num2str(i,'%02d')];
    exportgraphics(fig_list(i),fullfile(fig_folder,[name '.png']),'Resolution',300);
    savefig(fig_list(i),fullfile(fig_folder,[name '.fig']));
end

snr_table = array2table(snr_results,'VariableNames',matlab.lang.makeValidName(methods),'RowNames',noise_names);
writetable(snr_table,fullfile(result_folder,'snr_results.csv'),'WriteRowNames',true);

stoi_table = array2table([stoi_results stoi_noisy], ...
    'VariableNames',matlab.lang.makeValidName([methods {'Noisy'}]), ...
    'RowNames',noise_names);

writetable(stoi_table,fullfile(result_folder,'stoi_results.csv'),'WriteRowNames',true);

summary_table = table( ...
    mean(snr_results(:,1)),mean(snr_results(:,2)),mean(snr_results(:,3)),mean(snr_results(:,4)), ...
    'VariableNames',{'WienerAvgSNR','FxLMSAvgSNR','NFxLMSAvgSNR','DeepANCLiteAvgSNR'});

writetable(summary_table,fullfile(result_folder,'summary_results.csv'));

%% summary
fprintf('\nPROJECT SUMMARY\n');
fprintf('wiener avg snr = %.1f dB\n',avg_snr(1));
fprintf('fxlms avg snr = %.1f dB\n',avg_snr(2));
fprintf('nfxlms avg snr = %.1f dB\n',avg_snr(3));
fprintf('deep anc lite avg snr = %.1f dB\n',avg_snr(4));

%% play audio
k_p = 1;
play_n = 5*fs;
scl = @(x) x/(max(abs(x))+eps);

disp('noisy input')
sound(scl(noisy_sigs{k_p}(1:play_n)),fs)
pause(5.5)
disp('wiener anc')
sound(scl(wiener_out{k_p}(1:play_n)),fs)
pause(5.5)
disp('fxlms')
sound(scl(fx_out{k_p}(1:play_n)),fs)
pause(5.5)
disp('nfxlms')
sound(scl(nfx_out{k_p}(1:play_n)),fs)
pause(5.5)
disp('deep anc lite')
sound(scl(deep_out{k_p}(1:play_n)),fs)

%% functions
function x = load_audio_loop(file_name,fs,N)

[x,fs0] = audioread(file_name);

if size(x,2) > 1
    x = mean(x,2);
end

if fs0 ~= fs
    x = resample(x,fs,fs0);
end

x = x(:);

if length(x) < N
    reps = ceil(N/length(x));
    x = repmat(x,reps,1);
end

x = x(1:N);
x = x - mean(x);
x = x/(rms(x)+eps);

end

function P_ir = make_primary_path(fs)

imp = [1; zeros(255,1)];
[b,a] = butter(3,[80 2500]/(fs/2),'bandpass');
P_ir = filter(b,a,imp);
P_ir = [zeros(15,1); P_ir(1:end-15)];

end

function S_ir = make_secondary_path(fs)

imp = [1; zeros(255,1)];
[b,a] = butter(4,[160 4000]/(fs/2),'bandpass');
S_ir = filter(b,a,imp);
S_ir = [zeros(8,1); S_ir(1:end-8)];

end

function w = design_wiener_anc(x,target,L)

x = x(:);
target = target(:);

N = min(length(x),length(target));
x = x(1:N);
target = target(1:N);

r_xx = xcorr(x,L-1,'biased');
r_xx = r_xx(L:end);
R_xx = toeplitz(r_xx);

r_dx = xcorr(target,x,L-1,'biased');
r_dx = r_dx(L:end);

reg = 1e-5*trace(R_xx)/L;
w = (R_xx + reg*eye(L))\r_dx;

end

function [out,anti,w] = run_fxlms(ref,noisy,S_ir,L,mu,use_norm)

ref = ref(:);
noisy = noisy(:);
S_ir = S_ir(:);

N = length(ref);
M = length(S_ir);

w = zeros(L,1);
xbuf = zeros(L,1);
xpbuf = zeros(L,1);
sbuf = zeros(M,1);
ybuf = zeros(M,1);

anti = zeros(N,1);
out = zeros(N,1);

leak = 0.9999;
small = 1e-8;

for n = 1:N
    xbuf = [ref(n); xbuf(1:end-1)];

    sbuf = [ref(n); sbuf(1:end-1)];
    xp = S_ir.'*sbuf;
    xpbuf = [xp; xpbuf(1:end-1)];

    y = w.'*xbuf;

    ybuf = [y; ybuf(1:end-1)];
    anti(n) = S_ir.'*ybuf;

    out(n) = noisy(n) - anti(n);

    if use_norm
        p = xpbuf.'*xpbuf + small;
        w = leak*w + (mu/p)*xpbuf*out(n);
    else
        w = leak*w + mu*xpbuf*out(n);
    end
end

end

function [best_out,best_w,best_mu,best_snr] = pick_fxlms(ref,noisy,noise,music,S_ir,L,use_norm)

mu_list = [0.0001 0.0005 0.001 0.002 0.005 0.01];

best_out = noisy;
best_w = zeros(L,1);
best_mu = mu_list(1);
best_snr = -Inf;

for i = 1:length(mu_list)
    mu = mu_list(i);
    [out,~,w] = run_fxlms(ref,noisy,S_ir,L,mu,use_norm);

    if check_signal(out,noisy)
        s = snr_imp_aligned(music,noise,out);

        if s > best_snr
            best_snr = s;
            best_out = out;
            best_w = w;
            best_mu = mu;
        end
    end
end

end

function ok = check_signal(out,input)

bad = any(isnan(out) | isinf(out));
large = max(abs(out)) > 10*max(abs(input));
quiet = rms(out) < 1e-6;

ok = ~bad && ~large && ~quiet;

end

function [Xin,Yout] = make_deep_anc_training(ref,cancel,fs,win,overlap,fft_len)

X = stft(ref,fs,'Window',win,'OverlapLength',overlap,'FFTLength',fft_len);
Y = stft(cancel,fs,'Window',win,'OverlapLength',overlap,'FFTLength',fft_len);

frames = min(size(X,2),size(Y,2));
X = X(:,1:frames);
Y = Y(:,1:frames);

Xin = [real(X).' imag(X).'];
Yout = [real(Y).' imag(Y).'];

end

function y = predict_deep_anc(net,ref,fs,win,overlap,fft_len,Xmean,Xstd,Ymean,Ystd,N)

X = stft(ref,fs,'Window',win,'OverlapLength',overlap,'FFTLength',fft_len);

Xin = [real(X).' imag(X).'];
Xin = (Xin - Xmean)./Xstd;

Yhat = predict(net,Xin);
Yhat = Yhat.*Ystd + Ymean;

nBins = size(X,1);

Yr = Yhat(:,1:nBins).';
Yi = Yhat(:,nBins+1:end).';
Y = Yr + 1i*Yi;

y = istft(Y,fs,'Window',win,'OverlapLength',overlap,'FFTLength',fft_len);
y = real(y(:));

if length(y) < N
    y = [y; zeros(N-length(y),1)];
end

y = y(1:N);
y = y/(rms(y)+eps);

end

function imp = snr_imp_aligned(music,noise,output)

N = min([length(music),length(noise),length(output)]);

mu = music(1:N);
ns = noise(1:N);
op = output(1:N);

maxLag = round(0.02*length(mu));

[c,lags] = xcorr(op,mu,maxLag,'coeff');
[~,pos] = max(abs(c));
lag = lags(pos);

if lag > 0
    op = op(lag+1:end);
    mu = mu(1:end-lag);
    ns = ns(1:end-lag);
elseif lag < 0
    lag = abs(lag);
    op = op(1:end-lag);
    mu = mu(lag+1:end);
    ns = ns(lag+1:end);
end

res = op - mu;

snr_in = 20*log10(rms(mu)/(rms(ns)+eps));
snr_out = 20*log10(rms(mu)/(rms(res)+eps));

imp = snr_out - snr_in;

end