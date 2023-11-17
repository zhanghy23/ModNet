% clear all ;
% clc ;
sum_rate_h=zeros(6,2);
sum_rate_hf=zeros(6,2);
fc = 4e9 ; % carrier frequency
M = 128 ; % the number of subcarriers
% load('F.mat');
% load('W.mat');
% F = 10 * eye(M) ; % only for test
df = 15e3 ; % subcarrier spacing
T = 1 / df ; % period of rectangular waveform
Mcp = 24 ;
% delay spread
lmax = 20 ;
% number of paths 
taps = 12 ;
vmax = 360 ; % maximum velocity : kn
% SNR range
SNRb_dB = (5:5:30);
SNR_linear = 10.^(SNRb_dB/10) ;
% QPSK power and sigma2
qam_mod = 4 ;
qam_bit= log2(qam_mod) ;
eng_sqrt = (qam_mod==2)+(qam_mod~=2)*sqrt((qam_mod-1)/6*(2^2));
sigmas2 = eng_sqrt * eng_sqrt ;
sigma2_cand = (sigmas2) ./ SNR_linear ;
% only for test
F_precode_OFDM = 10 * eye(M) ;
F_M = dftmtx(M) / sqrt(M) ;
R_CP = [zeros(M,Mcp),eye(M)] ;
A_CP = [zeros(Mcp,M-Mcp),eye(Mcp);eye(M)] ;
W1= F_M*R_CP;
F1=A_CP*F_M'; 
load('WF.mat');
W=median(W,1);
F=median(F,1);
W = squeeze(complex(W(1,1,:,:),W(1,2,:,:)));
F = squeeze(complex(F(1,1,:,:),F(1,2,:,:)));
W = W*sqrt(128)/norm(W,'fro');
F = F*sqrt(152)/norm(F,'fro');
N_iter = 20000 ;
ber_OFDM = zeros(size(SNRb_dB)) ;
ber_OFDM_DFTs = zeros(size(SNRb_dB)) ;
ber_OFDM_precode = zeros(size(SNRb_dB)) ;
ber_OFDM_precode_DFTs = zeros(size(SNRb_dB)) ;
ber_full = zeros(size(SNRb_dB)) ; % replace OFDM fully
ber_full_DFTs = zeros(size(SNRb_dB)) ;
for iter_time = 1:N_iter
    if mod(iter_time-1,N_iter/100) == 0
        fprintf('%3.2f%% finished \n',(iter_time-1)/(N_iter/100)) 
    end 
    % generate the channel
%     [H_t,H_sc] = channel_generate_OFDM...
%         (fc,df,vmax,lmax,M,Mcp,taps,1,20) ;
    H_t = squeeze(H_t_all_360_12(iter_time,:,:));
    H_sc = squeeze(H_sc_all_360_12(iter_time,:,:));
    H_precode_sc = H_sc * F_precode_OFDM ; % precoded OFDM
    H_e_full = W * H_t * F ; % replace OFDM fully
    noise_var = diag(W * W') ; % noise variance
    % generate the data
    data_bit = randi([0,1],qam_bit,M) ;
    data_sym = qammod(data_bit,qam_mod,'InputType','bit') ;
    data_sym = data_sym.' ;
    data_sym_DFTs = fft(data_sym,M) / sqrt(M) ;
    % equivalent channel matrix of precoded OFDM
    y_OFDM_wn = H_sc * data_sym ;
    y_OFDM_DFTs_wn = H_sc * data_sym_DFTs ;
    y_OFDM_precode_wn = H_precode_sc * data_sym ;
    y_OFDM_precode_DFTs_wn = H_precode_sc * data_sym_DFTs ;
    y_full_wn = H_e_full * data_sym ;
    y_full_DFTs_wn = H_e_full * data_sym_DFTs ;
    n_norm = sqrt(1/2) * (randn(M+Mcp,1) + 1j * randn(M+Mcp,1)) ;
    for a = 1:length(SNRb_dB)
        H_1 = squeeze(real(H_sc).*real(H_sc)+imag(H_sc).*imag(H_sc));
        H_1_diag = diag(H_1);
        sum_rate_h(a,1) = sum(log2(1+(H_1_diag./((sum(H_1,2)-H_1_diag)+1/SNR_linear(a)))),1)+sum_rate_h(a,1);
        sum_rate_h(a,2) = min(log2(1+(H_1_diag./((sum(H_1,2)-H_1_diag)+1/SNR_linear(a)))))+sum_rate_h(a,2);
        H = abs(H_e_full).*abs(H_e_full);
        H_diag = diag(H);
        sum_rate_hf(a,1) = sum(log2(1+(H_diag./((sum(H,2)-H_diag)+1*sum(H,2)/SNR_linear(a)))),1)+sum_rate_hf(a,1);
        sum_rate_hf(a,2) = min(log2(1+(H_diag./((sum(H,2)-H_diag)+1*sum(H,2)/SNR_linear(a)))))+sum_rate_hf(a,2);
        sigma2 = sigma2_cand(a) ;
        % add awgn noise
        n_rx_time = sqrt(sigma2) * n_norm ;
        n_rx_sc = F_M * R_CP * n_rx_time ;
        y_OFDM = y_OFDM_wn + n_rx_sc ;
        y_OFDM_DFTs = y_OFDM_DFTs_wn + n_rx_sc ;
        y_OFDM_precode = y_OFDM_precode_wn + n_rx_sc ;
        y_OFDM_precode_DFTs = y_OFDM_precode_DFTs_wn + n_rx_sc ;
        y_full = y_full_wn + W * n_rx_time ;
        y_full_DFTs = y_full_DFTs_wn + W * n_rx_time ;
        % LMMSE equalization of initial OFDM
        data_est_OFDM = (y_OFDM .* conj(diag(H_sc)))./ ...
            (abs(diag(H_sc)).*abs(diag(H_sc))+sigma2/sigmas2) ;
        bit_est = qamdemod(data_est_OFDM.',qam_mod,'OutputType','bit') ;
        ber_OFDM(a) = ber_OFDM(a) + sum(sum(bit_est~=data_bit)) / M / qam_bit ;
        % LMMSE equalization of DFT-spread OFDM
        data_est_OFDM_DFTs = (y_OFDM_DFTs .* conj(diag(H_sc)))./ ...
            (abs(diag(H_sc)).*abs(diag(H_sc))+sigma2/sigmas2) ;
        data_est_OFDM_DFTs = ifft(data_est_OFDM_DFTs,M) * sqrt(M) ;
        bit_est1 = qamdemod(data_est_OFDM_DFTs.',qam_mod,'OutputType','bit') ;
        ber_OFDM_DFTs(a) = ber_OFDM_DFTs(a) + sum(sum(bit_est1~=data_bit)) / M / qam_bit ;
        % LMMSE equalization of precoded OFDM
        data_est_OFDM_precode = (y_OFDM_precode .* conj(diag(H_precode_sc)))./ ...
            (abs(diag(H_precode_sc)).*abs(diag(H_precode_sc))+sigma2/sigmas2) ;
        bit_est = qamdemod(data_est_OFDM_precode.',qam_mod,'OutputType','bit') ;
        ber_OFDM_precode(a) = ber_OFDM_precode(a) + sum(sum(bit_est~=data_bit)) / M / qam_bit ;
        % LMMSE equalization of precoded DFT-spread OFDM
        data_est_OFDM_precode_DFTs = (y_OFDM_precode_DFTs .* conj(diag(H_precode_sc)))./ ...
            (abs(diag(H_precode_sc)).*abs(diag(H_precode_sc))+sigma2/sigmas2) ;
        data_est_OFDM_precode_DFTs = ifft(data_est_OFDM_precode_DFTs,M) * sqrt(M) ;
        bit_est1 = qamdemod(data_est_OFDM_precode_DFTs.',qam_mod,'OutputType','bit') ;
        ber_OFDM_precode_DFTs(a) = ber_OFDM_precode_DFTs(a) + sum(sum(bit_est1~=data_bit)) / M / qam_bit ;
        % LMMSE equalization with new modulation
        data_est_full = (y_full .* conj(diag(H_e_full)))./ ...
            (abs(diag(H_e_full)).*abs(diag(H_e_full))+sigma2/sigmas2*noise_var) ;
        bit_est = qamdemod(data_est_full.',qam_mod,'OutputType','bit') ;
        ber_full(a) = ber_full(a) + sum(sum(bit_est~=data_bit)) / M / qam_bit ;
        % LMMSE equalization with DFT-spread new modulation
        data_est_full_DFTs = (y_full_DFTs .* conj(diag(H_e_full)))./ ...
            (abs(diag(H_e_full)).*abs(diag(H_e_full))+sigma2/sigmas2*noise_var) ;
        data_est_full_DFTs = ifft(data_est_full_DFTs,M) * sqrt(M) ;
        bit_est = qamdemod(data_est_full_DFTs.',qam_mod,'OutputType','bit') ;
        ber_full_DFTs(a) = ber_full_DFTs(a) + sum(sum(bit_est~=data_bit)) / M / qam_bit ;
    end
end
ber_OFDM_precode = ber_OFDM_precode / N_iter ;
ber_OFDM = ber_OFDM / N_iter ;
ber_OFDM_precode_DFTs = ber_OFDM_precode_DFTs / N_iter ;
ber_OFDM_DFTs = ber_OFDM_DFTs / N_iter ;
ber_full_DFTs = ber_full_DFTs / N_iter ;
ber_full = ber_full / N_iter ;
figure ;
plot(SNRb_dB,ber_OFDM,'b-^','LineWidth',1.2) ;
hold on ;
% plot(SNRb_dB,ber_OFDM_precode,'-*','Color',[0.4940 0.1840 0.5560],'LineWidth',1.2) ;
% hold on ;
plot(SNRb_dB,ber_full,'m-p','LineWidth',1.2) ;
hold on ;
% plot(SNRb_dB,ber_OFDM_DFTs,'b-s','LineWidth',1.2) ;
% hold on ;
% plot(SNRb_dB,ber_OFDM_precode_DFTs,'-o','Color',[0.4940 0.1840 0.5560],'LineWidth',1.2) ;
% hold on ;
% plot(SNRb_dB,ber_full_DFTs,'m-v','LineWidth',1.2) ;
% hold on ;
legend('OFDM','proposed modulation') ;
set(gca,'YScale','log') ;
xlabel('SNR / dB') ;
grid on ;
sum_rate_h=sum_rate_h/N_iter;
sum_rate_hf=sum_rate_hf/N_iter;

function [H_t,H_sc] = channel_generate_OFDM(fc,df,vmax,lmax,M,Mcp,taps,frac_flag,Ng)
% generate the channel matrix (including CP) in time domain 
% fc: carrier frequency (Hz, e.g., 4e9)
% df: subcarrier spacing (Hz, e.g., 15e3)
% vmax: maximum mobility velocity (km/h, e.g., 500)
% lmax: maximum discrete delay (less than Mcp, e.g., 20)
% M: number of subcarriers (e.g., 128)
% Mcp: length of CP (more than lmax, e.g., 24)
% taps: number of paths (e.g., 4)
% frac_flag: whether fractional Doppler is considered,
% =1 : considering
% =0 : ignoring
% Ng: the possible grids of Doppler, if frac_flag=1, it doesn't work
% H_t: the channel matrix at time domain
% H_sc: the channel matrix at subcarrier domain

% generate parameters
% gain of each path
h_taps = sqrt(1/taps/2) * (randn(taps,1) + 1j*randn(taps,1)) ;
% delay and Doppler for each path
delay_grid = 1:1:lmax ;
Doppler_max = fc * vmax / (3e8 * 3.6) ;  % maximum Doppler frequency
if frac_flag == 0
    Doppler_grid = linspace(-Doppler_max,Doppler_max,Ng) ;
end
Doppler_taps = zeros(taps,1) ;
delay_taps = zeros(taps,1) ;
for p = 1:taps
    delay_taps(p) = randsample(delay_grid,1) ;
    if frac_flag == 1
        Doppler_taps(p) = Doppler_max * cos(rand*2*pi-pi) ;
    else
        Doppler_taps(p) = randsample(Doppler_grid,1) ;
    end
end
% generate the channel matrix by adding each path
H_t = zeros(M+Mcp,M+Mcp) ;
T = 1 / df ; % symbol period
for p = 1:taps
    % fetch parameters
    lp = delay_taps(p) ; kp = Doppler_taps(p) ;
    hp = h_taps(p) ; 
    delay_H = [zeros(lp,M+Mcp);eye(M+Mcp-lp),zeros(M+Mcp-lp,lp)] ;
    Doppler_H = diag(exp(T/M*1j*2*pi*kp*(-Mcp:1:M-1))) ;
    H_t = H_t + hp * delay_H * Doppler_H ;
end
F_M = dftmtx(M) / sqrt(M) ;
R_CP = [zeros(M,Mcp),eye(M)] ;
A_CP = [zeros(Mcp,M-Mcp),eye(Mcp);eye(M)] ;
H_sc = F_M * R_CP * H_t * A_CP * F_M' ;
end


        
        
    
    