clear;
%% RAVDESS DATA %%
directory='../Ravdess_Dataset/test_wavdata/';
folders=dir(directory);
for i=3:length(folders)
    current=dir(strcat(directory,folders(i).name));
    for j=3:length(current)
        extract_prosody(strcat(directory,folders(i).name,'/',current(j).name),20,10);
        delete(strcat(directory,folders(i).name,'/',current(j).name));
    end
end

%% EMODB %%
% directory='../emodb_wav/test_wavdata/';
% folders=dir(directory);
% for i=4:length(folders)
%     current=dir(strcat(directory,folders(i).name));
%     for j=4:length(current)
%         extract_prosody(strcat(directory,folders(i).name,'/',current(j).name),20,10);
%         delete(strcat(directory,folders(i).name,'/',current(j).name));
%     end
% end

function extract_prosody(file_name,frame_length, frame_overlap)
[x, fs] = audioread(file_name);

[F0, T, R] = spPitchTrackCorr(x, fs, frame_length, frame_overlap, [], 'plot');
% Fit a legendre polynomial of order 16
% syms F0;
% roots=vpasolve(legendreP(16,F0) == 0);
% Normalize
x=x/(max(abs(x)));
[energy_roots, zcr_roots] = rms_zcr_features(x,fs,frame_length, frame_overlap);
[epochs,epoch_amps] = gci_features(x,fs,frame_length,frame_overlap);
% duration = length(x)/fs;
% Write to file
features= [transpose(F0) transpose(energy_roots) transpose(zcr_roots) transpose(epochs) transpose(epoch_amps)];
% fileID = fopen(strcat(file_name,'_prosody'),'w');
% fprintf(fileID,'%6.6f ',features);
dlmwrite(strcat(file_name,'_prosody'),features)
end

function [epochs,epoch_amps] = gci_features(speech,fs,frame_length,frame_overlap)
% Returns a vector of number of epochs per frame
[gci,goi] = dypsa(speech,fs); % from VOICEBOX
N = length(speech);
nsample = round(frame_length  * fs / 1000); % convert ms to points
noverlap = round(frame_overlap * fs / 1000); % convert ms to points
pos = 1; i = 1;
while (pos+nsample < N)
     cond = gci>=pos & gci<(pos+nsample);
     epoch_locations = gci(cond);
     epochs(i)=length(epoch_locations);
     total=0;
     for j=1:length(epoch_locations)
         total=total+speech(epoch_locations(j));
     end
     if isempty(epoch_locations)
        epoch_amps(i)=0;
     else
        epoch_amps(i)=total/length(epoch_locations);
     end
     pos = pos + (nsample - noverlap);
     i = i + 1;
end
end

function [energy, zcr] = rms_zcr_features(speech,fs,frame_length,frame_overlap)
% Set Paramaters
N = length(speech);
nsample = round(frame_length  * fs / 1000); % convert ms to points
noverlap = round(frame_overlap * fs / 1000); % convert ms to points

pos = 1; i = 1;
while (pos+nsample < N)
     frame = speech(pos:pos+nsample-1);
     %frame = frame - mean(frame); % mean subtraction
     energy(i)=rms(frame);
     zcr(i)= sum(abs(diff(frame>0)))/length(frame);
     pos = pos + (nsample - noverlap);
     i = i + 1;
end

% Fit a legendre polynomial of order 16
% syms energy;
% energy_roots=vpasolve(legendreP(16,energy) == 0);
% syms zcr;
% zcr_roots=vpasolve(legendreP(16,zcr) == 0);
end

function [F0, T, R] = spPitchTrackCorr(x, fs, frame_length, frame_overlap, maxlag, show)
 %% Initialization
 N = length(x);
 if ~exist('frame_length', 'var') || isempty(frame_length)
     frame_length = 30;
 end
 if ~exist('frame_overlap', 'var') || isempty(frame_overlap)
     frame_overlap = 20;
 end
 if ~exist('maxlag', 'var')
     maxlag = [];
 end
 if ~exist('show', 'var') || isempty(show)
     show = 0;
 end 
 nsample = round(frame_length  * fs / 1000); % convert ms to points
 noverlap = round(frame_overlap * fs / 1000); % convert ms to points

 %% Pitch detection for each frame
 pos = 1; i = 1;
 while (pos+nsample < N)
     frame = x(pos:pos+nsample-1);
     frame = frame - mean(frame); % mean subtraction
     R(:,i) = spCorr(frame, fs);
     F0(i) = spPitchCorr(R(:,i), fs);
     pos = pos + (nsample - noverlap);
     i = i + 1;
 end
 T = (round(nsample/2):(nsample-noverlap):N-1-round(nsample/2))/fs;

if show 
    % plot waveform
%     subplot(2,1,1);
%     t = (0:N-1)/fs;
%     plot(t, x);
%     legend('Waveform');
%     xlabel('Time (s)');
%     ylabel('Amplitude');
%     xlim([t(1) t(end)]);
% 
%     % plot F0 track
%     subplot(2,1,2);
%     plot(T,F0);
%     legend('pitch track');
%     xlabel('Time (s)');
%     ylabel('Frequency (Hz)');
%     xlim([t(1) t(end)]);
end
end

function [r] = spCorr(x, fs, maxlag, show)
 %% Initialization
 if ~exist('maxlag', 'var') || isempty(maxlag)
     maxlag = fs/50; % F0 is greater than 50Hz => 20ms maxlag
 end
 if ~exist('show', 'var') || isempty(show)
     show = 0;
 end

 %% Auto-correlation
 r = xcorr(x, maxlag, 'coeff');

 if show
     %% plot waveform
     t=(0:length(x)-1)/fs;        % times of sampling instants
     subplot(2,1,1);
     plot(t,x);
     legend('Waveform');
     xlabel('Time (s)');
     ylabel('Amplitude');
     xlim([t(1) t(end)]);

     %% plot autocorrelation
     d=(-maxlag:maxlag)/fs;
     subplot(2,1,2);
     plot(d,r);
     legend('Auto-correlation');
     xlabel('Lag (s)');
     ylabel('Correlation coef');
 end
end

function [f0] = spPitchCorr(r, fs)
 % search for maximum  between 2ms (=500Hz) and 20ms (=50Hz)
 ms2=floor(fs/500); % 2ms
 ms20=floor(fs/50); % 20ms
 % half is just mirror for real signal
 r = r(floor(length(r)/2):end);
 [maxi,idx]=max(r(ms2:ms20));
 f0 = fs/(ms2+idx-1);
end