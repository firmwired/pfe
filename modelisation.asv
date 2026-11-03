 
clear; clc; close all;
 
Fs = 1600;        % fréquence d'echantillonnage
T = 1;            % durée de chaque chantillon   
t = 0:1/Fs:T-1/Fs;    
N_samples = length(t);
f_rpm = 1800;    % vitess nominale du moteur 
fr = f_rpm / 60;       % fréquence rotation fondamentale (30 Hz)

plot(t, t)

% Dataset virtuel:
num_samples_per_class = 300; 
classes = {'sain', 'perturbation_mecanique', 'surcharge_electrique'};

% features list: [RMS_vibration, amplitude_FFT1x, amplitude_FFT2x, courant_moy]
% amplitude_FFT1x == fréquence de rotation
% amplitude_FFT2x == fréquence d'harmonique
X_features = zeros(num_samples_per_class * 3, 4); 
Y_labels = cell(num_samples_per_class * 3, 1);
 
index = 1;

fprintf('génération du dataset virtuel\n');
 
for c = 1:length(classes)
    current_class = classes{c};
    
    for i = 1:num_samples_per_class
        % Modélisation basique du signaux (moteur, bruit..)
        
        % bruit thermique et mécanique
        % gaussian noise
        bruit_vibr = 0.1 * randn(1, N_samples); 
        bruit_elec = 0.05 * randn(1, N_samples);
        
        switch current_class
            case 'sain'
                % si moteur sain: vibration, courant est nominal
                signal_vibr = bruit_vibr + 0.2 * sin(2*pi*fr*t); 
                signal_courant = 1.5 + bruit_elec;
                
            case 'perturbation_mecanique'
                % cournat reste presque nominal avec une faible ondulation
                % et un Pic énorme à la fréquence de rotation et harmonique
                
                force_pert_mec = 2.5 * sin(2*pi*fr*t) + 0.8 * sin(2*pi*2*fr*t);
                signal_vibr = bruit_vibr + force_pert_mec;
                signal_courant = 1.6 + 0.1 * sin(2*pi*fr*t) + bruit_elec;
                
            case 'surcharge_electrique'
                % surcharge: courant élevé avec vibration légèrement augmentée
                signal_vibr = bruit_vibr + 0.4 * sin(2*pi*fr*t);
                signal_courant = 3.5 + bruit_elec;
        end
        
        % fenetre hann 
        fenetre = hann(N_samples)';
        % on multiple par window pour atténuer les extremité de notre signal
        vibr_fenetree = signal_vibr .* fenetre;
        
        % FFT
        spectre = abs(fft(vibr_fenetree)) / N_samples;
        spectre_unilateral = 2 * spectre(1:N_samples/2+1);
        f_vecteur = Fs*(0:(N_samples/2))/N_samples;
        
         
        rms_vibr = rms(signal_vibr);
        
        % amplitude du fréquence de rotation 1X = 30 Hz
        [~, idx_1x] = min(abs(f_vecteur - fr));
        amp_1x = spectre_unilateral(idx_1x);

        % amplitude du fréquence de rotation 1X = 30 Hz
        [~, idx_2x] = min(abs(f_vecteur - (2*fr)));
        amp_2x = spectre_unilateral(idx_2x);
        
        courant_moyen = mean(signal_courant);            
        X_features(index, :) = [rms_vibr, amp_1x, amp_2x, courant_moyen];
        Y_labels{index} = current_class;
                
        if i == 1
            data.(current_class).t = t;
            data.(current_class).vibr = signal_vibr;
            data.(current_class).f = f_vecteur;
            data.(current_class).spectre = spectre_unilateral;
        end
        
        index = index + 1;
    end
end

 
fprintf('training the model\n');

% pour cet modélisation, on utilise un classifieur simple decision tree
% deploiment reél va utilisé un modèle quantifié sur µcontrôleur
modele_IA = fitctree(X_features, Y_labels);
 
cv_modele = crossval(modele_IA);
erreur_classif = kfoldLoss(cv_modele);
fprintf('accuracy : %.2f %%\n', (1 - erreur_classif) * 100);
 
figure('Name', 'analyse temporelle et spectrale', 'Position', [100, 100, 1000, 600]);

 
subplot(3, 2, 1);
plot(data.sain.t, data.sain.vibr, 'b'); 
title('temporel : sain');
xlabel('t'); ylabel('amp vibr'); 
grid on;

subplot(3, 2, 2);
plot(data.sain.f, data.sain.spectre, 'b'); 
title('FFT : sain');
xlim([0 100]); xlabel('f (Hz)'); 
ylabel('amplitude'); 
grid on;
 
subplot(3, 2, 3);
plot(data.perturbation_mecanique.t, data.perturbation_mecanique.vibr, 'r'); title('temporel : défaut mécanique ');
xlabel('t'); ylabel('amp vibr');
grid on;

subplot(3, 2, 4);
plot(data.perturbation_mecanique.f, data.perturbation_mecanique.spectre, 'r'); title('FFT : défaut mécanique ');
xlim([0 100]); xlabel('f (Hz)'); ylabel('amp'); grid on;

 
subplot(3, 2, 5);
plot(data.surcharge_electrique.t, data.surcharge_electrique.vibr, 'k');
title('Temporel : surcharge_electrique');
xlabel('t'); 
ylabel('amp vibr'); 
grid on;
subplot(3, 2, 6);
plot(data.surcharge_electrique.f, data.surcharge_electrique.spectre, 'k'); 
title('FFT : surcharge_electrique');
xlim([0 100]);
xlabel('f (Hz)'); 
ylabel('amplitude'); 
grid on;

 
figure('Name', 'performance');
predictions = predict(modele_IA, X_features);
confusionchart(Y_labels, predictions, 'Title', 'confusion matrix');