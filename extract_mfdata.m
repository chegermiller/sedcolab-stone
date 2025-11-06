function[amp,corr,phase,time,r,u,w] = extract_mfdata(trial_name,DD_file_start,theta)
% function that extracts amp, phase, cor, and timestamp, computes
% velocities given angle (in deg) between beams
% for aux1H, aux1L, aux2
% concatenates all files for a trial
% returns time, range, amplitudes, phases, and correlations (third axis for amp, phase, and cor goes aux1_H, aux1_L, aux2)

    clear vphs vamp vcor vtime sphs samp scor stime semb vemb slant_inds vert_inds amp_old phase_old cor_old vel_time w u Vs
    n=1;

    while true
        % read in MF data and time sync data
        mf_data = load(['MFdata\',trial_name,DD_file_start,num2str(n),'.mat']);
        data = mf_data;

        % clear old variables
        clear vert_p vert_a vert_c vert_timestamp slant_p slant_a slant_c slant_timestamp slant_ensemble vert_ensemble
        
        % read in 1400kHz data
        vert_p(:,:,1)=data.Data.DragonDop1_Phase_Aux_1H_1400kHz;
        vert_a(:,:,1)=data.Data.DragonDop1_Amp_Aux_1H_1400kHz ...
               .*data.Data.DragonDop1_Range;
        vert_c(:,:,1)=data.Data.DragonDop1_Cor_Aux_1H_1400kHz;
        vert_p(:,:,2)=data.Data.DragonDop1_Phase_Aux_1L_1400kHz;
        vert_a(:,:,2)=data.Data.DragonDop1_Amp_Aux_1L_1400kHz ...
               .*data.Data.DragonDop1_Range;
        vert_c(:,:,2)=data.Data.DragonDop1_Cor_Aux_1L_1400kHz;
        vert_timestamp = data.Data.DragonDop1_TimeStamp;
        vert_ensemble = data.Data.DragonDop1_Ensemble;


        slant_p(:,:,1)=data.Data.DragonDop2_Phase_Aux_2_1400kHz;
        slant_a(:,:,1)=data.Data.DragonDop2_Amp_Aux_2_1400kHz ...
               .*data.Data.DragonDop2_Range;
        slant_c(:,:,1)=data.Data.DragonDop2_Cor_Aux_2_1400kHz;
        slant_timestamp = data.Data.DragonDop2_TimeStamp;
        slant_ensemble = data.Data.DragonDop2_Ensemble;


        % concatenation
        if(n==1)
            vphs=vert_p;
            vamp=vert_a;
            vcor=vert_c;
            vtime = vert_timestamp;
            vemb = vert_ensemble;

            sphs=slant_p;
            samp=slant_a;
            scor=slant_c;
            stime = slant_timestamp;
            semb = slant_ensemble;
        else
            vphs=cat(1,vphs,vert_p);
            vamp=cat(1,vamp,vert_a);
            vcor=cat(1,vcor,vert_c);
            vtime = cat(1,vtime,vert_timestamp);
            vemb = cat(1,vemb,vert_ensemble);

            sphs=cat(1,sphs,slant_p);
            samp=cat(1,samp,slant_a);
            scor=cat(1,scor,slant_c);
            stime = cat(1,stime,slant_timestamp);
            semb = cat(1,semb,slant_ensemble);

        end
        r = data.Data.DragonDop2_Range;
        if exist(['MFdata\',trial_name,DD_file_start,num2str(n+1),'.mat'])
            n = n+1;
        else
            break
        end
    end

    
    % find common ensemble numbers between Dop1 and Dop2
    [common_ensemble, vert_inds, slant_inds] = intersect(vemb, semb);

    % make new arrays with insecting data from beams
    amp_old = cat(3,vamp(vert_inds,:,:),samp(slant_inds,:));
    phase_old = cat(3,vphs(vert_inds,:,:),sphs(slant_inds,:));
    cor_old = cat(3,vcor(vert_inds,:,:),scor(slant_inds,:));
    time_old = vtime(vert_inds); % find intersecting times

    % need to make a new grid for ensembles
    dt = mean(diff(time_old)); % find average dt
    max_ens = max(common_ensemble); % max ensemble number
    min_ens = min(common_ensemble); % min ensemnle number

    % create new list of ensemble numbers with no gaps
    new_ens = min_ens:1:max_ens;

    

    % New time axis (constant dt)
    time = time_old(1) + (new_ens - min_ens) * dt;

    % new time axis (not constant dt, use for velocities)
    vel_time = nan(length(new_ens),1);
   

    % preallocate arrays for data with nans
    amp  = nan(length(new_ens),length(r),3);
    phase = nan(length(new_ens),length(r),3);
    corr  = nan(length(new_ens),length(r),3);
    
    % Find where the original ensembles slot into the new axis, vice-versa
    [locnew,~] = ismember(new_ens,common_ensemble);
    [locold,~] = ismember(common_ensemble,new_ens);

    % Fill in data
    amp(locnew,:,:)  = amp_old(locold,:,:);
    phase(locnew,:,:) = phase_old(locold,:,:);
    corr(locnew,:,:)  = cor_old(locold,:,:);
    vel_time(locnew) = time_old(locold);



    % calculate vertical velocities from aux1_H phases
    freq = 1400*10^3; 
    tau = diff(vel_time)./10; % pulse repitition interval
    c = 1500; % speed of sound (change with temp)
    k = 2*pi*freq/c; % wavenumber
    w = phase(1:end-1,:,1)./2./k./tau; % vertical speed

    % calculate horizontal velocities from aux2 phases and w
    Vs = phase(1:end-1,:,3)./2./k./tau; % slant speed
    u = -(Vs - w.*cosd(theta/2))./sind(theta/2);

end