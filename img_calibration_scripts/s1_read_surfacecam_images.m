close all
clear all
clc
addpath(genpath('/Users/annagoldman/Desktop/RESEARCH/imgcalibration'))

%% STEP 0: Define files


% edit below for different trials. can use one or multiple fnames,
% depending how much data you want to process in one run
group               = 'intrinsic_calib';
trial               = '2025-08-25_15h11m11sUTC';
fnames              = ["2025-08-25_15h11m11s_Cam2_EO_.dat"];

for i = 1:length(fnames)
    fname = char(fnames(i))

    % build file infrastructure
    projfolder          = '/Volumes/STONE-IG-3/IG-STONE';
    figfolder           = [projfolder,'/figures/Surface_Cameras/',group,'/',trial,'/',datestr(datetime('now'),'yyyy-mm-dd'),'/'];
    camfile             = [projfolder,'/raw-data/Surface_Cameras/',group,'/',trial,'/',fname];
    mkdir(figfolder) % create a folder for figures

    %% STEP 1: Read imagery 


    % imagery parameters
    bit_header_offset   = 64;
    bit_spacer          = 64;
    bit_depth           = 8;
    width               = 2448;
    height              = 2048;
    format              = 1;
    readinParams        = [bit_header_offset bit_spacer bit_depth width height format];
    
    % read imagery
    [info,varout]       = readRawIRL(camfile,readinParams);
    
    %% STEP 2: Plot and/or save imagery
    framenums = [42, 47, 55, 64, 68, 88, 94, 101, 108, 110, 218, 221, 227, 235, 242, 270, 278, 290, 298, 376, 395];
    for i = 1:length(framenums)
        
        %{
        % [For testing] Create single plot
        
        figure('Color','w')
        imagesc(demosaic(varout(:,:,framenums(i)),'rggb')) %rggb
        colormap('gray')
        axis equal
        xlim([0 info.width])
        ylim([0 info.height])
        figfilename = "FIGimgcamtwo" + i + ".jpg";
        saveas(gcf, fullfile(figfolder,figfilename))
        %}

        %convert to color and save frames as jpg files
        rgbImg = demosaic(varout(:,:,framenums(i)),'rggb');
        filename = "imgcamtwo" + i + ".jpg";
        imwrite(rgbImg, fullfile(figfolder, filename));

    end

    %{
    % Optional features (not used in intrinsic/extrinsic calibration
    process)

    % Plot to check frame dropping
    figure('Color','w')
    scatter(1:length(info.camTime)-1,diff(info.camTime),'fill','k'); 
    ylabel('$\Delta t$ (s)'); 
    xlabel('Frame Number')
    filename_dropping = "dropping" + fname + ".png";
    saveas(gcf, fullfile(figfolder, filename_dropping))
    
    % create video
    figure('Color', 'w')
    demotype = 'rggb';
    v = VideoWriter([figfolder, fname(1:end-5),'_offset',num2str(bit_header_offset),'_spacer',num2str(bit_spacer),'_',demotype], 'MPEG-4');
    v.FrameRate = 10; %Hz
    open(v);
    % Loop through frames
    for k = 1:size(varout,3)
        imagesc(demosaic(varout(:,:,k),demotype));
        box on
        % colormap('gray')
        
        % colorbar;
        title(['Frame ' num2str(k)]);
        axis equal
        xlim([0 info.width])
        ylim([0 info.height])
        frame = getframe(gcf);
        writeVideo(v, frame);
    end
    close(v);
    
    %% STEP 3: Count dropped frames
    counts = 0;
    for i= 1:length(info.camTime)-1
    diffvar = info.camTime(i+1) - info.camTime(i);
        if diffvar > 0.15
        counts = counts + ((diffvar - 0.1)/0.1);
        end
    end
    %}
end
    