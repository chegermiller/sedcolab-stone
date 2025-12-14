function [info,varargout] = readRawIRL(filename,varargin)
%
% readRawIRL reads selected frames in the raw 16 bit IR data image streams
%   and reshapes it into images
%
% Usage: [info,data] = readRawIRL(filename,readinParams)
%        [info,data] = readRawIRL(filename,readinParams,findex)
%		 [info,data] = readRawIRL(filename,readinParams,findex,UVList)
%               info = readRawIRL(filename,readinParams,'info')
% Input:
%   filename - filename with full path as a string
%   findex - {optional} an index of the specific frames to down load.  If omitted
%     all the frames are read in, so BEWARE as this will usually hose the memory
%     'cause of the size of most raw IR files.
%	readinParams - vector with the elements [bit_header_offset bit_spacer bit_depth width height {format}]
%     format = 0 by default, or numbered for different version labview metadata files, current options are:
%           0 - original airborne and station IR labview
%           1 - lab sampling system meta data and with big endian data (eg. Melissa/Christine's TRC system)
%           2 - lab sampling system meta data and with big endian data (eg. sedCam system)     
%   UVList are an Nx2 list of UV to sample.  Must be integers within the field of view.
%
% Output:
%   info - a structure with fields:
%     nframes - total number of frames requested, or total number of frames for
%       second usage
%     width - width of the image in pixels
%     height - height of the image in pixels
%     epoch - epoch time of each frame, according to timestamp in the raw file
%			bitdepth - pixel bit depth
%  data - a matrix (size height x width x nframes) of the IR data
%
% Example usage for IG-STONE:
%   bit_header_offset   = 64;
%   bit_spacer          = 64;
%   bit_depth           = 8;
%   width               = 2448;
%   height              = 2048;
%   format              = 1;
%   readinParams        = [bit_header_offset bit_spacer bit_depth width height format];
%   [info,varout]       = readRawIRL(camfile,readinParams);
%
% Notes on how to identify the bit_header_offset and bit_spacer: 
%   (1) read in the first image with bit_header_offset = 0 and bit_spacer = 0.
%   (2) Compute the header shift as the number of shifted columns x bits_depth. 
%   (3) Read in the first and second image in with the newly defined bit_header_offset and bit_spacer = 0.
%   (4) Plot the second frame and compute the spacer bit as number of shifted columns x bit_depth.
%
% Author: Chris Chickadel
% Last updated: Oct 2025

% hardwire image bits/pixel here, will determine real value later

% get params for reading in and making data output, like a stack
infoOnlyFlag = 0;
switch length(varargin)
	case 1
		params = varargin{1};
		findex = [];
	case 2
		params = varargin{1};
		if ischar(varargin{2})
			infoOnlyFlag = 1;
		else
			findex = varargin{2};
		end
	case 3
		params = varargin{1};
		findex = varargin{2};
		uvs = varargin{3};
		info.UV = uvs;
	otherwise
		error('readRawIRL needs at least two inputs, see help.')
end

hbit = params(1);
obit = params(2);
bit = params(3);
sbit = mod(bit,8)+bit; % save data as byte interger data
info.bitdepth = bit;
info.width = params(4);
info.height = params(5);
metaformat = 0; %default
mpformat = 'n';
if numel(params)>5
  metaformat = params(6);
  switch metaformat
    case 1
      mpformat = 'b';
    case 2
      mpformat = 'n'; 
      metaformat = 1; % need to reset to meat data format
  end  
end

% deblank filename
filename = deblank(fliplr(deblank(fliplr(filename))));

% open file and parse inputs
dirF = dir(filename);
fid = fopen(filename,'r',mpformat);
if fid<0
  error(['Cannot open ' filename '!'])
end
nFrames = floor((dirF.bytes-(hbit-obit)/8)/((info.width*info.height*info.bitdepth + obit)/8)); % calculated frame number
info.nframes = nFrames;
if ~exist('findex') | isempty(findex)
  findex = 1:nFrames;
end
info.epoch = nan(info.nframes,1); % make epoch variable bin

% read time and temp from timesFile
[junk nameTemp] = strtok(fliplr(filename),'.');  % get times 'cause we forgot
nameTemp = fliplr(nameTemp(2:end));
switch metaformat
  case 0
    %disp('format 0')
    timesFile = [nameTemp '.txt'];
    [junk nlines] = unix(['wc ' timesFile]);  % lines in time file
    nlines = str2num(strtok(nlines,' ')); % lines in time file
    dirT = dir(timesFile);
    if ~isempty(dirT)
      if infoOnlyFlag & (nlines~=nFrames)% & nlines~=(nFrames+1) % one of the files is shorter, should warn people
          info.nframes = min(nlines,nFrames);
          findex = 1:info.nframes;
          warning(sprintf('image frame number and time file count do not match'))
      end
      tfid = fopen(timesFile);
      tline = fgetl(tfid);
      sTime = sscanf(tline,'%f',1);
      frewind(tfid); % rewind file
      nT = sum(tline == ','); % assume csv
      if sTime>posixtime(datetime(2013, 04, 01, 0, 0, 0)) % after ~1 April 2013 temp probes switched
        switch nT
        case 0

        case 2
          tout = textscan(tfid,'%f%s%f','headerlines',0,'delimiter',',');
          info.epoch = tout{1};
          info.probeTemp = tout(2:end); % leave it all together and sort later
          for jp = 1:length(info.probeTemp) % just return the wanted probe temps
            info.probeTemp{jp} = info.probeTemp{jp}(findex);
          end
        case 4
          tout = textscan(tfid,'%f%s%f%s%f','headerlines',0,'delimiter',',');
          info.epoch = tout{1};
          info.probeTemp = tout(2:end); % leave it all together and sort later
          for jp = 1:length(info.probeTemp) % just return the wanted probe temps
            info.probeTemp{jp} = info.probeTemp{jp}(findex);
          end
            case 5
          tout = textscan(tfid,'%f%*s%s%f%s%f','headerlines',0,'delimiter',',');
          info.epoch = tout{1};
          info.probeTemp = tout(2:end); % leave it all together and sort later
          for jp = 1:length(info.probeTemp) % just return the wanted probe temps
            info.probeTemp{jp} = info.probeTemp{jp}(findex);
          end
        case 6
          tout = textscan(tfid,'%f%*s%s%f%s%f%s','headerlines',0,'delimiter',',');
          info.epoch = tout{1};
          info.probeTemp = tout(2:end-1); % leave it all together and sort later
          info.shutterOpen = strncmpi(tout{end},'open',4); info.shutterOpen = info.shutterOpen(findex);
          for jp = 1:length(info.probeTemp) % just return the wanted probe temps
            info.probeTemp{jp} = info.probeTemp{jp}(findex);
          end
        case 11
          try
            tout = textscan(tfid,'%f%f%f%f%f%f%f%s%s%f%s%f',inf,'headerlines',0,'delimiter',',');
            for jp = 2:length(tout) % just return the time indices of each paramter, except the epoch time
              tout{jp} = tout{jp}(findex);
            end
            info.epoch = tout{1};
            info.lat = tout{2};
            info.lon = tout{3};
            [info.N, info.E, UTMZone] = ll2UTM(info.lat,info.lon);
            info.elev = tout{4};
            info.roll = tout{5};
            info.pitch = tout{6};
            info.azmth = tout{7};
            info.INSmsg = tout{8};
            info.probeTemp = tout(9:end); % leave it all together and sort later
          catch
            frewind(tfid);
            tout = textscan(tfid,'%f%*s%*s%*s%*s%*s%*s%*s%s%f%s%f','headerlines',0,'delimiter',',');
            info.epoch = tout{1};
            info.probeTemp = tout(2:end);
          end
        case 12
          try
            tout = textscan(tfid,'%f%f%f%f%f%f%f%s%s%f%s%f%s',inf,'headerlines',0,'delimiter',',');
            for jp = 2:length(tout) % just return the time indices of each paramter, except the epoch time
              tout{jp} = tout{jp}(findex);
            end
            info.epoch = tout{1};
            info.lat = tout{2};
            info.lon = tout{3};
            [info.N, info.E, UTMZone] = ll2UTM(info.lat,info.lon);
            info.elev = tout{4};
            info.roll = tout{5};
            info.pitch = tout{6};
            info.azmth = tout{7};
            info.INSmsg = tout{8};
            info.probeTemp = tout(9:end-1); % leave it all together and sort later
            info.shutterOpen = strncmpi(tout{end},'open',4); info.shutterOpen = info.shutterOpen(findex);
          catch
            frewind(tfid);
            tout = textscan(tfid,'%f%*s%*s%*s%*s%*s%*s%*s%s%f%s%f%s','headerlines',0,'delimiter',',');
            info.epoch = tout{1};
            info.probeTemp = tout(2:end);
            info.shutterOpen = char(tout{end}); % maybe shutter cnodition is the last one
          end
        otherwise
          tout = textscan(tfid,'%f %*[^\n]','headerlines',0,'delimiter',','); % get time and skip remainder of the line
          info.epoch = [tout{:}];
        end
      else
        tout = textscan(tfid,repmat('%f',[1 nT+1]),'headerlines',0,'delimiter',',');
        info.epoch = tout{1};
        switch nT
          case 2
            info.camTemp = tout{2}(findex);
            info.refTemp = tout{3}(findex);
          case 4
            info.camTemp = tout{2}(findex);
            info.refTemp = tout{3}(findex);
            info.lensTemp = tout{2}(findex);
            info.caseTemp = tout{3}(findex);
          otherwise % assume no temp info, just time
            info.epoch = [tout{:}];
        end
      end
      if length(info.epoch) < nFrames% found a file with fewer times than frames, guess first frames are okay and add more, not the best solution
        %dt = abs(mean(diff(info.epoch)));
        %info.epoch = [info.epoch; cumsum([1:(nFrames - length(info.epoch))]'*dt)+info.epoch(end)];
      end
      fclose(tfid);
    else % oops, txt file there just output nans
      info.epoch = nan*[1:nFrames]';
      info.camTemp = nan*[1:nFrames]';
      info.refTemp = nan*[1:nFrames]';
    end
  case 1 % Melissa's ior STONE camera system
    %disp('format 1')
    dirT = dir([nameTemp '*meta*']); % look for metadata file
    timesFile = [dirT.folder filesep dirT.name];
    if ~isempty(dirT)       
      [junk nlines] = unix(['wc ' timesFile]);  % lines in time file
      nlines = str2num(strtok(nlines,' '))-2; % lines in time file
      if infoOnlyFlag & (nlines~=nFrames)% & nlines~=(nFrames+1) % one of the files is shorter, should warn people
          info.nframes = min(nlines,nFrames);
          findex = 1:info.nframes;
          warning(sprintf('image frame number, %d, and time file count, %d, do not match',info.nframes,nlines))
      end
      tfid = fopen(timesFile);
      tout = textscan(tfid,'%d %f %f %f %s','headerlines',2);
      fclose(tfid);
      info.frameID = tout{1}(findex);
      info.expTime = tout{2}(findex);
      info.gain = tout{3}(findex);
      info.camTime = tout{4}(findex);
      info.epoch = posixtime(datetime(datenum(tout{5}), "ConvertFrom", "datenum"));
    else % no file, fill in nans
      info.frameID = nan*findex(:);
      info.expTime = nan*findex(:);
      info.gain = nan*findex(:);
      info.camTime = nan*findex(:);
      info.epoch = nan*findex(:);
    end
end

% exit on infoFlag
if infoOnlyFlag
  fclose(fid); % close file
  info.epoch = info.epoch(findex);
  return % get out 'cause all you wanted was the info
end

% read in image data
Nf = length(findex);
info.nframes = Nf;
if ~exist('uvs')
	data = zeros([info.height info.width Nf],sprintf('uint%d',sbit));
	for j = 1:Nf
		fseek(fid,(info.width*info.height*(bit/8)+obit/8)*(findex(j)-1)+hbit/8,'bof');
		dataT = fread(fid,info.width*info.height,sprintf('ubit%d=>uint%d',bit,sbit)); % separate for profile
		data(:,:,j) = reshape(dataT,[info.width info.height])';
		if Nf > 1
			fprintf(1,'  %d of %d\r',j,Nf)
		end
	end
else % get specific uv's, if so requested
	if numel(uvs) == 4 % make sure to take care of the one pixel sampling scheme. Dumb, I know
		uvsT = unique(uvs,'rows');
		if size(uvsT,1) ~= size(uvs,1)
			uvs = uvsT;
		end
	end
	uvsInd = sub2ind([info.width info.height],uvs(:,1),uvs(:,2));
	data = zeros([Nf size(uvs,1)],sprintf('uint%d',sbit));
	for j = 1:Nf
		fseek(fid,(info.width*info.height*(bit/8)+obit/8)*(findex(j)-1)+hbit/8,'bof');
		dataT = fread(fid,info.width*info.height,sprintf('ubit%d=>uint%d',bit,sbit)); % separate for profile
		data(j,:) = dataT(uvsInd(:)');
		if Nf > 1
			fprintf(1,'  %d of %d\r',j,Nf)
		end
	end
end
fclose(fid); % close file
info.nframes = length(findex);
info.epoch = info.epoch(findex);
varargout{1} = data;


