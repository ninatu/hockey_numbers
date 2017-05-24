function estimateSegments( dsetPath, dsetOutput )

mcgPath = '../mcg/pre-trained/';
run([mcgPath 'install.m']);
addpath(mcgPath);

imgPath = '/image';
dsetInfo =  h5info(dsetPath);
groups = dsetInfo.Groups;
imgDsets = [];
for i=1:length(groups)
	curInfo = groups(i);
	if strcmp(curInfo.Name, imgPath)
		imgDsets = curInfo.Datasets;
		break
	end
end

% read imgs, compute and save segments
imgNames = {imgDsets.Name};
segPath = '/seg';

if exist(dsetOutput, 'file')==2
    delete(dsetOutput);
end

modeUCM = 'accurate';
reverseStr = '';

for imgIdx=1:length(imgNames)
    msg = sprintf('SEGMENT: processing image (%d of %d)', imgIdx, length(imgNames));
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));
	
    imgName = imgNames{imgIdx};
	imgData = h5read(dsetPath, fullfile(imgPath, imgName));
    imgData = permute(imgData, [3, 2, 1]);
    ucm = permute(im2ucm(imgData, modeUCM), [2, 1]);
    h5create(dsetOutput, char(fullfile(segPath, imgName)), size(ucm));
    h5write(dsetOutput, char(fullfile(segPath, imgName)), ucm);
end
fprintf('\nProcessing completed!\n');

end
