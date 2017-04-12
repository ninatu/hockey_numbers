function estimateDepths( dsetPath, dsetOutput )

dirDepthEstimator = '../dcnf-fcsp/';
installDepthEstimPath();

ds_config=[];
%settings we used for training our model:
% 1. sp_size: average superpixel size in SLIC 
% 2. max_img_edge: resize the image with the largest edge <= max_img_edge
ds_config.sp_size=20; 
ds_config.max_img_edge=640; 
%indoor scene model
trained_model_file=[dirDepthEstimator 'model_trained/model_dcnf-fcsp_NYUD2'];   

opts=[];
opts.useGpu=true;
if opts.useGpu
    if gpuDeviceCount==0
        disp('WARNNING!!!!!! no GPU found!');
        disp('any key to continue...');
        pause;
        opts.useGpu=false;
    end
end

fprintf('\nloading trained model...\n\n');
model_trained=load(trained_model_file); 
model_trained=model_trained.data_obj;

opts_eval=[];
opts_eval.useGpu = opts.useGpu;
%turn this on to show depths in log scale, better visulization for outdoor scenes
opts_eval.do_show_log_scale=false; 


% read list input image
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

% read imgs, compute and save depths
imgNames = {imgDsets.Name};
depthPath = '/depth';

if exist(dsetOutput, 'file')==2
    delete(dsetOutput);
end

reverseStr = '';
for imgIdx=1:length(imgNames)
    
    msg = sprintf('DEPTH: processing image (%d of %d)', imgIdx, length(imgNames));
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'), 1, length(msg));

	imgName = imgNames{imgIdx};
	imgData = h5read(dsetPath, fullfile(imgPath, imgName));
    imgData = permute(imgData, [3, 2, 1]);
    [depths_pred depths_inpaint] = getDepthByImage(imgData, model_trained, ds_config, opts_eval);
	depths = cat(3, depths_pred, depths_inpaint);
    h5create(dsetOutput, char(fullfile(depthPath, imgName)), size(depths));
    h5write(dsetOutput, char(fullfile(depthPath, imgName)), depths);
end
fprintf('\nProcessing completed!`\n');

end
