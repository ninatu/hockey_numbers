function installDepthEstimPath()

dirDepthEstimator = '../dcnf-fcsp/';
dirVlfeatToolbox = '../dcnf-fcsp/libs/vlfeat-0.9.18/toolbox/';
dirMatConvNet='../dcnf-fcsp/libs/matconvnet_20141015/matlab/';

run([dirVlfeatToolbox 'vl_setup.m']);
addpath(genpath(dirMatConvNet));
run([dirMatConvNet 'vl_setupnn.m']);
addpath([dirDepthEstimator 'demo'])
end
