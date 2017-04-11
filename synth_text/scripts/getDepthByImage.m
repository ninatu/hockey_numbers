function [depths_pred depths_inpaint] = getDepthByImage(img_data, model_trained, ds_config, opts_eval)

if ~isempty(ds_config.max_img_edge)
    max_img_edge=ds_config.max_img_edge;

    img_size1=size(img_data, 1);
    img_size2=size(img_data, 2);
    
    if img_size1>img_size2
        if img_size1>max_img_edge
            img_data=imresize(img_data, [max_img_edge, NaN]);
        end
    else
        if img_size2>max_img_edge
            img_data=imresize(img_data, [NaN, max_img_edge]);
        end
    end
end


sp_info=gen_supperpixel_info(img_data, ds_config.sp_size);
pws_info=gen_feature_info_pairwise(img_data, sp_info);

ds_info=[];
ds_info.img_idxes=1;
ds_info.img_data=img_data;
ds_info.sp_info{1}=sp_info;
ds_info.pws_info=pws_info;
ds_info.sp_num_imgs=sp_info.sp_num;

depths_pred = do_model_evaluate(model_trained, ds_info, opts_eval);
depths_inpaint = do_inpainting(depths_pred, img_data, sp_info);

% normalize depths
%{
label_norm_info=model_trained.label_norm_info;
max_d=label_norm_info.max_d;
min_d=label_norm_info.min_d;
norm_type=label_norm_info.norm_type;
if  norm_type==2
    max_d=power(2, max_d);
    min_d=power(2, min_d);
elseif norm_type==3
    max_d=power(10, max_d);
    min_d=power(10, min_d);
end

if opts_eval.do_show_log_scale
    scaling_label=log10(max_d)-log10(min_d);
    offset_label=log10(min_d);
else
    scaling_label=max_d-min_d;
    offset_label=min_d;
end
%}

end
