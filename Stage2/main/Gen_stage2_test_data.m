function Gen_stage2_test_data()

FindFiles = './Stage1_test_result/';   
Files = dir(fullfile(FindFiles));
filenames = {Files.name}';
filenames = filenames(3:length(filenames));
filenames=filenames';

 
 for i = 1:length(filenames)
    load([FindFiles,filenames{i}]);
    [num_data, num_points,~] = size(input_point_cloud);
    Training_data_stage2 = {};
    for j =1:num_data
        tic
       %%
        input_points = squeeze(input_point_cloud(j,:,:));
 
        edge_points_pre = squeeze(pred_labels_key_p_val(j,:,:));
        edge_points_pre = exp(edge_points_pre);
        sum_edge_pre = sum(edge_points_pre,2);
        edge_points_pre = edge_points_pre./repmat(sum_edge_pre,1,2);
        edgepoint_label_pre = edge_points_pre(:,2)>0.7;
        edge_label_pre_ind = find(edgepoint_label_pre);
        
        corner_points_pre = squeeze(pred_labels_corner_p_val(j,:,:));
        corner_points_pre = exp(corner_points_pre);
        sum_pre = sum(corner_points_pre,2);
        corner_points_pre = corner_points_pre./repmat(sum_pre,1,2);
        corner_label_pre = corner_points_pre(:,2)>0.9;        
        conrer_label_pre_ind = find(corner_label_pre); 


        %
        pred_corner_idx = unique(conrer_label_pre_ind);
        pred_corner = input_points(pred_corner_idx,:); 

        num_matrix = 128;
        num_pre_corner = numel(pred_corner_idx);
        if num_pre_corner < num_matrix
           num_padding = num_matrix - num_pre_corner;
           pad_num = round(1 + (num_pre_corner-1).*rand([num_padding 1]));
           pre_corner_idx_pad = pred_corner_idx(pad_num, :);
           pre_corner_point_pad = pred_corner(pad_num, :);

           pre_corner_idx_all = [pred_corner_idx; pre_corner_idx_pad];
           pre_corner_point_all = [pred_corner; pre_corner_point_pad];
        else  
           [~,temp_sample_idx] = Farthest_Point_Sampling_piont_and_idx(pred_corner,num_matrix);
           pre_corner_idx_all = pred_corner_idx(temp_sample_idx,:);
           pre_corner_point_all = pred_corner(temp_sample_idx,:);
        end

       lin_mat = ones(num_matrix,num_matrix);
       lin_mat = lin_mat - tril(ones(num_matrix,num_matrix));
       [lin_left_idx,lin_right_idx,~] = find(lin_mat);   

    %%
        rename_temp_gt.input_point_cloud_edge_corner = [input_points,edge_points_pre(:,2),corner_points_pre(:,2)];
        train_all_pair = [pre_corner_idx_all(lin_left_idx),pre_corner_idx_all(lin_right_idx)] - 1;
        rename_temp_gt.train_all_pair_rand = train_all_pair; 
        Training_data_stage2 = [Training_data_stage2; rename_temp_gt]; 

        disp('i = ')
        disp(i)
        disp('j = ')
        disp(j)        
        toc

    end
    temp_name = filenames{i};
    DST_PATH_t = './test_data_2_1/';
    save_path = [DST_PATH_t,temp_name];   
    save(save_path,'Training_data_stage2');
    
 end
end 

 
 
 
 
 
 
 
 
        
