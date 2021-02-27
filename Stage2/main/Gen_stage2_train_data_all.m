function Gen_stage2_train_data_all()
 
FindFiles = './Stage1_network_test_result/';   % Stage 1 test_result path.
FindFiles_test_data =  './Stage1_network_train_data/';  % Stage1 train_data path.  
Files = dir(fullfile(FindFiles_test_data));
filenames = {Files.name}';
filenames = filenames(3:length(filenames));
filenames=filenames';
 
 for i = 1:length(filenames)
    load([FindFiles,'test_pred_',filenames{i}]);
    load([FindFiles_test_data,filenames{i}]);
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
        corner_label_pre = corner_points_pre(:,2)>0.85;        
        conrer_label_pre_ind = find(corner_label_pre);        

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

       training_data_temp = Training_data{j};

       open_gt_pair_idx = training_data_temp.open_gt_pair_idx;
       open_gt_type = training_data_temp.open_gt_type;
       open_gt_res = training_data_temp.open_gt_res;
       open_gt_type_pos_idx = find(open_gt_type>0);
       open_gt_type_pos_type = open_gt_type(open_gt_type_pos_idx,:);
       open_gt_pair_pos_idx = open_gt_pair_idx(open_gt_type_pos_idx,:);
       open_gt_pos_res = open_gt_res(open_gt_type_pos_idx,:);

       gt_conrer_left_idx = open_gt_pair_pos_idx(:,1) + 1;
       gt_left_conrer = input_points(gt_conrer_left_idx,:) + open_gt_pos_res(:,1:3);

       gt_conrer_right_idx = open_gt_pair_pos_idx(:,2) + 1;
       gt_right_conrer = input_points(gt_conrer_right_idx,:) + open_gt_pos_res(:,1:3);

       gt_conrer_idx = [gt_conrer_left_idx; gt_conrer_right_idx];
       gt_conrer = [gt_left_conrer; gt_right_conrer];

       [gt_corner_idx_uni, idx_gt_corner_idx_uni] = unique(gt_conrer_idx);
       gt_conrer_uni = gt_conrer(idx_gt_corner_idx_uni,:);

       gt_corner_left_dis = Distance_Points1_Points2(gt_left_conrer,gt_conrer_uni);
       gt_corner_right_dis = Distance_Points1_Points2(gt_right_conrer,gt_conrer_uni);

       %
       [dis_left_value,~] = sort(gt_corner_left_dis,2);
       min_left_value = dis_left_value(:,2); 
       thr_left_pos = 0.25*min_left_value; 
       thr_left_neg = 0.25*min_left_value;

       %
       [dis_right_value,~] = sort(gt_corner_right_dis,2);
       min_right_value = dis_right_value(:,2); 
       thr_right_pos = 0.25*min_right_value; 
       thr_right_neg = 0.25*min_right_value;    

       thr_pos_all = [thr_left_pos; thr_right_pos];
       thr_pos_rep = repmat(thr_pos_all,1,size(pre_corner_point_all,1));

       dis_gt_pred = Distance_Points1_Points2(gt_conrer,pre_corner_point_all);
       diff_dis_thr = dis_gt_pred - thr_pos_rep;
       sel_pos_sample = diff_dis_thr;
       sel_pos_sample(sel_pos_sample < 0) = -1;
       sel_pos_sample(sel_pos_sample >= 0) = 0;
       sel_pos_sample(sel_pos_sample == -1) = 1;

       sel_pos_sample_cell_all = mat2cell(sel_pos_sample,ones(size(sel_pos_sample,1),1),size(sel_pos_sample,2));
       sel_pos_sample_idx_cell = cellfun(@(x) find(x==1), sel_pos_sample_cell_all,'Unif',0);

       sel_pos_sample_idx_cell_uni = sel_pos_sample_idx_cell(idx_gt_corner_idx_uni,:); 
       sel_pos_sample_idx_cell_uni = cellfun(@(x) Judge(x),sel_pos_sample_idx_cell_uni,'Unif',0); 
       sel_pos_sample_idx_cell_uni(cellfun(@isempty,sel_pos_sample_idx_cell_uni))=[];
       idx_cell_uni_nchoose = cellfun(@(x) nchoosek(x,2), sel_pos_sample_idx_cell_uni,'Unif',0);
       idx_cell_uni_nchoose_mat = cell2mat(idx_cell_uni_nchoose);
       idx_cell_uni_nchoose_wei = ones(size(idx_cell_uni_nchoose_mat,1),1);
       not_mat = Graph_to_matrix(idx_cell_uni_nchoose_mat, idx_cell_uni_nchoose_wei, num_matrix);

       not_mat = not_mat.*triu(ones(num_matrix,num_matrix));
       not_mat = not_mat + triu(ones(num_matrix,num_matrix));
       not_mat(logical(eye(size(not_mat))))=0 ;
       [not_left_idx,not_right_idx,not_label] = find(not_mat);   
       not_label = not_label - 1;


       num_left = size(gt_left_conrer,1);
       left_pos_idx_cell = sel_pos_sample_idx_cell(1:num_left);
       right_pos_idx_cell = sel_pos_sample_idx_cell(num_left+1:end);   
       Free_combine_con_cell = cellfun(@(x,y) Free_combine(x,y), left_pos_idx_cell, right_pos_idx_cell,'Unif',0);
       Free_combine_con_cell(cellfun(@isempty,Free_combine_con_cell))=[];
       Free_combine_con_cell_mat = cell2mat(Free_combine_con_cell);
       Free_combine_con_cell_mat_wei = ones(size(Free_combine_con_cell_mat,1),1);
       con_mat = Graph_to_matrix(Free_combine_con_cell_mat, Free_combine_con_cell_mat_wei, num_matrix);
       con_mat = con_mat.*triu(ones(num_matrix,num_matrix));
       con_mat = con_mat + triu(ones(num_matrix,num_matrix));
       con_mat(logical(eye(size(con_mat))))=0 ;
       [con_left_idx,con_right_idx,con_label] = find(con_mat);   
       con_label = con_label - 1;

       corner_pair_cyc = open_gt_type_pos_type;
       corner_pair_cyc(corner_pair_cyc==3) = 0;
       corner_pair_cyc(corner_pair_cyc>0) = 1;
       corner_pair_cyc = num2cell(corner_pair_cyc);   

       Free_combine_cyc_cell = cellfun(@(x,y,z) Free_combine_add_label(x,y,z), left_pos_idx_cell, right_pos_idx_cell,corner_pair_cyc,'Unif',0);
       Free_combine_cyc_cell(cellfun(@isempty,Free_combine_cyc_cell))=[];
       Free_combine_cyc_cell_mat = cell2mat(Free_combine_cyc_cell);
       if isempty(Free_combine_cyc_cell_mat)
          cyc_mat = Graph_to_matrix(Free_combine_cyc_cell_mat, Free_combine_cyc_cell_mat, num_matrix);
       else
          cyc_mat = Graph_to_matrix(Free_combine_cyc_cell_mat(:,1:2), Free_combine_cyc_cell_mat(:,3), num_matrix);
       end
       cyc_mat = cyc_mat.*triu(ones(num_matrix,num_matrix));
       cyc_mat = cyc_mat + triu(ones(num_matrix,num_matrix));
       cyc_mat(logical(eye(size(cyc_mat))))=0 ;
       [cyc_left_idx,cyc_right_idx,cyc_label] = find(cyc_mat);   
       cyc_label = cyc_label - 1;

       corner_pair_lin = open_gt_type_pos_type;
       corner_pair_lin(corner_pair_lin<3) = 0;
       corner_pair_lin(corner_pair_lin==3) = 1;
       corner_pair_lin = num2cell(corner_pair_lin);

       Free_combine_lin_cell = cellfun(@(x,y,z) Free_combine_add_label(x,y,z), left_pos_idx_cell, right_pos_idx_cell,corner_pair_lin,'Unif',0);
       Free_combine_lin_cell(cellfun(@isempty,Free_combine_lin_cell))=[];
       Free_combine_lin_cell_mat = cell2mat(Free_combine_lin_cell);
       if isempty(Free_combine_lin_cell_mat)
          lin_mat = Graph_to_matrix(Free_combine_lin_cell_mat, Free_combine_lin_cell_mat, num_matrix);
       else
          lin_mat = Graph_to_matrix(Free_combine_lin_cell_mat(:,1:2), Free_combine_lin_cell_mat(:,3), num_matrix);
       end


       lin_mat = lin_mat.*triu(ones(num_matrix,num_matrix));
       lin_mat = lin_mat + triu(ones(num_matrix,num_matrix));
       lin_mat(logical(eye(size(lin_mat))))=0 ;
       [lin_left_idx,lin_right_idx,lin_label] = find(lin_mat);   
       lin_label = lin_label - 1;


        training_data_temp = Training_data{j};
        closed_gt_type = training_data_temp.closed_gt_type;
        closed_gt_type_pos = find(closed_gt_type>0);
        closed_gt_sample_points = training_data_temp.closed_gt_sample_points;
        closed_gt_sample_points_pos = closed_gt_sample_points(closed_gt_type_pos,:,:); 
        num_gt_pos = size(closed_gt_sample_points_pos,1);

        gt_closed_curve_points_all = zeros(num_gt_pos*64,3);
        for x = 1:num_gt_pos
           gt_closed_curve_points_all((x-1)*64+1:x*64,:) = squeeze(closed_gt_sample_points_pos(x,:,:));   
        end
        gt_closed_curve_points_all_uni = unique(gt_closed_curve_points_all,'rows');

        global_closed_edge_idx = Distance_Points1_Points2_with_num(gt_closed_curve_points_all_uni,input_points,1); 
        Closed_label = zeros(size(input_points,1),1);
        Closed_label(global_closed_edge_idx) = 1;

        temp_points = input_points;
        All_edge_points_idx = training_data_temp.PC_8096_edge_points_label_bin;
        All_edge_points_idx = find(All_edge_points_idx == 1);

        %
        open_gt_sample_points = training_data_temp.open_gt_sample_points;
        open_gt_type = training_data_temp.open_gt_type;
        open_gt_type_pos_idx = find(open_gt_type>0);
        open_gt_sample_points_pos = open_gt_sample_points(open_gt_type_pos_idx,:,:);

        num_open_curves = size(open_gt_sample_points_pos,1);
        gt_open_curve_points_all = zeros(num_open_curves*64,3);
        for x = 1:num_open_curves
           gt_open_curve_points_all((x-1)*64+1:x*64,:) = squeeze(open_gt_sample_points_pos(x,:,:));   
        end
        gt_open_curve_points_all_uni = unique(gt_open_curve_points_all,'rows');
        global_open_edge_idx = Distance_Points1_Points2_with_num(gt_open_curve_points_all_uni,input_points,1);

        Open_label  = zeros(size(input_points,1),1);
        Open_label(global_open_edge_idx) = 1; 

        open_lin_gt_sample_points = training_data_temp.open_gt_sample_points;
        open_lin_gt_type = training_data_temp.open_gt_type;
        open_lin_gt_type_pos_idx = find(open_lin_gt_type == 3);
        open_lin_gt_sample_points_pos = open_lin_gt_sample_points(open_lin_gt_type_pos_idx,:,:);

        num_open_lin_curves = size(open_lin_gt_sample_points_pos,1);
        gt_open_lin_curve_points_all = zeros(num_open_lin_curves*64,3);
        for x = 1:num_open_lin_curves
           gt_open_lin_curve_points_all((x-1)*64+1:x*64,:) = squeeze(open_lin_gt_sample_points_pos(x,:,:));   
        end
        gt_open_lin_curve_points_all_uni = unique(gt_open_lin_curve_points_all,'rows');
        global_open_lin_edge_idx = Distance_Points1_Points2_with_num(gt_open_lin_curve_points_all_uni,input_points,1);

        Open_lin_label  = zeros(size(input_points,1),1);
        Open_lin_label(global_open_lin_edge_idx) = 1; 

        Open_cyc_label = Open_label - Open_lin_label;


        rename_temp_gt.input_point_cloud_edge_corner = [input_points,edge_points_pre(:,2),corner_points_pre(:,2)];
        train_all_pair = [pre_corner_idx_all(lin_left_idx),pre_corner_idx_all(lin_right_idx)] - 1;
        train_all_pair_con = con_label;
        train_all_pair_not = not_label;
        train_all_pair_lin = lin_label;
        train_all_pair_cyc = cyc_label;

         rename_temp_gt.train_all_pair_rand = train_all_pair; 
         rename_temp_gt.train_all_pair_con_rand = train_all_pair_con;
         rename_temp_gt.train_all_pair_not_rand = train_all_pair_not;
         rename_temp_gt.train_all_pair_lin_rand = train_all_pair_lin;
         rename_temp_gt.train_all_pair_cyc_rand = train_all_pair_cyc; 
         rename_temp_gt.closed_label = Closed_label;
         rename_temp_gt.open_label = Open_label;
         rename_temp_gt.oli_label = Open_lin_label;
         rename_temp_gt.ocy_label = Open_cyc_label;

        Training_data_stage2 = [Training_data_stage2; rename_temp_gt]; 

        disp('i = ')
        disp(i)
        disp('j = ')
        disp(j)        
        toc  
    
    end
    temp_name = num2str(i);
    DST_PATH_t = './train_data_2_1/';
    save_path = [DST_PATH_t,temp_name,'.mat'];   
    save(save_path,'Training_data_stage2');
    
 end
end 


function All_combine = Free_combine(idx_set_1, idx_set_2)

   num_set_1 = numel(idx_set_1);
   num_set_2 = numel(idx_set_2);
   
   if (num_set_1 ==0 || num_set_2 ==0)
      All_combine = [];
   else
       set1_mat = repmat(idx_set_1', numel(idx_set_2),1);
       set2_mat = repmat(idx_set_2', 1,numel(idx_set_1));
       [f_x,f_y]=size(set2_mat);
       set2_mat_vector = reshape(set2_mat',1,f_x*f_y);
       set2_mat_vector = set2_mat_vector';
       All_combine = [set1_mat,set2_mat_vector];
   end
end

function All_combine_with_label = Free_combine_add_label(idx_set_1, idx_set_2,label)

   num_set_1 = numel(idx_set_1);
   num_set_2 = numel(idx_set_2);
   
   if (num_set_1 ==0 || num_set_2 ==0)
      All_combine_with_label = [];
   else
       set1_mat = repmat(idx_set_1', numel(idx_set_2),1); 
       set2_mat = repmat(idx_set_2', 1,numel(idx_set_1));
       [f_x,f_y]=size(set2_mat);
       set2_mat_vector = reshape(set2_mat',1,f_x*f_y); 
       set2_mat_vector = set2_mat_vector'; 
       All_combine = [set1_mat,set2_mat_vector];
       All_label = label*ones(size(All_combine,1),1);
       All_combine_with_label = [All_combine,All_label];
   end
end

function adjMatrix = Graph_to_matrix(edges, weights, num_matrix)

if isempty(edges)
    adjMatrix = zeros(num_matrix,num_matrix);
else
    adjMatrix = full(sparse(edges(:,1),edges(:,2),weights ,num_matrix,num_matrix)); 
    adjMatrix = adjMatrix + adjMatrix';
    adjMatrix(adjMatrix>0) = 1;  
end
end











function global_path_idx_cell_all = search_path(input_points, all_train_pairs, all_edge_label_pre_ind)

        pairs_points_left = input_points(all_train_pairs(:,1),:);
        pairs_points_right = input_points(all_train_pairs(:,2),:);
        
        [~,pair_left_local_idx] = ismember(all_train_pairs(:,1),all_edge_label_pre_ind);
        [~,pair_righ_local_idx] = ismember(all_train_pairs(:,2),all_edge_label_pre_ind);
        
        pred_edge_points = input_points(all_edge_label_pre_ind,:);
        
        num_neibor = 8;
        Weight_matrix = Distance_Points1_Points2_matrix(pred_edge_points, pred_edge_points);
        [~,idx_wei_mat_1] = sort(Weight_matrix,2);
        idx_wei_mat = idx_wei_mat_1(:,2:num_neibor);
        Weight_matrix_mask = zeros(size(pred_edge_points,1),size(pred_edge_points,1));
        for row_wei_mat = 1:size(pred_edge_points,1)
            temp_mask = zeros(1,size(pred_edge_points,1));
            temp_mask(idx_wei_mat(row_wei_mat,:)) = 1;
            Weight_matrix_mask(row_wei_mat,:) = temp_mask;
        end
        Weight_matrix = Weight_matrix.* Weight_matrix_mask;
        Weight_matrix = Weight_matrix + Weight_matrix';
        
%         num_edge_points = size(pred_edge_points,1);
%         idx_all = repmat([1:num_edge_points],num_edge_points,1);
%         left_idx = reshape(idx_all,size(idx_all,1)*size(idx_all,2),1);
%         right_idx = reshape(idx_all',size(idx_all,1)*size(idx_all,2),1);
%         weights = reshape(Weight_matrix,size(idx_all,1)*size(idx_all,2),1);
%         G = graph(A)
%         G.Edges
%         plot(G)
        G = graph(Weight_matrix);
        source = pair_left_local_idx;
        target = pair_righ_local_idx;
        
        global_path_idx_cell_all = cell(numel(source),1);
        for i_source = 1:numel(source)
          [TR,D] = shortestpathtree(G,source(i_source),target(i_source),'OutputForm','cell');  
          global_path_idx = all_edge_label_pre_ind(TR{1});
          global_path_idx_cell_all{i_source} = [global_path_idx;source(i_source);target(i_source)];
        end
        
end



function vector = Judge(vector)
  if numel(vector)==1
      vector=[];
  end
end

function Dis_matrix = Distance_Points1_Points2_matrix(vertices_ball,vertices_points)

B=vertices_ball; % ball_num*3;
P=vertices_points; % points_num*3;
B1=sum(B.^2,2);
P1=sum(P.^2,2);
Num_b=numel(B1);
Num_p=numel(P1);
Dis_matrix= repmat(B1,1,Num_p) + repmat(P1',Num_b,1) - 2*B*P';  % Num_b * Num_P distance matrix;

end


% ����ͶӰ�в��ÿ�� curve proposal �� edge points �е�����ڣ�֮������ƥ��ȣ���Haff distance;
% 1, Ϊcurve proposal�е�ÿ�� sample points ��һ�������?��in pred_edge_points��
% 2, ͬʱ���㣬ÿ��sample point �� edge neibor֮���?residual distance;
% 3, �ҳ� residual distance �е� max value���ɣ�
% ��Ϊÿ�� curve proposal �� max residual distance;
function project_score = compute_project_residual(proposal_sample_points,pred_edge_sample_points)
% input: proposal_sample_points: 33*100;  =�� 33�� 100*3����
% input: pred_edge_sample_points: N(>0.75)*3
% output: 33*1 max residual distance;

% 1, for each sample point  find a neibor from pred_edge_sample_points
% proposals_sample_points 256 cell: 101*3
for i = 1:numel(proposals_sample_points)
   Dis_matrix = Distance_Points1_Points2(vertices_ball,vertices_points);
   dist_vector = min(dist_matrix,[],2);
   dist = max(dist_vector);
end

% 2, 

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--------Open Pridiction Module---------%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for each corner pair selected one bet fitting curve from (Pre_line, bspline, cycle)
function temp_points_pre_color_1 = Curve_selection_for_each_corner_pair(temp_points_edge_mask, curve_type)
    load color
    if curve_type == 1
        % prediction cycle vis
        %Points_cycle_arc = Cycle_pre_vis_three_points_1(temp_points_edge_mask);
        Points_cycle_arc = arcPlot(temp_points_edge_mask);
        
        temp_points_pre = Points_cycle_arc;
        temp_points_pre_color_label = 10*ones(size(temp_points_pre,1),1);
        temp_points_pre_color = color(temp_points_pre_color_label,:);
        temp_points_pre_color_1 = [temp_points_pre,temp_points_pre_color];
        plot3(temp_points_pre_color_1(:,1),temp_points_pre_color_1(:,2),temp_points_pre_color_1(:,3),'r-')
    elseif curve_type == 2
        % prediction bspline vis
        Points_b_spline = spline_vis(temp_points_edge_mask);
        Points_b_spline = Points_b_spline';
        temp_points_pre_color_label = 10*ones(size(Points_b_spline,1),1);
        temp_points_pre_color = color(temp_points_pre_color_label,:);
        temp_points_pre_color_1 = [Points_b_spline,temp_points_pre_color];
        plot3(temp_points_pre_color_1(:,1),temp_points_pre_color_1(:,2),temp_points_pre_color_1(:,3),'g-')
    elseif curve_type == 3
        % prediction line vis
        Points_line = line_vis(temp_points_edge_mask);
        temp_points_pre = Points_line;
        temp_points_pre_color_label = 10*ones(size(temp_points_pre,1),1);
        temp_points_pre_color = color(temp_points_pre_color_label,:);
        temp_points_pre_color_1 = [temp_points_pre,temp_points_pre_color];
        plot3(temp_points_pre_color_1(:,1),temp_points_pre_color_1(:,2),temp_points_pre_color_1(:,3),'b-')
    end
    

end



function vis_curve_gt_and_best_proposal(all_input_points_cloud, train_sample, All_proposals_sample_points,mask_4_max_res_val, All_corner_pair_idx)
    
    load color
%% fig3: prediction edge points  and  corner points

    
    
    
%% fig 1: gt curve 
    fig_1 = figure(1);

%     temp_points = all_input_points_cloud;
%     num_points = size(temp_points,1);
%     temp_color = color(1,:);  
%     temp_color = repmat(temp_color,num_points,1);
%     temp_all = [temp_points,temp_color];
%     scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.'); 
%     
%     hold on
    % 3.1��gt_curves
   open_gt_sample_points = train_sample.open_gt_sample_points;
   open_gt_256_64_idx = train_sample.open_gt_256_64_idx;
   open_gt_mask = train_sample.open_gt_mask;
   open_gt_type = train_sample.open_gt_type;
   open_gt_type_pos_idx = find(open_gt_type>0);
   open_gt_sample_points_pos = open_gt_sample_points(open_gt_type_pos_idx,:,:);

%    open_gt_256_64_idx_pos = open_gt_256_64_idx(open_gt_type_pos_idx,:) + 1;
%    open_gt_mask_pos = open_gt_mask(open_gt_type_pos_idx,:);   
   num_gt = size(open_gt_sample_points_pos,1);
    
    % ���ӻ� gt_curve;
    for i = 1:num_gt
    gt_curve_points = squeeze(open_gt_sample_points_pos(i,:,:));
    gt_curve_points_color = color(2,:); 
    num_points_gt_curve = size(gt_curve_points,1);
    gt_curve_points_color = repmat(gt_curve_points_color,num_points_gt_curve,1);
    gt_curve_all = [gt_curve_points,gt_curve_points_color];
    scatter3(gt_curve_all(:,1),gt_curve_all(:,2),gt_curve_all(:,3),50,gt_curve_all(:,4:6),'.');   
    hold on    
    end
    
    axis equal
    axis off
    title('Input Point Cloud & GT curve points');
    hold off
    
    
%% fig 2: best curve proposal 
    fig_2 = figure(2);

%     temp_points = all_input_points_cloud;
%     temp_color = color(1,:); 
%     num_points = size(temp_points,1);
%     temp_color = repmat(temp_color,num_points,1);
%     temp_all = [temp_points,temp_color];
%     scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.'); 
%     
%     hold on
    % 3.1��proposals_curves
%     mask_3_max_res_val, 
%     mask_4_max_res_val
%     All_proposals_sample_points
    % ���ӻ� proposal_curve;
    %open_gt_type_pos_idx
    for i = 1:num_gt
        pre_pair_temp = All_corner_pair_idx(i,:);
        if pre_pair_temp(1) ~= pre_pair_temp(2)
            pro_curve_points = All_proposals_sample_points{i};  % 33*101,3 points;
            scores = mask_4_max_res_val((i-1)*33+1:i*33);
            if open_gt_type(open_gt_type_pos_idx(i)) == 3   % if curve_type is line
                %best_score = scores(1);
                best_idx = 1;
            else
                [~,best_idx] = min(scores(2:end));  % other curve_types
                best_idx = best_idx + 1;
            end
            best_pro_curve_points = pro_curve_points((best_idx-1)*101+1:best_idx*101,:);
            
            pro_curve_points_color = color(2,:);  
            pro_curve_points_color = repmat(pro_curve_points_color,101,1);
            pro_curve_all = [best_pro_curve_points, pro_curve_points_color];
            scatter3(pro_curve_all(:,1),pro_curve_all(:,2),pro_curve_all(:,3),50,pro_curve_all(:,4:6),'.');   
            hold on 
        end
    end
    
    axis equal
    axis off
    title('Input Point Cloud & prediction curve points');
    hold off    


end








% 1), ����open_gt_sample_points:  ����ÿ��curve�Ĳ����?
%open_gt_valid_mask: �����?pos+neg=1;   padding=0;
%open_gt_256_64_idx: ÿ����  �� 8096�ϵĲ����?
%open_gt_mask:   open_gt_sample_points ÿ������  open_gt_256_64_idx�е� ����ڣ�?
% �� gt_curve_sample_points_mask (ֻȡpositive gt curve); 
% 2), ����mask_3 (curve proposal project all input points cloud), ȡ 128/4 *
% 33 * 8096
% �� curve_proposal_mask;
% 3), �� gt_curve_sample_points_mask and curve_proposal_mask IOU;
% 3.1, mask_gt_proposal = gt_curve_sample_points_mask + curve_proposal_mask;
% 3.2, mask_gt_proposal_union = mask_gt_proposal;  mask_gt_proposal_union>0 = 1;
% 3.3, mask_gt_proposal_intersction = mask_gt_proposal; 
% 3.3.1, mask_gt_proposal_intersction < 2 = 0;
% 3.3.1, mask_gt_proposal_intersction == 2 = 1;
% 3.4,  IOU = sum(mask_gt_proposal_intersction)./ sum(mask_gt_proposal_union);
% match_score = IOU;
function dist = compute_score(proposals_sample_points,train_sample)
% input: proposal_sample_points: 33*100;  =�� 33�� 100*3����
% input: open_gt_sample_points: Num*Points  =��Num�� Points*3����
% output: 33*1 scores
% score = hausdorff(A, B)
%    open_gt_pair_idx = Training_data{j}.open_gt_pair_idx;
%    open_gt_type = Training_data{j}.open_gt_type;
%    open_gt_res = Training_data{j}.open_gt_res;
%    open_gt_type_pos_idx = open_gt_type>0;
%    open_gt_pair_pos_idx = open_gt_pair_idx(open_gt_type_pos_idx,:);
   
   % 3.1��gt_curves
   open_gt_sample_points = train_sample.open_gt_sample_points;
   open_gt_256_64_idx = train_sample.open_gt_256_64_idx;
   open_gt_mask = train_sample.open_gt_mask;
   open_gt_type = train_sample.open_gt_type;
   open_gt_type_pos_idx = open_gt_type>0;
   open_gt_sample_points_pos = open_gt_sample_points(open_gt_type_pos_idx,:,:);
   open_gt_256_64_idx_pos = open_gt_256_64_idx(open_gt_type_pos_idx,:) + 1;
   open_gt_mask_pos = open_gt_mask(open_gt_type_pos_idx,:);   
   num_gt = size(open_gt_sample_points_pos,1);
   
%   % 3.2) all_curve_proposal_sample_points;
%   num_pos = size(All_proposals_sample_points,1)/4;
%   all_proposal_pos = All_proposals_sample_points(1:num_pos); % extract positive sample;
  
   
% for i = 1:num_pos
%     each_pair_curve_proposals = all_proposal_pos{i}; %33*101,3 points
    for j = 1:33
       each_curve_proposal = proposals_sample_points((j-1)*101+1:j*101,:);
       for l = 1:num_gt
           gt_sample_points = squeeze(open_gt_sample_points_pos(l,:,:));
           dist = hausdorff(each_curve_proposal,gt_sample_points);
           dist = sqrt(dist);
       end
       %max
    end 
% end


end


function all_curve_sample_points = sample_para(para_matrix,Input_point_cloud_8096)
% input: para_matrix: 32*3; 32 curve proposals;  3 point parameters
% input: num: number of sample points
% output: 32+1 curves points set: [33,101]  % ��������⣬Ӧ��?3��101��3�Ŷԣ�

num = 100; 
% 1, line_sample
para_points_idx = para_matrix(1,1:2);
para_points = Input_point_cloud_8096(para_points_idx,:);
line_sample_points = line_vis(para_points,num);

% 2, curve_sample
curve_sample_points = [];
for i= 1:size(para_matrix,1)
  para_curve_points_idx = para_matrix(i,:);
  para_curve_points = Input_point_cloud_8096(para_curve_points_idx,:);
  temp_sample_points = arcPlot(para_curve_points,num);
  curve_sample_points = [curve_sample_points; temp_sample_points];
end
all_curve_sample_points = [line_sample_points; curve_sample_points];  % [33*101, 3]
end


function [mask_1, mask_2, mask_3, mask_4, mask_3_max_res_val, mask_4_max_res_val] = Open_curve_sample_points_mask(Input_point_cloud_8096, All_corner_pair_idx, All_edge_point_idx, All_proposals_sample_points)
% input: 
%   temp_training_data: include 8096 points; temp_training_data.PC_8096_edge_points_label_bin
%   corner_pair_idx: all corner pair index;
%   num_sample: for each pair corner, sampling how many points;

%1, extract pos, neg, remain corner pair;
%     pos_corner_idx_pairs = All_corner_pair_idx.pos_corner_idx_pairs;
%     neg_corner_idx_pairs = All_corner_pair_idx.neg_corner_idx_pairs; 
%     pad_corner_idx_pairs = All_corner_pair_idx.pad_corner_idx_pairs;
%     pos_neg_pad_idx = [pos_corner_idx_pairs;neg_corner_idx_pairs;pad_corner_idx_pairs];
    pos_neg_idx = All_corner_pair_idx;
    
%2, generate all center; and radius;
    all_pair_start_point = Input_point_cloud_8096(pos_neg_idx(:,1),:);
    all_pair_end_point   = Input_point_cloud_8096(pos_neg_idx(:,2),:);
    all_pair_center = all_pair_start_point + (all_pair_end_point - all_pair_start_point)./2;
    center_start_vector = all_pair_center - all_pair_start_point;
    all_pair_radius = sum(center_start_vector.*center_start_vector,2)*(1 + 0.14);
    
%3.1, mask1: samlpe points from all_input_points_cloud(8096) in radius ball for each pair
    radius = all_pair_radius;
    xyz_input_all = Input_point_cloud_8096;
    xyz_query = all_pair_center;
    xyz_input = xyz_input_all;
    
    Dis_matrix_1 = Distance_Points1_Points2_matrix(xyz_query,xyz_input); 
    Thres_r2 = repmat(radius,1,size(xyz_input,1));
    Thres_r2 = Thres_r2 + 1e-6;
    Dis_matrix = Thres_r2 - Dis_matrix_1;
    Dis_matrix(Dis_matrix>=0) = 1; 
    Dis_matrix(Dis_matrix<0) = 0; 
    mask_1 = Dis_matrix;
    
%3.2, mask2: samlpe points in radius ball for each pair from pred_edge_points   
    radius = all_pair_radius;
    edge_points_idx = All_edge_point_idx;
    xyz_input_all = Input_point_cloud_8096;
    index = edge_points_idx';
    xyz_input = xyz_input_all(index,:);
    xyz_query = all_pair_center;
    
    Dis_matrix_1 = Distance_Points1_Points2_matrix(xyz_query,xyz_input); 
    Thres_r2 = repmat(radius,1,size(xyz_input,1));
    Thres_r2 = Thres_r2 + 1e-6;
    Dis_matrix = Thres_r2 - Dis_matrix_1;
    Dis_matrix(Dis_matrix>=0) = 1; 
    Dis_matrix(Dis_matrix<0) = 0; 
    
    num_pairs = size(xyz_query,1);
    mask_2 = zeros(num_pairs,8096);
   
   % for each corner pair to sample points(FPS) 
   for i = 1:size(xyz_query,1)
       % generate mask
       ball_points_mask = Dis_matrix(i,:);
       
       % just extract dis< radius points from 8096 points;
       ball_all_points_idx = ball_points_mask>0;  % This idx is global related to 8096;
      
      % transform local idx into global index;
      global_index = index(ball_all_points_idx);
      temp_mask_2_vector = zeros(1,8096);
     
      temp_mask_2_vector(global_index) = 1;     
      mask_2(i,:) = temp_mask_2_vector;
   end

% 3.3 mask3: each curve proposal  project    all_input_points_cloud    to
% generate mask
    xyz_input_all = Input_point_cloud_8096; 
    %All_proposals_sample_points;  % 256 cell [33*101,3]  33 curve proposals sample points
    temp_vector = 1:33;
    temp_vector = repmat(temp_vector,101,1);
    temp_vector = reshape(temp_vector,33*101,1);
    mask_3_max_res_val = [];
    mask_3 = [];
    for i = 1:size(All_corner_pair_idx,1)
        %33*101,3;  ��  8096*3    distance
        xyz_query = All_proposals_sample_points{i};
        %mask_1 = Dis_matrix;
        xyz_input_idx = find(mask_1(i,:)==1);
        xyz_input = xyz_input_all(xyz_input_idx,:);
        Dis_matrix_1 = Distance_Points1_Points2_matrix(xyz_query,xyz_input);
        [dist_vector, idx] = min(Dis_matrix_1,[],2);
        idx_global = xyz_input_idx(idx);
        dis_mat = reshape(dist_vector,101,33);
        mask_3_max_res_val_temp = max(dis_mat',[],2);   % max project residual value;
        mask_3_max_res_val = [mask_3_max_res_val; mask_3_max_res_val_temp];
        % idx: vector [33*101, 1]
        idx_global = reshape(idx_global, 3333,1);
        temp_idx = [temp_vector,idx_global];
        mask_temp = zeros(33,8096);
        mask_temp(sub2ind(size(mask_temp), temp_idx(:,1), temp_idx(:,2))) = 1;  
        mask_3 = [mask_3; mask_temp];
        
    end   
   

% 3.4 mask4: each curve propsal   project    pred_edge_points   to
% generate mask 
    %All_proposals_sample_points;  % 256 cell [33*101,3]  33 curve proposals sample points
    %edge_points_idx = All_edge_point_idx;
    xyz_input_all = Input_point_cloud_8096;
%     index = edge_points_idx';
%     xyz_input = xyz_input_all(index,:);
    
    temp_vector = 1:33;
    temp_vector = repmat(temp_vector,101,1);
    temp_vector = reshape(temp_vector,33*101,1);
    mask_4_max_res_val = [];
    mask_4 = [];
    for i = 1:size(All_corner_pair_idx,1)
        %33*101,3;  ��  num_edge*3    distance
        xyz_query = All_proposals_sample_points{i};
        %mask_2 = Dis_matrix;
        xyz_input_idx = find(mask_2(i,:)==1);
        xyz_input = xyz_input_all(xyz_input_idx,:);
        Dis_matrix_2 = Distance_Points1_Points2_matrix(xyz_query,xyz_input);
        [dist_vector, idx] = min(Dis_matrix_2,[],2);
        idx_global = xyz_input_idx(idx);
        dis_mat = reshape(dist_vector,101,33);
        mask_4_max_res_val_temp = max(dis_mat',[],2);   % max project residual value;
        mask_4_max_res_val = [mask_4_max_res_val; mask_4_max_res_val_temp];
        % idx: vector [33*101, 1]
        idx_global = reshape(idx_global, 3333,1);
        temp_idx = [temp_vector,idx_global];      
        
        mask_4_temp = zeros(33,8096);
        mask_4_temp(sub2ind(size(mask_4_temp), temp_idx(:,1), temp_idx(:,2)))=1; 
        mask_4 = [mask_4; mask_4_temp];
    end   

end





function All_pair_down_sample_idx = Open_curve_sample_points(Input_point_cloud_8096, All_corner_pair_idx, All_edge_point_idx, num_sample)
% input: 
%   temp_training_data: include 8096 points; temp_training_data.PC_8096_edge_points_label_bin
%   corner_pair_idx: all corner pair index;
%   num_sample: for each pair corner, sampling how many points;

%1, extract pos, neg, remain corner pair;
%     pos_corner_idx_pairs = All_corner_pair_idx.pos_corner_idx_pairs;
%     neg_corner_idx_pairs = All_corner_pair_idx.neg_corner_idx_pairs; 
%     pad_corner_idx_pairs = All_corner_pair_idx.pad_corner_idx_pairs;
%     pos_neg_pad_idx = [pos_corner_idx_pairs;neg_corner_idx_pairs;pad_corner_idx_pairs];
    pos_neg_idx = All_corner_pair_idx;
    
%2, generate all center; and radius;
    all_pair_start_point = Input_point_cloud_8096(pos_neg_idx(:,1),:);
    all_pair_end_point   = Input_point_cloud_8096(pos_neg_idx(:,2),:);
    all_pair_center = all_pair_start_point + (all_pair_end_point - all_pair_start_point)./2;
    center_start_vector = all_pair_center - all_pair_start_point;
    all_pair_radius = sum(center_start_vector.*center_start_vector,2);
    
    
    %test-------------------------------------
%     Dis_matrix_1 = Distance_Points1_Points2_matrix(xyz2,xyz1); 
%    Thres_r2 = repmat(radius,1,size(xyz1,1));
%    Thres_r2 = Thres_r2 * 1.1;
%    Dis_matrix = Thres_r2 - Dis_matrix_1;
%    Dis_matrix(Dis_matrix>=0) = 1; 
%    Dis_matrix(Dis_matrix<0) = 0; 
   
   temp_start_center = all_pair_start_point - all_pair_center;
   dis_start_center = sum(temp_start_center.*center_start_vector,2);
   cmp_1 = dis_start_center - all_pair_radius*1.1;
   find(cmp_1>0)
   
   temp_end_center = all_pair_end_point - all_pair_center;
   dis_end_center = sum(temp_end_center.*temp_end_center,2);
   cmp_2 = dis_end_center - all_pair_radius*1.1;
   find(cmp_2>0)
   
    %-----------------------------------------
    
    
    
    

%3, samlpe n points in radius ball for each pair;   (just for each positve and negative pair to FPS)
   
    radius = all_pair_radius;
    edge_points_idx = All_edge_point_idx;
    xyz_input_all = Input_point_cloud_8096;
    %index = find(edge_points_idx>0);
    index = edge_points_idx';
    xyz_input = xyz_input_all(index,:);
    xyz_query = all_pair_center;
    All_pair_down_sample_idx = query_ball_point(radius, num_sample, xyz_input, xyz_query);   % now is local idx;
    All_pair_down_sample_idx = index(All_pair_down_sample_idx);
    
%     % top 2 cols, we need put these two corner points
%     All_pair_down_sample_idx(:,1)= pos_neg_idx(:,1);
%     All_pair_down_sample_idx(:,2)= pos_neg_idx(:,2);

    
% %4, for pad_corner pair, we just repeat these two index into size_num_sample;
%     cols = num_sample/2;
%     Pad_pair_down_sample_idx = repmat(pad_corner_idx_pairs,1,cols);
%     
% %5, combine together
%     All_pair_down_sample_idx = [All_pair_down_sample_idx; Pad_pair_down_sample_idx];
    
end


function All_pair_down_sample_idx = query_ball_point(radius, nsample, xyz1, xyz2)
%     '''
%     Input:
%         radius: float32, ball search radius
%         nsample: int32, number of points selected in each ball region
%         xyz1: (batch_size, ndataset, 3) float32 array, input points
%         xyz2: (batch_size, npoint, 3) float32 array, query points
%         pair_two_points_index: (batch_size, opoint,2) int32, two points
%         of pair index from 8096 points;
%     Output:
%         idx: (batch_size, npoint, nsample) int32 array, indices to input points
%         pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
%     '''
%    return grouping_module.query_ball_point(xyz1, xyz2, radius, nsample)

   Dis_matrix_1 = Distance_Points1_Points2_matrix(xyz2,xyz1); 
   Thres_r2 = repmat(radius,1,size(xyz1,1));
   Thres_r2 = Thres_r2 + 5e-4;
   Dis_matrix = Thres_r2 - Dis_matrix_1;
   Dis_matrix(Dis_matrix>=0) = 1; 
   Dis_matrix(Dis_matrix<0) = 0; 
   
   num_pairs = size(xyz2,1);
   All_pair_down_sample_idx = zeros(num_pairs,nsample);
   %x = sum(Dis_matrix,2)
   % for each corner pair to sample points(FPS) 
   for i = 1:size(xyz2,1)
       %i
       % generate mask
       ball_points_mask = Dis_matrix(i,:);
       
       % just extract dis< radius points from 8096 points;
       ball_all_points_idx = find(ball_points_mask>0);  % This idx is global related to 8096;
       ball_points = xyz1(ball_all_points_idx,:);
      
      % this idx(down_sample_point_idx) is local index related ball points;
      [~,down_sample_point_idx] = Farthest_Point_Sampling_piont_and_idx(ball_points,nsample);
      
      % transform local idx into global index;
      global_index = ball_all_points_idx(down_sample_point_idx);
      All_pair_down_sample_idx(i,:) = global_index;     
   end
   
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     % test visualization
%     load color
%     figure(2)
%     
%     down_sample_point1 = xyz2;
%     color_label = 1*ones(size(down_sample_point1,1),1);
%     Points_color_sharp_rgb1 = color(color_label,:);
%     point_surfaces2 = [down_sample_point1,Points_color_sharp_rgb1];
%     scatter3(point_surfaces2(:,1),point_surfaces2(:,2),point_surfaces2(:,3),500,point_surfaces2(:,4:6),'.');  
%     hold on
%     
%     down_sample_point1 = xyz1;
%     color_label = 100*ones(size(down_sample_point1,1),1);
%     Points_color_sharp_rgb1 = color(color_label,:);
%     point_surfaces2 = [down_sample_point1,Points_color_sharp_rgb1];
%     scatter3(point_surfaces2(:,1),point_surfaces2(:,2),point_surfaces2(:,3),50,point_surfaces2(:,4:6),'.');  
%     hold on
%     
%     down_sample_point = xyz1;
%     for j = 1:size(xyz2,1)
%         idx = All_pair_down_sample_idx(j,:);
%         down_sample_point1 = down_sample_point(idx,:);
%         color_label = (j)*ones(size(down_sample_point1,1),1);
%         Points_color_sharp_rgb1 = color(color_label,:);
%         point_surfaces2 = [down_sample_point1,Points_color_sharp_rgb1];
%         scatter3(point_surfaces2(:,1),point_surfaces2(:,2),point_surfaces2(:,3),50,point_surfaces2(:,4:6),'.');  
%         hold on
%       axis equal 
%     end
%     hold off
  
   
end










function dis = distance(a,b)
    A_B = a - b;
    dis = sum(A_B.^2);
end


function Dis_matrix = Distance_Points1_Points2(vertices_ball,vertices_points)

B=vertices_ball; % ball_num*3;
P=vertices_points; % points_num*3;
B1=sum(B.^2,2);
P1=sum(P.^2,2);
Num_b=numel(B1);
Num_p=numel(P1);
Dis_matrix= repmat(B1,1,Num_p)+ repmat(P1',Num_b,1) - 2*B*P';  % Num_b * Num_P distance matrix;
end
 
 
function Map_idx = Distance_Points1_Points2_with_num_return_distance(vertices_ball,vertices_points,num)

B=vertices_ball; % ball_num*3;
P=vertices_points; % points_num*3;
B1=sum(B.^2,2);
P1=sum(P.^2,2);
Num_b=numel(B1);
Num_p=numel(P1);
Dis_matrix= repmat(B1,1,Num_p)+ repmat(P1',Num_b,1) - 2*B*P';  % Num_b * Num_P distance matrix;

[~,Map_idx_1] = sort(Dis_matrix,2);
Map_idx = Map_idx_1(:,1:num);

end
  
function Map_idx = Distance_Points1_Points2_with_num(vertices_ball,vertices_points,num)

B=vertices_ball; % ball_num*3;
P=vertices_points; % points_num*3;
B1=sum(B.^2,2);
P1=sum(P.^2,2);
Num_b=numel(B1);
Num_p=numel(P1);
Dis_matrix= repmat(B1,1,Num_p)+ repmat(P1',Num_b,1) - 2*B*P';  % Num_b * Num_P distance matrix;

[~,Map_idx_1] = sort(Dis_matrix,2);
Map_idx = Map_idx_1(:,1:num);
end 
 
 



 
 
 
 
 
 
 
 
        