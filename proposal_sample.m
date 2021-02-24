function All_pair_down_sample_idx_cell_all = Open_curve_sample_points(Input_point_cloud_8096, All_corner_pair_idx, All_edge_point_idx, num_sample)
%1, 
    pos_neg_idx = All_corner_pair_idx;
    
%2, 
    all_pair_start_point = Input_point_cloud_8096(pos_neg_idx(:,1),:);
    all_pair_end_point   = Input_point_cloud_8096(pos_neg_idx(:,2),:);
    all_pair_center = all_pair_start_point + (all_pair_end_point - all_pair_start_point)./2;
    center_start_vector = all_pair_center - all_pair_start_point;
    all_pair_radius = sum(center_start_vector.*center_start_vector,2)*(1 + 0.5);

%3,  
    radius = all_pair_radius;
    edge_points_idx = All_edge_point_idx;
    xyz_input_all = Input_point_cloud_8096;
    index = edge_points_idx';
    xyz_input = xyz_input_all(index,:);
    xyz_query = all_pair_center;
    All_pair_down_sample_idx_cell = query_ball_point(radius, num_sample, xyz_input, xyz_query); 
    All_pair_down_sample_idx_cell = cellfun(@(x) index(x),All_pair_down_sample_idx_cell,'Unif',0);
    pos_neg_idx_cell_1 = num2cell(pos_neg_idx(:,1));
    pos_neg_idx_cell_2 = num2cell(pos_neg_idx(:,2));
    
    All_pair_down_sample_idx_cell_all = cellfun(@(x,y,z) [x,y,z],pos_neg_idx_cell_1, pos_neg_idx_cell_2, All_pair_down_sample_idx_cell,'Unif',0);
end


function All_pair_down_sample_idx_cell = query_ball_point(radius, nsample, xyz1, xyz2)
   Dis_matrix_1 = Distance_Points1_Points2(xyz2,xyz1); 
   Thres_r2 = repmat(radius,1,size(xyz1,1));
   Thres_r2 = Thres_r2 + 1e-6;
   Dis_matrix = Thres_r2 - Dis_matrix_1;
   Dis_matrix(Dis_matrix>=0) = 1; 
   Dis_matrix(Dis_matrix<0) = 0; 
   
   num_pairs = size(xyz2,1);
   All_pair_down_sample_idx_cell = cell(num_pairs,1);
   
   % 
   for i = 1:size(xyz2,1)
       % 
       ball_points_mask = Dis_matrix(i,:);
       ball_all_points_idx = find(ball_points_mask>0);  % This idx is related to input_set;
       All_pair_down_sample_idx_cell{i} = ball_all_points_idx;
   end

end


function Dis_matrix = Distance_Points1_Points2(vertices_ball,vertices_points)

B=vertices_ball; 
P=vertices_points; 
B1=sum(B.^2,2);
P1=sum(P.^2,2);
Num_b=numel(B1);
Num_p=numel(P1);
Dis_matrix= repmat(B1,1,Num_p)+ repmat(P1',Num_b,1) - 2*B*P'; 
end