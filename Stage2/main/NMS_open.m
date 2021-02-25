function  idx_global = NMS_open(proposals_points, proposals, res_vals, overlap_thr, dis_thr)  %idx_global

% 1,
num_proposal = length(proposals);
idx_mat = triu(ones(num_proposal)) - eye(num_proposal);
[left_idx,right_idx,~] = find(idx_mat);
iou_path_12 = zeros(length(left_idx),1);
for i = 1:length(left_idx)
    path1 = proposals{left_idx(i)};
    path2 = proposals{right_idx(i)};
    iou_path_12(i) = iou_open(path1,path2);
end
iou_mat = Graph_to_matrix([left_idx,right_idx], iou_path_12, num_proposal);

% 2,
iou_mat(iou_mat>overlap_thr) = 1;
iou_mat(iou_mat<=overlap_thr) = 0;
iou_mat = iou_mat + iou_mat';

% 3,
res_com = zeros(length(left_idx),1);
for i = 1:length(left_idx)
    path1 = proposals{left_idx(i)};
    path2 = proposals{right_idx(i)};
    res_com(i) = compair_path_1_2(path1,path2, left_idx(i), right_idx(i), res_vals);
end
res_com_mat = Graph_to_matrix([left_idx,right_idx], res_com, num_proposal);

% 4, select com_thr < 0
res_com_mat = res_com_mat - res_com_mat';
res_com_mat(res_com_mat>0) = 1;
res_com_mat(res_com_mat<=0) = 0;


% 5, 
iou_res_mat = iou_mat.*res_com_mat;
sum_iou_res = sum(iou_res_mat,2);
idx_val = find(sum_iou_res == 0);


% 6,
proposals_points_sel = proposals_points(idx_val);
num_proposal_sel = length(proposals_points_sel);
idx_sel_mat = triu(ones(num_proposal_sel)) - eye(num_proposal_sel);
[left_sel_idx,right_sel_idx,~] = find(idx_sel_mat);
dis_path_12 = zeros(length(left_sel_idx),1);
for i = 1:length(left_sel_idx)
    points1 = proposals_points_sel{left_sel_idx(i)};
    points2 = proposals_points_sel{right_sel_idx(i)};
    dis_path_12(i) = compute_dis_path12(points1, points2);
end
dis_mat = Graph_to_matrix([left_sel_idx,right_sel_idx], dis_path_12, num_proposal_sel);
dis_mat = dis_mat + dis_mat';
dis_mat(dis_mat>dis_thr) = 10000;
dis_mat(dis_mat<=dis_thr) = 1;
dis_mat(dis_mat>1) = 0;


% 3, gen res_com_matrix
res_com_sel = zeros(length(left_sel_idx),1);
proposals_sel = proposals(idx_val);
for i = 1:length(left_sel_idx)
    path1 = unique(proposals_sel{left_sel_idx(i)});
    path2 = unique(proposals_sel{right_sel_idx(i)});
    if length(path1) >= length(path2)
        res_com_sel(i) = length(path2) - length(path1)-1;    
    else
        res_com_sel(i) = 1;
    end
end
res_com_mat_sel = Graph_to_matrix([left_sel_idx,right_sel_idx], res_com_sel, num_proposal_sel);

res_com_mat_sel = res_com_mat_sel - res_com_mat_sel';
res_com_mat_sel(res_com_mat_sel>0) = 1;
res_com_mat_sel(res_com_mat_sel<=0) = 0;

% 7,
dis_res_mat = dis_mat.*res_com_mat_sel;
sum_dis_res_mat = sum(dis_res_mat,2);
idx_dis_res_val = find(sum_dis_res_mat == 0); 

idx_global = idx_val(idx_dis_res_val);


end

function dis = compute_dis_path12(points1, points2)
[min_xyz1,max_xyz1] = aabb(points1);
[min_xyz2,max_xyz2] = aabb(points2);
min_max_xyz = max([min_xyz1;min_xyz2]);
max_min_xyz = min([max_xyz1;max_xyz2]);
temp_box = max_min_xyz - min_max_xyz;
  
if temp_box(1)>0 && temp_box(2)>0 && temp_box(3)>0       
    box1 = max_xyz1 - min_xyz1;
    vol1 = box1(1)*box1(2)*box1(3);
    box2 = max_xyz2 - min_xyz2;
    vol2 = box2(1)*box2(2)*box2(3);
    vol_12 = temp_box(1)*temp_box(2)*temp_box(3);
    vol_rate = max(vol_12/vol1,vol_12/vol2);
    dis = hausdorff(points1,points2); 
else
   dis = 10000;
end

end



function [min_xyz,max_xyz] = aabb(point_set)
   % aabb bounding box
    min_xyz = min(point_set);
    max_xyz = max(point_set);        
    min_max_vector = min_xyz - max_xyz;        
    max_min_vector = max_xyz - min_xyz;
    min_xyz = min_xyz + 0.15*min_max_vector;
    max_xyz = max_xyz + 0.15*max_min_vector;
    
   length_xyz = max_xyz - min_xyz;
   idx = find(length_xyz < 1e-4);
   min_xyz(idx) = min_xyz(idx) - 1e-4;
   max_xyz(idx) = max_xyz(idx) + 1e-4;

end



function adjMatrix = Graph_to_matrix(edges, weights, num_matrix)
if isempty(edges)
    adjMatrix = zeros(num_matrix,num_matrix);
else
    adjMatrix = full(sparse(edges(:,1),edges(:,2),weights ,num_matrix,num_matrix)); 
end
end


%%
function [dist] = hausdorff(A, B) 
if(size(A,2) ~= size(B,2)) 
    fprintf( 'WARNING: dimensionality must be the same\n' ); 
    dist = []; 
    return; 
end
dist = min(compute_dist(A, B), compute_dist(B, A));
end

%% Compute distance
function dist = compute_dist(A, B) 

dist_matrix = Distance_Points1_Points2(A,B);
dist_vector = min(dist_matrix,[],2);
dist = sqrt(mean(dist_vector));

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



function iou_path_12 = iou_open(path1,path2)

% union
path_12_u = length(unique([path1,path2]));

% inter
path_12_i = length(intersect(unique(path1),unique(path2)));

% iou
iou_path_12 = path_12_i/path_12_u;

end


%
function res_com = compair_path_1_2(path1,path2,p1_idx, p2_idx, res_vals)
   res_val_1 = res_vals(p1_idx);
   res_val_2 = res_vals(p2_idx);
   num_path1 = length(unique(path1));
   num_path2 = length(unique(path2));
   
   res_com = res_val_1/(num_path1^0.5) - res_val_2/(num_path2^0.5);  
end

