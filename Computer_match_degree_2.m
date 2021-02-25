function Computer_match_degree_2(Input_point_cloud_8096, train_all_pair_val, open_pre_label_ind, open_lin_idx,open_cyc_idx)

%
All_pair_down_sample_idx_cell = proposal_sample(Input_point_cloud_8096, train_all_pair_val, open_pre_label_ind, 128);
 
% 
path_size = cell2mat(cellfun(@(x) length(x), All_pair_down_sample_idx_cell,'Unif',0));
path_size_zero_idx = find(path_size<=3);  %小于2个
All_pair_down_sample_idx_cell(path_size_zero_idx) = [];


% 1,
All_proposals_sample_points = generate_proposal(Input_point_cloud_8096,All_pair_down_sample_idx_cell);

% 2，
[max_res_val_scale_cell,max_res_val_cell,max_res_val_idx_cell,max_res_pro_idx_cell] = Computer_res(Input_point_cloud_8096, All_proposals_sample_points, All_pair_down_sample_idx_cell,open_lin_idx,open_cyc_idx);

path_size = cell2mat(cellfun(@(x) length(unique(x)), max_res_pro_idx_cell,'Unif',0));
path_size_zero_idx = find(path_size<=4);  %小于3个
All_pair_down_sample_idx_cell(path_size_zero_idx) = [];
All_proposals_sample_points(path_size_zero_idx) = [];
max_res_val_scale_cell(path_size_zero_idx) = [];
max_res_val_cell(path_size_zero_idx) = [];
max_res_val_idx_cell(path_size_zero_idx) = [];
max_res_pro_idx_cell(path_size_zero_idx) = [];

% 4,
thr_1 = 0.01;
max_res_val_idx_mat = cell2mat(max_res_val_cell);
idx_thr_1 = find(max_res_val_idx_mat<thr_1);

num_min = 50;
if length(idx_thr_1) < num_min
   if length(max_res_val_idx_mat)<num_min;
      idx_thr_1 = 1:length(max_res_val_idx_mat);
   else
      [val_res,idx_res] = sort(max_res_val_idx_mat);
      idx_thr_1 = idx_res(1:num_min);
   end
end

sample_cell_thr_1 = All_proposals_sample_points(idx_thr_1);
path_cell_thr_1 = max_res_pro_idx_cell(idx_thr_1);
max_res_val_idx_mat_thr_1 = max_res_val_idx_mat(idx_thr_1);
All_pair_down_sample_idx_cell_thr_1 = All_pair_down_sample_idx_cell(idx_thr_1);
max_res_val_idx_cell_thr_1 = max_res_val_idx_cell(idx_thr_1);


%%
num_sample_cell_thr_1 = length(sample_cell_thr_1);
sel_points_cell = cell(num_sample_cell_thr_1,1);
for i = 1:num_sample_cell_thr_1
    temp_points = sample_cell_thr_1{i};
    temp_idx = max_res_val_idx_cell_thr_1{i};
    sel_points_cell{i} = temp_points((temp_idx-1)*101+1:temp_idx*101,:);
end


% %test
% % % test---------------------------
% load color
% fig_100 = figure(100);
% num_pairs = numel(sample_cell_thr_1);
% %num_curves = cellfun(@(x) size(x,1)/101, All_proposals_sample_points,'Unif',0);
% 
% for i = 1:num_pairs
%     i
%     temp_proposal_sample_points = sample_cell_thr_1{i};
%     best_pro_curve_points = temp_proposal_sample_points((1-1)*101+1:1*101,:);
%     
%     pro_curve_points_color = color(i+1000,:);  
%     pro_curve_points_color = repmat(pro_curve_points_color,101,1);
%     pro_curve_all = [best_pro_curve_points, pro_curve_points_color];
% %     scatter3(pro_curve_all(:,1),pro_curve_all(:,2),pro_curve_all(:,3),50,pro_curve_all(:,4:6),'.');   
%     temp_point_color = pro_curve_all([1,end],:);
%     scatter3(temp_point_color(:,1),temp_point_color(:,2),temp_point_color(:,3),50,temp_point_color(:,4:6),'.');
%     line(temp_point_color(:,1),temp_point_color(:,2),temp_point_color(:,3));
%     hold on
%     
% %     temp1 = path_cell_thr_1{i};
% %     temp1(:) = i;
% %     temp2 = path_cell_thr_1{i};
% %     Points_color = Input_point_cloud_8096(temp2,:);
% %     Points_color_label = temp1;
% %     Points_color_rgb = color(Points_color_label,:);     
% %     point_color = [Points_color,Points_color_rgb];
% %     scatter3(point_color(:,1),point_color(:,2),point_color(:,3),50,point_color(:,4:6),'.'); % 画图 
% %     line(point_color(:,1),point_color(:,2),point_color(:,3));
%     
%     pause(0.1)
%     axis equal
% end
% 
% title('Between corner points path')    
% hold off




%%
scores = max_res_val_idx_mat_thr_1;
boxes = path_cell_thr_1;
overlap_thr = 0.5;
dis_thr = 0.02;
pick = NMS_open(sel_points_cell, boxes, scores, overlap_thr, dis_thr);

All_pair_do_sa_cell = All_pair_down_sample_idx_cell_thr_1(pick);
All_proposals_cell = sample_cell_thr_1(pick);
All_res_val_idx_cell = max_res_val_idx_cell_thr_1(pick);

%%
[all_proposals_cell, all_res_val_idx_cell] = merge_node(Input_point_cloud_8096, All_pair_do_sa_cell, All_proposals_cell, All_res_val_idx_cell, open_pre_label_ind, open_lin_idx,open_cyc_idx);

% test vis pick
load color
fig_4 = figure(4);
num_pairs = numel(all_proposals_cell);


for i = 1:num_pairs
    temp_proposal_sample_points = all_proposals_cell{i};
    temp_proposal_sampel_idx = all_res_val_idx_cell{i};
    best_pro_curve_points = temp_proposal_sample_points((temp_proposal_sampel_idx-1)*101+1:temp_proposal_sampel_idx*101,:);
    
    pro_curve_points_color = color(i+1000,:);  
    pro_curve_points_color = repmat(pro_curve_points_color,101,1);
    pro_curve_all = [best_pro_curve_points, pro_curve_points_color];
    scatter3(pro_curve_all(:,1),pro_curve_all(:,2),pro_curve_all(:,3),50,pro_curve_all(:,4:6),'.');
    hold on
    axis equal
end

title('Between corner points path')    
hold off


end



function [all_proposals_cell, all_res_val_idx_cell] = merge_node(input_pl, All_pair_do_sa_cell, All_proposals_cell, All_res_val_idx_cell, open_pre_label_ind, open_lin_idx,open_cyc_idx)

%0,
pick_cell= All_pair_do_sa_cell; 
All_pair_idx_cell = cellfun(@(x) [x(1:2)],pick_cell,'Unif',0);
All_pair_idx_mat = cell2mat(All_pair_idx_cell);

%1,
x = All_pair_idx_mat(:);
x = sort(x);
d=diff([x;max(x)+1]);
count = diff(find([1;d]));
y =[x(find(d)) count];

idx_one = find(y(:,2)==1);
all_ele_int = y(:,1);
ele_one = all_ele_int(idx_one);



%3，
num_ele_one = length(ele_one);
pair_con_re = zeros(num_ele_one,1);
all_proposals_cell = All_proposals_cell;
all_res_val_idx_cell = All_res_val_idx_cell;
for i = 1:num_ele_one
    if pair_con_re(i)>0
       continue;
    end
    
    %1,
    [idx_row1,idx_col1] = find(All_pair_idx_mat == ele_one(i));
    all_ele = unique(All_pair_idx_mat);
    
    %2，
    all_ele_points = input_pl(all_ele,:);
    ele_one_points = input_pl(ele_one(i),:);
    dis_one_all = Distance_Points1_Points2_matrix(ele_one_points, all_ele_points);
    ele_one_pairs = All_pair_idx_mat(idx_row1,:);
    [~,ind_all_ele] = ismember(ele_one_pairs,all_ele);
    temp_row = dis_one_all;
    temp_row(ind_all_ele) = 10000;
    dis_one_all = temp_row;
    [dis_one_all_order, idx_order] = sort(dis_one_all);
    one_neibor_idx = all_ele(idx_order(1));
    
    %3, 
    [idx_row2,idx_col2] = find(All_pair_idx_mat == one_neibor_idx);
    one_neibor_idx_count = length(idx_row2);
    
    %4,
    pair_con_re(i) = 1; 
    if one_neibor_idx_count > 1
       temp_pair1 =  All_pair_idx_mat(idx_row1,:);
       temp_pair1(idx_col1) = one_neibor_idx; 
       All_pair_idx_mat(idx_row1,idx_col1) = one_neibor_idx;
        
      All_pair_down_sample_idx_cell = proposal_sample(input_pl, temp_pair1, open_pre_label_ind, 128);
      All_proposals_points = generate_proposal(input_pl,All_pair_down_sample_idx_cell);
      [max_res_val_scale_cell,max_res_val_cell,max_res_val_idx_cell,max_res_pro_idx_cell] = Computer_res(input_pl, All_proposals_points, All_pair_down_sample_idx_cell,open_lin_idx,open_cyc_idx);
      all_proposals_cell(idx_row1) = All_proposals_points(1);
      all_res_val_idx_cell(idx_row1) = max_res_val_idx_cell(1);
      
    else    
      temp_pair1 =  All_pair_idx_mat(idx_row1,:);
      temp_pair2 =  All_pair_idx_mat(idx_row2,:);
      
      idx_ele_nei = find(ele_one == one_neibor_idx);
      pair_con_re(idx_ele_nei) = 1;
      temp_pair1_re = temp_pair1;
      temp_pair1_re(idx_col1) = temp_pair2(idx_col2);
      temp_pair2_re = temp_pair2;
      temp_pair2_re(idx_col2) = temp_pair1(idx_col1);
      temp_pair1 = [temp_pair1;temp_pair2_re;temp_pair1_re;temp_pair2];

       
      All_pair_sample_idx = proposal_sample(input_pl, temp_pair1, open_pre_label_ind, 128); 
      All_proposals_points = generate_proposal(input_pl,All_pair_sample_idx);
      [max_res_val_scale_cell,max_res_val_cell,max_res_val_idx_cell,max_res_pro_idx_cell] = Computer_res(input_pl, All_proposals_points, All_pair_sample_idx,open_lin_idx,open_cyc_idx);

      max_res_val_vector = cell2mat(max_res_val_cell);
      if sum(max_res_val_vector(1:2)) <= sum(max_res_val_vector(3:4))
         %pair_con_re([idx_row1,idx_row2]) = ele_one(i);
         all_proposals_cell(idx_row1) = All_proposals_points(1);
         all_proposals_cell(idx_row2) = All_proposals_points(2);
         all_res_val_idx_cell(idx_row1) = max_res_val_idx_cell(1);
         all_res_val_idx_cell(idx_row2) = max_res_val_idx_cell(2);
         All_pair_idx_mat(idx_row2,idx_col2) = ele_one(i);
      else
         %pair_con_re([idx_row1,idx_row2]) = one_neibor_idx; 
         all_proposals_cell(idx_row1) = All_proposals_points(3);
         all_proposals_cell(idx_row2) = All_proposals_points(4);
         all_res_val_idx_cell(idx_row1) = max_res_val_idx_cell(3);
         all_res_val_idx_cell(idx_row2) = max_res_val_idx_cell(4);
         All_pair_idx_mat(idx_row1,idx_col1) = one_neibor_idx;           
      end
      
    end
end


end








function All_proposals_sample_points = generate_proposal(Input_point_cloud_8096,All_path_cell)


input_points = Input_point_cloud_8096;
num_sample = 128;
num_path_cell = numel(All_path_cell);
All_proposals_sample_points = cell(num_path_cell,1);


for i = 1:num_path_cell
    temp_path_i = All_path_cell{i};
    temp_path_i = [temp_path_i(1:2),setdiff(unique(temp_path_i),temp_path_i(1:2))];
    num_temp_eles = length(temp_path_i);
    if num_temp_eles == 2
       temp_path = [temp_path_i(1),temp_path_i(2),temp_path_i(2)];
    elseif num_temp_eles > num_sample + 2
       path_eles = temp_path_i(3:end);
       path_eles_points = input_points(path_eles,:);
       %num_sample-2;
       [~,down_sample_point_idx] = Farthest_Point_Sampling_piont_and_idx(path_eles_points,num_sample);
       sample_points_global_idx = path_eles(down_sample_point_idx);
       temp_path = [temp_path_i(1),sample_points_global_idx,temp_path_i(2)];
    elseif num_temp_eles <= num_sample + 2
       temp_path = [temp_path_i(1),temp_path_i(3:end),temp_path_i(2)]; 
    end

    num_temp_path = length(temp_path)-2;
    para_1 = temp_path(:,1);
    para_1 = repmat(para_1,1,num_temp_path);
    para_2 = temp_path(:,end);
    para_2 = repmat(para_2,1,num_temp_path);
    para_3 = temp_path(2:end-1);
    para_all = [para_1',para_2',para_3'];

    
    one_proposal_sample_points = sample_para(para_all,Input_point_cloud_8096);
    All_proposals_sample_points{i} = one_proposal_sample_points;
end

end


function all_curve_sample_points = sample_para(para_matrix,Input_point_cloud_8096)

num = 100; 
% 1,
para_points_idx = para_matrix(1,1:2);
para_points = Input_point_cloud_8096(para_points_idx,:);
line_sample_points = line_vis(para_points,num);

% 2,
curve_sample_points = [];
for i= 1:size(para_matrix,1)
  para_curve_points_idx = para_matrix(i,:);
  para_curve_points = Input_point_cloud_8096(para_curve_points_idx,:);
  temp_sample_points = arcPlot(para_curve_points,num);
  curve_sample_points = [curve_sample_points; temp_sample_points];
end
all_curve_sample_points = [line_sample_points; curve_sample_points];

end


function [max_res_val_scale_cell,max_res_val_cell,max_res_val_idx_cell,max_res_pro_idx_cell] = Computer_res(Input_point_cloud_8096, All_proposals_sample_points, All_pair_down_sample_idx_cell,open_lin_idx,open_cyc_idx)

    xyz_input_all = Input_point_cloud_8096;
    num_pairs = numel(All_proposals_sample_points);
    max_res_val_cell = cell(num_pairs,1);
    max_res_val_scale_cell = cell(num_pairs,1);
    max_res_val_idx_cell = cell(num_pairs,1);
    max_res_pro_idx_cell = cell(num_pairs,1);
    for i = 1:length(All_pair_down_sample_idx_cell)
        xyz_query = All_proposals_sample_points{i}; 
        xyz_input_idx = All_pair_down_sample_idx_cell{i};  
        xyz_input = xyz_input_all(xyz_input_idx,:);
        Dis_matrix_2 = Distance_Points1_Points2_matrix(xyz_query,xyz_input);
        
        
        [dist_vector, idx] = min(Dis_matrix_2,[],2);
        idx_global = xyz_input_idx(idx);
        num_proposals = length(dist_vector)/101;
        dis_mat = reshape(dist_vector,101,num_proposals);
        idx_global_mat = reshape(idx_global,101,num_proposals);
        max_res_val_temp = mean(sqrt(dis_mat'),2); 
        
        [val_min_res,idx_min_res] = min(max_res_val_temp);
        max_res_val_cell{i} = max_res_val_temp; %val_min_res; 
        max_res_val_idx_cell{i} = idx_min_res;
        max_res_pro_idx_cell{i} = idx_global_mat(:,idx_min_res);
        
        temp_res_vals = max_res_val_cell{i};
        temp_res_pro_idx = unique(max_res_pro_idx_cell{i}); 
        temp_lin_inter = intersect(temp_res_pro_idx,open_lin_idx);
        temp_cyc_inter = intersect(temp_res_pro_idx,open_cyc_idx);

        if length(temp_lin_inter) > length(temp_cyc_inter)  
           min_temp_res_val = temp_res_vals(1);
           min_temp_res_idx = 1;
        else
           [min_temp_res_val, min_temp_res_idx] = min(temp_res_vals(2:end));
           min_temp_res_idx = min_temp_res_idx + 1;
        end
        max_res_val_cell{i} = min_temp_res_val;
        max_res_val_idx_cell{i} = min_temp_res_idx;      

        path_length = norm(xyz_input(1,:) - xyz_input(2,:));
        path_scale = 1./path_length; 
        max_res_val_scale_cell{i} = min_temp_res_val.*path_scale;
        
    end
end

function Dis_matrix = Distance_Points1_Points2_matrix(vertices_ball,vertices_points)

B=vertices_ball; 
P=vertices_points; 
B1=sum(B.^2,2);
P1=sum(P.^2,2);
Num_b=numel(B1);
Num_p=numel(P1);
Dis_matrix= repmat(B1,1,Num_p) + repmat(P1',Num_b,1) - 2*B*P';  
Dis_matrix(Dis_matrix<0) = 0;
end



function Points = arcPlot(temp_points_edge_mask,num)  %()


% 
temp_points_corner_1 = temp_points_edge_mask(1,:);
temp_points_corner_2 = temp_points_edge_mask(2,:); 
temp_points_control_1 = temp_points_edge_mask(3,:);
pos = [temp_points_corner_1;temp_points_control_1;temp_points_corner_2];
pos1 = unique(pos,'rows'); 

%
if size(pos1,1)==2
   Points = line_vis(pos1,num);
   return;
elseif size(pos1,1)==1
   pos(2,:) = pos(2,:)+ 1e-4;
   Points = line_vis(pos,num);
   return;
end

arcFlag = 0;

p1=pos(1,:);
p2=pos(2,:);
p3=pos(3,:);

p1p2=p2-p1;
u1=p1p2/sum(p1p2.^2);

p2p3=p3-p2;
u2=unitVec(p2p3);

normal=cross(p1p2,p2p3);
normal=unitVec(normal);


per2=cross(normal,u2);
per2=unitVec(per2);

mid1=(p1+p2)/2;
mid2=(p2+p3)/2;
dis2=sqrt(sum(p2p3.^2));
L=(vecDot(mid1,u1)-vecDot(mid2,u1))/vecDot(per2,u1);

arc_R=sqrt(L^2+1.0/4*dis2^2);

C=mid2+L*per2;

vec1=p1-C;
vec1=unitVec(vec1);
vec2=p3-C;
vec2=unitVec(vec2);
theta_1=vecDot(vec1,vec2);
normal2=cross(vec1,vec2);
normal2 = unitVec(normal2);
if(arcFlag==0)       
    if fix(theta_1) == -1;      
        theta = pi;
    elseif fix(theta_1) == 1; 
        Points = line_vis(pos1,num);
        return;
    else
        theta=acos(theta_1);
    end
    if(sqrt(sum((normal-normal2).^2))>1.0e-5)
        theta=2*pi-theta;
    end
else
    theta=2*pi;
end


vx=[1,0,0];
vy=[0,1,0];
vz=[0,0,1];
v1=vec1;
v3=normal;
v2=cross(v3,v1);
v2=unitVec(v2);
r11=vecDot(v1,vx);r12=vecDot(v2,vx);r13=vecDot(v3,vx);
r21=vecDot(v1,vy);r22=vecDot(v2,vy);r23=vecDot(v3,vy);
r31=vecDot(v1,vz);r32=vecDot(v2,vz);r33=vecDot(v3,vz);
Tr=[r11,r12,r13,0;
    r21,r22,r23,0;
    r31,r32,r33,0;
    0,0,0,1];
Tt=[1,0,0,C(1);
    0,1,0,C(2);
    0,0,1,C(3);
    0,0,0,1];

t=0:(theta)/num:theta;
x1=arc_R*cos(t);
y1=arc_R*sin(t);
z1=t*0;
pt=[x1;y1;z1;ones(size(x1))];
pt=Tt*Tr*pt;

Points = pt(1:3,:);
Points = Points';
end


%%
function vector=unitVec(vec)
vector=vec/sqrt(sum(vec.^2));
end

%%
function z=vecDot(x,y)
z=sum(x.*y);
end

function Points = line_vis(temp_points_edge_mask,num)  %()

temp_points_corner_1 = temp_points_edge_mask(1,:);
temp_points_corner_2 = temp_points_edge_mask(2,:);
pos  = [temp_points_corner_1;temp_points_corner_2];

pos1 = unique(pos,'rows'); 

if size(pos1,1)==1
   pos(2,:) = pos(2,:)+ 1e-4;
end

p1=pos(1,:);
p2=pos(2,:);

t=0:1/num:1;
x1=p1(1) + (p2(1)-p1(1))*t;
y1=p1(2) + (p2(2)-p1(2))*t;
z1=p1(3) + (p2(3)-p1(3))*t;
Points=[x1;y1;z1]';

end



