function Vis_open_final()

FindFiles_test_data = '.\test_result_2_1\';

Files = dir(fullfile(FindFiles_test_data));
filenames = {Files.name}';
filenames = filenames(3:length(filenames));
filenames=filenames';
load color

for i = 1:length(filenames)
    load([FindFiles_test_data,filenames{i}]);
    filenames{i}
    [num_data, num_points,~] = size(input_points_edge_corner);
    for j =1:num_data 
    i
    j
        tic 
   %% input   
   train_all_pair = squeeze(input_corner_pair(j,:,:)) + 1;
   con_label_pre = squeeze(pre_label_1(j,:,:));
   not_label_pre = squeeze(pre_label_2(j,:,:));
   input_pl = squeeze(input_points_edge_corner(j,:,1:3));
   edge_pro = squeeze(input_points_edge_corner(j,:,4));
   corner_pro = squeeze(input_points_edge_corner(j,:,5));

%% predict data pre-processing
%
con_pro_thr = 0.2;
con_label_pre = exp(con_label_pre);
sum_con_pre = sum(con_label_pre,2);
con_label_pre = con_label_pre./repmat(sum_con_pre,1,2);

con_pre_label_ind = find(con_label_pre(:,2)>con_pro_thr);
con_pre_label = zeros(size(con_label_pre,1),1);
con_pre_label(con_pre_label_ind) = 1;
   
%
open_pro_thr_pro = 0.4;
open_pro =squeeze(pre_label_5(j,:,:)) ;
open_pro = exp(open_pro);
sum_open_pre = sum(open_pro,2);
open_pro = open_pro./repmat(sum_open_pre,1,2);

open_pre_label_ind = find(open_pro(:,2)>open_pro_thr_pro);
open_edge_points_pre = open_pre_label_ind;
open_pre_label = zeros(size(open_pro,1),1);
open_pre_label(open_pre_label_ind) = 1;

%
closed_pro_thr_pro = 0.5;
closed_pro =squeeze(pre_label_6(j,:,:)) ;
closed_pro = exp(closed_pro);
sum_closed_pre = sum(closed_pro,2);
closed_pro = closed_pro./repmat(sum_closed_pre,1,2);

closed_pre_label_ind = find(closed_pro(:,2)>closed_pro_thr_pro);
closed_edge_points_pre = closed_pre_label_ind; %setdiff(closed_pre_label_ind,open_edge_points_pre);
closed_pre_label = zeros(size(closed_pro,1),1);
closed_pre_label(closed_pre_label_ind) = 1;


%
open_lin_pro_thr_pro = 0.5;
open_lin_pro =squeeze(pre_label_7(j,:,:)) ;
open_lin_pro = exp(open_lin_pro);
sum_open_line_pre = sum(open_lin_pro,2);
open_lin_pro = open_lin_pro./repmat(sum_open_line_pre,1,2);

open_lin_pre_label_ind = find(open_lin_pro(:,2)>open_lin_pro_thr_pro);
open_lin_edge_points_pre = open_lin_pre_label_ind; %setdiff(closed_pre_label_ind,open_edge_points_pre);
open_lin_pre_label = zeros(size(open_lin_pro,1),1);
open_lin_pre_label(open_lin_pre_label_ind) = 1;


%
open_cyc_pro_thr_pro = 0.5;
open_cyc_pro =squeeze(pre_label_8(j,:,:)) ;
open_cyc_pro = exp(open_cyc_pro);
sum_open_cyc_pre = sum(open_cyc_pro,2);
open_cyc_pro = open_cyc_pro./repmat(sum_open_cyc_pre,1,2);

open_cyc_pre_label_ind = find(open_cyc_pro(:,2)>open_cyc_pro_thr_pro);
open_cyc_edge_points_pre = open_cyc_pre_label_ind; %setdiff(closed_pre_label_ind,open_edge_points_pre);
open_cyc_pre_label = zeros(size(open_cyc_pro,1),1);
open_cyc_pre_label(open_cyc_pre_label_ind) = 1;


%%
num_matrix = 128;
order_corner_idx_global = [train_all_pair((end-num_matrix+2):end,1); train_all_pair((end-num_matrix+2),2)]; 
All_corner_points = input_pl(order_corner_idx_global,:);

%1，
idx_conflict = intersect(closed_pre_label_ind,order_corner_idx_global);

%2, 
num_corner = numel(idx_conflict);
for i_coner = 1:num_corner
   idx_1_corner = find(train_all_pair(:,1)==idx_conflict(i_coner));
   idx_2_corner = find(train_all_pair(:,2)==idx_conflict(i_coner));   
   idx_all = [idx_1_corner;idx_2_corner];
   train_all_pair(idx_all,:) = 0;
   con_pre_label(idx_all,:) = 0;
   not_pre_label(idx_all,:) = 0;
end

%%
All_open_closed_edge_points_pre = [open_edge_points_pre;closed_edge_points_pre];
All_open_closed_edge_points_pre_label = [ones(1,size(open_edge_points_pre,1)),zeros(1,size(closed_edge_points_pre,1))];
All_open_closed_edge_points = input_pl(All_open_closed_edge_points_pre,:);
dis_closed_all = Distance_Points1_Points2_matrix(All_corner_points , All_open_closed_edge_points);

num_neibor_from_all = 8;
[~,idx_neibor_from_all] = sort(dis_closed_all,2);
idx_neibor_from_all = idx_neibor_from_all(:,2:num_neibor_from_all);
closed_edge_points_ind = zeros(size(All_corner_points,1),num_neibor_from_all-1);
for i_closed_edge = 1:size(All_corner_points,1)
   closed_edge_points_ind(i_closed_edge,:) = All_open_closed_edge_points_pre_label(idx_neibor_from_all(i_closed_edge,:));
end
closed_edge_points_ind_sum =  sum(closed_edge_points_ind,2);
fake_closed_edge_idx = find(closed_edge_points_ind_sum<num_neibor_from_all/2);

lin_mat = ones(num_matrix,num_matrix);
lin_mat = lin_mat - tril(ones(num_matrix,num_matrix));
[lin_left_idx,lin_right_idx,~] = find(lin_mat);  

%%
num_f = numel(fake_closed_edge_idx);
for i_fake = 1:num_f
   idx_1 = find(lin_left_idx == fake_closed_edge_idx(i_fake));
   idx_2 = find(lin_right_idx == fake_closed_edge_idx(i_fake));
   idx_all = [idx_1;idx_2];
   train_all_pair(idx_all,:) = 0;
   con_pre_label(idx_all,:) = 0;
   not_pre_label(idx_all,:) = 0;
end



%% 
[~, train_all_pair_uni_idx] = unique(sort(train_all_pair,2),'rows'); %unique(train_all_pair,'rows');
all_train_all_pair_uni_idx = 1:8128;
res_train_all_pair_uni_idx = setdiff(all_train_all_pair_uni_idx,train_all_pair_uni_idx);
con_pre_label(res_train_all_pair_uni_idx) = 0;
not_pre_label(res_train_all_pair_uni_idx) = 0;


%%
if isequal(train_all_pair,zeros(size(train_all_pair,1),2))
    continue;
else
    all_ele = sort(unique(train_all_pair));
    one_ele = all_ele(end);
    train_all_pair(train_all_pair == 0) = one_ele;
end

%%
fig1_vis(input_pl, edge_pro, unique(train_all_pair))

%% fig6 all open and closed edge;

fig2_vis(input_pl, closed_pre_label, open_pre_label);


%%
idx_con_pre_label = find(con_pre_label==1);
train_all_pair_val = train_all_pair(idx_con_pre_label,:);
Computer_match_degree_2(input_pl,train_all_pair_val,open_pre_label_ind,open_lin_pre_label_ind,open_cyc_pre_label_ind); % path_cell



    end
 end

end




function fig2_vis(input_points, Closed_label, Open_label)

    % figure 11: all open edge and closed edge
    load color
    fig_2 = figure(2);
    global_closed_edge_idx = find(Closed_label);
    closed_edge_points = input_points(global_closed_edge_idx,:);
    temp_points = closed_edge_points;
    temp_color = color(1,:);  
    temp_color = repmat(temp_color,size(closed_edge_points,1),1);
    temp_all = [temp_points,temp_color];
    scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.');  

    hold on

    global_open_edge_idx = find(Open_label);
    open_edge_points = input_points(global_open_edge_idx,:);
    temp_points = open_edge_points;
    temp_color = color(2,:);  
    temp_color = repmat(temp_color,size(open_edge_points,1),1);
    temp_all = [temp_points,temp_color];
    scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.');
    axis equal
    title('GT Edge and Corner points') %,'color','r'?)  %,'FontSize','12'        
    hold off  

end


%% 输入图
function fig11_vis(input_points, All_edge_points_idx, corner_idx)

        fig11 = figure(11)
        load color
 
        All_edge_points = input_points(All_edge_points_idx,:);

        temp_color_label = zeros(size(input_points,1),1)+2;
        All_edge_points_color_idx = temp_color_label(All_edge_points_idx)+1;
        All_edge_points_color = color(All_edge_points_color_idx,:);     
        temp_all = [All_edge_points,All_edge_points_color];
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.');
        
        hold on
        
        

        down_sample_point2 = input_points(corner_idx,:);
        Points_color_sharp_rgb2 = color(corner_idx,:);
        point_surfaces3 = [down_sample_point2,Points_color_sharp_rgb2]; 
        
        scatter3(point_surfaces3(:,1),point_surfaces3(:,2),point_surfaces3(:,3),500,point_surfaces3(:,4:6),'.');
        axis equal
        title('Our Edge and Corner points') %,'color','r'?)  %,'FontSize','12'        
        hold off

end


%% 输入图
function fig1_vis(input_points, edge_pro, corner_idx)

        fig1 = figure(1);
        load color
        temp_points = input_points;
        edgepoint_label_pre = edge_pro>0.75;
        All_edge_points_idx = edgepoint_label_pre;
        All_edge_points_idx = logical(All_edge_points_idx);
        All_edge_points = temp_points(logical(All_edge_points_idx),:);

        temp_color_label = zeros(size(input_points,1),1);
        All_edge_points_color_idx = temp_color_label(logical(All_edge_points_idx))+1;
        All_edge_points_color = color(All_edge_points_color_idx,:);     
        temp_all = [All_edge_points,All_edge_points_color];
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.');
        
        hold on

        down_sample_point2 = input_points(corner_idx,:);
        Points_color_sharp_rgb2 = color(corner_idx,:);
        point_surfaces3 = [down_sample_point2,Points_color_sharp_rgb2]; 
        
        scatter3(point_surfaces3(:,1),point_surfaces3(:,2),point_surfaces3(:,3),500,point_surfaces3(:,4:6),'.');
        axis equal
        title('Our Edge and Corner points') %,'color','r'?)  %,'FontSize','12'        
        hold off

end

% %% 连通图
% function fig2_vis(input_points, train_all_pair, train_all_pair_con)
%         
%         train_all_pair_pos_idx = find(train_all_pair_con);
%         train_all_pair_pos_pair = train_all_pair(train_all_pair_pos_idx,:);
%         
%         fig2 = figure(2);
%         load color
%         
%         global_idx_unique = [train_all_pair_pos_pair(:,1); train_all_pair_pos_pair(:,2)];
%         global_idx_unique = unique(global_idx_unique);
%         feature_points = input_points(global_idx_unique,:);
%         temp_points = feature_points;
%         color_label = find(global_idx_unique);
%         temp_color = color(color_label,:);  
%         temp_all = [temp_points,temp_color];
%         scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.');       
%         hold on
%                 
%         axis equal
%         title('GT Edge and Corner points') %,'color','r'?)  %,'FontSize','12'        
%         hold on
%         
%         num_edge = size(train_all_pair_pos_pair,1);
%         for i_edge = 1:num_edge
%             edge_l = input_points(train_all_pair_pos_pair(i_edge,1),:);
%             edge_r = input_points(train_all_pair_pos_pair(i_edge,2),:);
%             line([edge_l(1);edge_r(1)],[edge_l(2);edge_r(2)],[edge_l(3);edge_r(3)]);
%             hold on
%         end
%         hold off
% end 



function Dis_matrix = Distance_Points1_Points2_matrix(vertices_ball,vertices_points)

B=vertices_ball; 
P=vertices_points; 
B1=sum(B.^2,2);
P1=sum(P.^2,2);
Num_b=numel(B1);
Num_p=numel(P1);
Dis_matrix= repmat(B1,1,Num_p) + repmat(P1',Num_b,1) - 2*B*P'; 

end
