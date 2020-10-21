function visualization()

FindFiles = 'D:\所有资料\NIPS2020\new_experiments\different_noise_test_data_and_pre\new_model\test_data_pre/';

Files = dir(fullfile(FindFiles));
filenames = {Files.name}';
filenames = filenames(3:length(filenames));
filenames=filenames';

load color
 
 for i = 1:length(filenames)
    load([FindFiles,filenames{i}]);
    [num_data, num_points] = size(input_labels_key_p);
    for j =1:num_data        
        %% figure 1: 8096 input points
        input_points = squeeze(input_point_cloud(j,:,:));

        fig_1 = figure(1)
        temp_points = input_points;      
   
        temp_color_label = zeros(size(input_points,1),1)+2;
        All_points_color_idx = temp_color_label+1;
        All_points_color = color(All_points_color_idx,:);     
        temp_all = [temp_points,All_points_color];
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.'); 

        axis equal
        title('Input Point Clouds')       
        hold off


       %%  figure 2 :  prediction edge points
        % pre edge points
        input_points1 = input_points;
        
        edge_points_pre = squeeze(pred_labels_key_p_val(j,:,:));
        edge_points_pre = exp(edge_points_pre);
        sum_edge_pre = sum(edge_points_pre,2);
        edge_points_pre = edge_points_pre./repmat(sum_edge_pre,1,2);
        edgepoint_label_pre = edge_points_pre(:,2)>0.7;
        
        corner_points_pre = squeeze(pred_labels_corner_p_val(j,:,:));
        corner_points_pre = exp(corner_points_pre);
        sum_pre = sum(corner_points_pre,2);
        corner_points_pre = corner_points_pre./repmat(sum_pre,1,2);
        corner_label_pre = corner_points_pre(:,2)>0.95;
        corner_label_pre_idx = find(corner_points_pre(:,2)>0.95);
        
        corner_pre_pro = corner_points_pre(:,2);
        corner_pre_pro = corner_pre_pro(corner_label_pre_idx,:);
        corner_pre_points = input_points1(corner_label_pre_idx,:);
        local_max_pro_idx = Corner_NMS(corner_pre_pro,corner_pre_points); 
        
        global_idx = corner_label_pre_idx(local_max_pro_idx);
        corner_label_filter = zeros(length(corner_label_pre),1);
        corner_label_filter(global_idx) = 1;
                
        fig_2 = figure(2)
        
        temp_points = input_points;
        All_edge_points_idx = edgepoint_label_pre;
        All_edge_points_idx = logical(All_edge_points_idx);
        All_edge_points = temp_points(logical(All_edge_points_idx),:);

        All_edge_points_color_idx = temp_color_label(logical(All_edge_points_idx))+1;
        All_edge_points_color = color(All_edge_points_color_idx,:);     
        temp_all = [All_edge_points,All_edge_points_color];     
        
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.');
        
        hold on
        axis equal
        title('Our Edge points')       
        hold off
    end
 end
end 


function local_max_pro_idx = Corner_NMS(corner_pre_pro,corner_pre_points)
 
 %1, distance between all corner_pre_points
 dis = Distance_Points1_Points2(corner_pre_points,corner_pre_points);
 
 %2, threshold t = 0.01;
 t = 0.006;
 dis_t = dis;
 dis_t(dis_t<=t)= 0;
 dis_t(dis_t>t)= 1;
 dis_t = 1-dis_t;
 
 %3, corner_pre_pro.*
 prob_mask = repmat(corner_pre_pro',length(corner_pre_pro),1);
 dis_and_prob = dis_t.*prob_mask;
 
 %4, sort and to find the maximum index;
 [~,Map_idx_1] = sort(dis_and_prob,2);
 max_idx = Map_idx_1(:,end);
 
 %5, compare with 1:length(corner_pre_pro). judge whether this point is
 %local maximum probability corner point;
 % if yes, then to remain; else no, then to delete this point;
 idx_num = 1:length(corner_pre_pro);
 idx_num = idx_num';
 max_idx = max_idx - idx_num;
 
 local_max_pro_idx = find(max_idx == 0);
 
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
        