function Vis

FindFiles = '.\test_result\';

Files = dir(fullfile(FindFiles));
filenames = {Files.name}';
filenames = filenames(3:length(filenames));
filenames=filenames';

load color
 
 for i = 1:1 %length(filenames)
    load([FindFiles,filenames{i}]);
    [num_data, num_points,~] = size(input_point_cloud);
    for j =1:num_data        
       %% gt 8096 points
        input_points = squeeze(input_point_cloud(j,:,:));
 
%%  figure 1 : input point cloud       
        fig_1 = figure(1);

        temp_points = input_points;
        temp_color = color(1,:);  
        temp_color = repmat(temp_color,num_points,1);
        temp_all = [temp_points,temp_color];
        

        ALL_LINE.num_lines = 0;
        ALL_LINE.lines = {};
        ALL_LINE.lines_label = {};
        ALL_LINE.lines_size = {};

        ALL_POINT_SET.num_point_set = 1;
        ALL_POINT_SET.point_set = {};		
        ALL_POINT_SET.point_color_label = {6};
        ALL_POINT_SET.point_size = {5};				

        ALL_POINT_SET.point_set = [ALL_POINT_SET.point_set,temp_points];
        
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.'); 
        axis equal
        axis off
        title('Input Point Cloud');


%%  figure 2 :  prediction edge and corner points       
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
        
        fig_2 = figure(2)

        temp_points = input_points;
        All_edge_points_idx = edgepoint_label_pre;
        All_edge_points_idx = logical(All_edge_points_idx);
        All_edge_points = temp_points(logical(All_edge_points_idx),:);

        ALL_LINE.num_lines = 0;
        ALL_LINE.lines = {};
        ALL_LINE.lines_label = {};
        ALL_LINE.lines_size = {};

        ALL_POINT_SET.num_point_set = 2;
        ALL_POINT_SET.point_set = {};		
        ALL_POINT_SET.point_color_label = {5,4};
        ALL_POINT_SET.point_size = {5,15};				

        ALL_POINT_SET.point_set = [ALL_POINT_SET.point_set,All_edge_points];         
        

        temp_color_label = zeros(size(input_points,1),1)+2;
        All_edge_points_color_idx = temp_color_label(logical(All_edge_points_idx))+1;
        All_edge_points_color = color(All_edge_points_color_idx,:);     
        temp_all = [All_edge_points,All_edge_points_color];
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.'); 
        hold on
        edge_points_pre1 = squeeze(pred_labels_key_p_val(j,:,:));
        edge_points_pre1 = exp(edge_points_pre1);
        sum_edge_pre1 = sum(edge_points_pre1,2);
        edge_points_pre1 = edge_points_pre1./repmat(sum_edge_pre1,1,2);
        edgepoint_label_pre1 = edge_points_pre1(:,2)>0.75;
        All_edge_points_idx1 = edgepoint_label_pre1;
        All_edge_points_idx1 = logical(All_edge_points_idx1);
        edge_int = int16(All_edge_points_idx1);
        
        corner_points_label = corner_label_pre;
        corner_int = int16(corner_points_label)*2;
        conrner_edge_intersection = corner_int - edge_int;
        
        
        down_sample_point2 = input_points(conrner_edge_intersection == 1,:);
        color_label2 = find(conrner_edge_intersection == 1);
        Points_color_sharp_rgb2 = color(color_label2,:);
        point_surfaces3 = [down_sample_point2,Points_color_sharp_rgb2];

        ALL_POINT_SET.point_set = [ALL_POINT_SET.point_set,down_sample_point2]; 
        
        scatter3(point_surfaces3(:,1),point_surfaces3(:,2),point_surfaces3(:,3),500,point_surfaces3(:,4:6),'.');
        axis equal
        title('Our Edge and Corner points') %,'color','r'?)  %,'FontSize','12'        
        hold off


%%  figure 3 :  prediction edge and corner points (NMS)
        % pre edge points and corner points
        max(input_points)
        scale = 0.5./max(input_points);
        mat_scale = [scale(1),0,0;0,scale(2),0,;0,0,scale(3)];
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
        local_max_pro_idx = Local_Maximum_Filter(corner_pre_pro,corner_pre_points); 
        
        global_idx = corner_label_pre_idx(local_max_pro_idx);
        corner_label_filter = zeros(length(corner_label_pre),1);
        corner_label_filter(global_idx) = 1;
        
        
        
        fig_3 = figure(3)
        
        temp_points = input_points;
        All_edge_points_idx = edgepoint_label_pre;
        All_edge_points_idx = logical(All_edge_points_idx);
        All_edge_points = temp_points(logical(All_edge_points_idx),:);

        All_edge_points_color_idx = temp_color_label(logical(All_edge_points_idx))+1;
        All_edge_points_color = color(All_edge_points_color_idx,:);     
        temp_all = [All_edge_points,All_edge_points_color];
        
        ALL_LINE.num_lines = 0;
        ALL_LINE.lines = {};
        ALL_LINE.lines_label = {};
        ALL_LINE.lines_size = {};

        ALL_POINT_SET.num_point_set = 2;
        ALL_POINT_SET.point_set = {};		
        ALL_POINT_SET.point_color_label = {5,4};
        ALL_POINT_SET.point_size = {5,15};				

        ALL_POINT_SET.point_set = [ALL_POINT_SET.point_set,All_edge_points];        
        
        scatter3(temp_all(:,1),temp_all(:,2),temp_all(:,3),50,temp_all(:,4:6),'.');
        
        hold on
        edge_points_pre1 = squeeze(pred_labels_key_p_val(j,:,:));
        edge_points_pre1 = exp(edge_points_pre1);
        sum_edge_pre1 = sum(edge_points_pre1,2);
        edge_points_pre1 = edge_points_pre1./repmat(sum_edge_pre1,1,2);
        edgepoint_label_pre1 = edge_points_pre1(:,2)>0.75;
        All_edge_points_idx1 = edgepoint_label_pre1;
        All_edge_points_idx1 = logical(All_edge_points_idx1);
        edge_int = int16(All_edge_points_idx1);
        
        corner_points_label = corner_label_filter;
        corner_int = int16(corner_points_label);
        conrner_edge_intersection = corner_int; %- edge_int;
        
        
        down_sample_point2 = input_points(conrner_edge_intersection == 1,:);
        color_label2 = find(conrner_edge_intersection == 1);
        Points_color_sharp_rgb2 = color(color_label2,:);
        point_surfaces3 = [down_sample_point2,Points_color_sharp_rgb2];
        
        ALL_POINT_SET.point_set = [ALL_POINT_SET.point_set,down_sample_point2];


        scatter3(point_surfaces3(:,1),point_surfaces3(:,2),point_surfaces3(:,3),500,point_surfaces3(:,4:6),'.');
        axis equal
        title('Our Edge and Corner points (Local Max Filter)') %,'color','r'?)  %,'FontSize','12'        
        hold off
        

    end
 end
end 

%%
function For_rendering_picture(RENDER,file_name)
 
% open the file with write permission 
fid = fopen(file_name, 'wt');
 % 1, write lines
   ALL_LINE = RENDER{1};   
 % 1.1  num_lines;
   num_lines = ALL_LINE.num_lines;   
 % 1.2  each line
   lines = ALL_LINE.lines;
   lines_label = ALL_LINE.lines_label;
   lines_size = ALL_LINE.lines_size;
   fprintf(fid, '%d', num_lines);
   fprintf(fid, '\n');
   if num_lines>0
       for i = 1:num_lines
          % 1.2.1 num_points_per_line,  line_size, line_color_label
          temp_line = lines{i};
          temp_line_size = lines_size{i};
          temp_line_label = lines_label{i};
          num_points_per_line = size(temp_line,1);
          fprintf(fid, '%d %d %d ', num_points_per_line, temp_line_size, temp_line_label);
          fprintf(fid, '\n');
          % 1.2.2 write n*3 matrix
          for j = 1:num_points_per_line
              % 1.2.2.1 write each point
              fprintf(fid, '%f %f %f ', temp_line(j,1),temp_line(j,2),temp_line(j,3));
              fprintf(fid, '\n');
          end
       end
   end
   
 % 2, write points
 ALL_POINT_SET = RENDER{2};
 % 2.1 num_point_set
   num_point_set = ALL_POINT_SET.num_point_set;
 % 2.2 each_point_set
   point_set = ALL_POINT_SET.point_set;		
   point_color_label = ALL_POINT_SET.point_color_label;
   point_size = ALL_POINT_SET.point_size;
   fprintf(fid, '%d', num_point_set);
   fprintf(fid, '\n');
   if num_point_set>0
     for i = 1:num_point_set
       % 2.2.1 num_points_per_set, point_size, point_color_label
       temp_point_set = point_set{i};
       temp_point_size = point_size{i};
       temp_point_color_label = point_color_label{i};
       num_points_per_set = size(temp_point_set,1); 
       fprintf(fid, '%d %d %d ', num_points_per_set, temp_point_size, temp_point_color_label);
       fprintf(fid, '\n');
       % 2.2.2 write n*3 matrix
       for j = 1:num_points_per_set
           % 2.2.2.1 write each point
           fprintf(fid, '%f %f %f ', temp_point_set(j,1),temp_point_set(j,2),temp_point_set(j,3));
           fprintf(fid, '\n');
       end
     end
   end
 fclose(fid);  
end

function dis = distance(a,b)
    A_B = a - b;
    dis = sum(A_B.^2);
end

function pick = nms_corner(boxes, overlap)
% top = nms(boxes, overlap)
% Non-maximum suppression. (FAST VERSION)
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected
% detection.
%
% NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
% but an inner loop has been eliminated to significantly speed it
% up in the case of a large number of boxes

% Copyright (C) 2011-12 by Tomasz Malisiewicz
% All rights reserved.
% 
% This file is part of the Exemplar-SVM library and is made
% available under the terms of the MIT license (see COPYING file).
% Project homepage: https://github.com/quantombone/exemplarsvm


if isempty(boxes)
  pick = [];
  return;
end


x1 = boxes(:,1);
y1 = boxes(:,2);
x2 = boxes(:,3);
y2 = boxes(:,4);
s = boxes(:,end);

area = (x2-x1+1) .* (y2-y1+1);
[vals, I] = sort(s);

pick = s*0;
counter = 1;
while ~isempty(I)
  last = length(I);
  i = I(last);  
  pick(counter) = i;
  counter = counter + 1;
  
  xx1 = max(x1(i), x1(I(1:last-1)));
  yy1 = max(y1(i), y1(I(1:last-1)));
  xx2 = min(x2(i), x2(I(1:last-1)));
  yy2 = min(y2(i), y2(I(1:last-1)));
  
  w = max(0.0, xx2-xx1+1);
  h = max(0.0, yy2-yy1+1);
  
  inter = w.*h;
  o = inter ./ (area(i) + area(I(1:last-1)) - inter);
  
  I = I(find(o<=overlap));
end

pick = pick(1:(counter-1));

end

function local_max_pro_idx= Local_Maximum_Filter(corner_pre_pro,corner_pre_points) %local_max_pro_idx
 
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
 
 
 
 
 
 
 
 
 
 
 
 
 
 
        