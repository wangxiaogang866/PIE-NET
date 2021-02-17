function [down_sample_point, down_sample_point_idx] = Farthest_Point_Sampling_piont_and_idx(point_raw,down_sample_num)
point_num = size(point_raw,1);
down_sample_point_idx = zeros(1,down_sample_num);
distance = ones(point_num,1)*Inf;
down_sample_point_idx = Farthest_Point_Sampling_Step(point_raw,point_num,distance,down_sample_point_idx,down_sample_num,0);
down_sample_point = point_raw(down_sample_point_idx,:);
end

function down_sample_point_idx = Farthest_Point_Sampling_Step(point_raw,point_num,distance,down_sample_point_idx,down_sample_num_all,down_sample_num_remain)
if(down_sample_num_remain~=down_sample_num_all)
    if (down_sample_num_remain==0)
        distance = Compute_Dis(point_raw,distance,unidrnd(size(point_raw,1)));
        [~,idx] = max(distance);
        down_sample_num_remain = down_sample_num_remain + 1;
        down_sample_point_idx(down_sample_num_remain) = idx;
        down_sample_point_idx = Farthest_Point_Sampling_Step(point_raw,point_num,distance,down_sample_point_idx,down_sample_num_all,down_sample_num_remain);
    else
        distance = Compute_Dis(point_raw,distance,down_sample_point_idx(down_sample_num_remain));
        [~,idx] = max(distance);
        down_sample_num_remain = down_sample_num_remain + 1;
        down_sample_point_idx(down_sample_num_remain) = idx;
        down_sample_point_idx = Farthest_Point_Sampling_Step(point_raw,point_num,distance,down_sample_point_idx,down_sample_num_all,down_sample_num_remain);
    end
end
end
function distance = Compute_Dis(point_raw,distance,a)
% low speed
% temp = point_raw-repmat(point_raw(a,:),point_num,1);
% temp = temp.*temp;
% temp_distance = temp(:,1)+temp(:,2)+temp(:,3);
% distance = min(distance,temp_distance);

%high speed
%tic;
temp_p = point_raw(a,:);
temp1 = point_raw(:,1)-temp_p(1);
temp2 = point_raw(:,2)-temp_p(2);
temp3 = point_raw(:,3)-temp_p(3);
temp = temp1.*temp1+temp2.*temp2+temp3.*temp3;
distance = min(distance,temp);
%toc;
end