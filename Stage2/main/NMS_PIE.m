function pick = NMS_PIE(cycles, scores, overlap_the)    
  
if isempty(cycles)  
  pick = [];  
else       
  [vals, I] = sort(scores,'descend');      
  pick = [];  
  while ~isempty(I)  
    last = length(I);       
    i = I(last);          
    pick = [pick; i];      
    suppress = [last];      
    for pos = 1:last-1      
      j = I(pos);          
      points_idx_1 = cycles(i,:); 
      points_idx_2 = cycles(j,:);
      o = IOU_PIE(points_idx_1', points_idx_2');
        if o > overlap_the     
          suppress = [suppress; pos];   
        end   
    end  
    I(suppress) = [];
  end    
end

end

function overlap =  IOU_PIE(points_idx_1, points_idx_2)  % col vector

All_combine = Free_combine(points_idx_1, points_idx_2);
com_reduce = All_combine(:,1) - All_combine(:,2);

[idx,~] = find(com_reduce == 0);

temp = All_combine(:,1);
inter_eles = unique(temp(idx));

if isempty(inter_eles)
    overlap = 0;
else
    all_elements = [points_idx_1; points_idx_2];
    All_combine_2 = Free_combine(inter_eles, all_elements);
    com_reduce_2 =  All_combine_2(:,1) - All_combine_2(:,2);
    inter_sum = length(find(com_reduce_2==0));

    ele_sum = numel(all_elements);
    overlap = inter_sum/ele_sum;
end
    
end


function All_combine = Free_combine(idx_set_1, idx_set_2)
   num_set_1 = numel(idx_set_1);
   num_set_2 = numel(idx_set_2);
   
   if (num_set_1 ==0 || num_set_2 ==0)
      All_combine = [];
   else
       set1_mat = repmat(idx_set_1, numel(idx_set_2),1); %[m*n,1]
       set2_mat = repmat(idx_set_2, 1,numel(idx_set_1));
       [f_x,f_y]=size(set2_mat);
       set2_mat_vector = reshape(set2_mat',1,f_x*f_y); %[1,m*n]
       set2_mat_vector = set2_mat_vector'; % [m*n,1]
       All_combine = [set1_mat,set2_mat_vector];
   end
end

