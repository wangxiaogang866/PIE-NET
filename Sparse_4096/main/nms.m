function nms
FindFiles = 'test_data/';
Files = dir(fullfile(FindFiles,'*.mat'));
filenames = {Files.name}';

for i = 1:size(filenames,1)
    temp_name = filenames{i};
    load_path_1 = [FindFiles temp_name];
    load(load_path_1);
    temp_name = temp_name(11:length(temp_name)-4);
    load_path2 = ['./test_pred_' temp_name '.mat'];
    load(load_path2);
    a = size(Training_data,1);
    pred_simmat = mat2cell(pred_simmat_val,ones(a,1));
    pred_conf = mat2cell(pred_conf_logits_val,ones(a,1));
    cellfun(@(x,y,z) compute(x,y,z),pred_simmat,pred_conf,Training_data,'Unif',0);
end
end

function compute(pred_simmat,pred_conf,Training_data)
tic;
pred_simmat = reshape(pred_simmat,4096,4096);
GT_proposal = Training_data.proposal;
GT_proposal(1,:) = [];
s_mat = zeros(4096);
s_mat(pred_simmat<=70)=1;
p_num = sum(s_mat,2);
s_mat(p_num<20,:) = [];
pred_conf(p_num<20) = [];
max_score = pred_conf;
chose_idx = max_score>=0.3;
s_mat(chose_idx==0,:) = [];
max_score(chose_idx==0) = [];
[vals,I] = sort(max_score);
pick = [];
while ~isempty(I)
    last = length(I);
    i = I(last);
    suppress = [last];
    merge = [];
    for pos = 1:last-1
        j = I(pos);
        proposal_1 = s_mat(i,:);
        proposal_2 = s_mat(j,:);
        inter = proposal_1.*proposal_2;
        union = logical(proposal_1+proposal_2);
        overlap = sum(inter)/sum(union);
        if overlap>=0.3
            suppress = [suppress;pos];
        end
        if overlap>=0.90
            merge = [merge;j];
        end
    end
    if size(merge,1)~=0
        pick_num = 0;
        pick_iou = 0;
        for k = 1:size(merge,1)
            t_p = s_mat(merge(k),:);
            t_inter = proposal_1.*t_p;
            iou = sum(t_inter)/sum(t_p);
            if iou>pick_iou
                pick_num = merge(k);
                pick_iou = iou;
            end
        end
    end
    if isempty(merge)==0
        pick = [pick;pick_num];
    else
        pick = [pick;i];
    end
    I(suppress)=[];
end
pick_mat = s_mat(pick,:);

end
