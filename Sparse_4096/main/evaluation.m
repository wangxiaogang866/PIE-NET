function evaluation2
filepath = 'test_result_s_3';
file = dir(fullfile(filepath,'*.mat'));
filenames = {file.name}';
all_p_error = [];
all_d_err = [];
all_epe = [];
all_t_err = [];
all_iou_before = [];
all_iou_after = [];
for i =1:size(filenames,1)
    temp_name = filenames{i};
    load_path = [filepath '/' temp_name];
    load(load_path);
    [p_error,d_err,epe,T_err,temp_iou_1,temp_iou_2] = cellfun(@(x) solve(x),Training_data,'Unif',0);
    all_p_error = [all_p_error;p_error];   % MD
    all_d_err = [all_d_err;d_err];  %OE
    all_epe = [all_epe;epe];  %EPE
    all_t_err = [all_t_err;T_err];  %TA
    all_iou_before = [all_iou_before;temp_iou_1];  % motion part segmentatin IoU before Motion Optimization Network (MON).
    all_iou_after = [all_iou_after;temp_iou_2];  % motion part segmentatin IoU after Motion Optimization Network (MON).
    fprintf([temp_name, '\n']);
    p_error_mean = mean(cell2mat(p_error));
    fprintf('p_error_mean is %f\n',p_error_mean);
    
    d_err_mean = mean(cell2mat(d_err));
    fprintf('d_err_mean is %f\n',d_err_mean);
    
    epe_mean = mean(cell2mat(epe));
    fprintf('epe is %f\n',epe_mean);
    
    T_err_mean = mean(cell2mat(T_err));
    fprintf('T_err_mean is %f\n',T_err_mean);
    
    all_iou_before_mean = mean(cell2mat(temp_iou_1));
    fprintf('all_iou_before_mean is %f\n',all_iou_before_mean);
    
    all_iou_after_mean = mean(cell2mat(temp_iou_2));
    fprintf('all_iou_after_mean is %f\n',all_iou_after_mean);
end

all_p_error_mean = mean(cell2mat(all_p_error));   
fprintf('p_error_mean is %f\n',all_p_error_mean);

all_d_err_mean = mean(cell2mat(all_d_err));
fprintf('d_err_mean is %f\n',all_d_err_mean);

all_epe_mean = mean(cell2mat(all_epe));
fprintf('epe is %f\n',all_epe_mean);

all_t_err_mean = mean(cell2mat(all_t_err));
fprintf('all_t_err_mean is %f\n',all_t_err_mean);

all_iou_before_mean = mean(cell2mat(all_iou_before));
fprintf('all_iou_before_mean is %f\n',all_iou_before_mean);
    
all_iou_after_mean = mean(cell2mat(all_iou_after));
fprintf('all_iou_after_mean is %f\n',all_iou_after_mean);
end

function [p_error,d_err,epe,T_err,iou_1,iou_2] = solve(Training_data)

pred_dof_net = Training_data.dof_pred;
GT_dof = Training_data.GT_dof;
GT_proposal_nx = Training_data.GT_proposal_nx;
proposal = Training_data.proposal;
pred_proposal = Training_data.pred_proposal;
inputs_all = Training_data.inputs_all;

p_error = compute_dis(pred_dof_net(1:3),GT_dof);

d_err1 = compute_D_err(pred_dof_net(4:6),GT_dof(4:6));
d_err2 = compute_D_err(-pred_dof_net(4:6),GT_dof(4:6));
d_err = min(d_err1,d_err2);
epe1 = compute_epe(inputs_all,GT_proposal_nx,GT_dof,pred_dof_net);
epe2 = compute_epe(inputs_all,GT_proposal_nx,GT_dof,[pred_dof_net(1:3) -pred_dof_net(4:6) pred_dof_net(7)]);
epe = min(epe1,epe2);
if (pred_dof_net(7) == GT_dof(7))
    T_err = 1;
else
    T_err = 0;
end

inter = GT_proposal_nx.*proposal;
union = logical(GT_proposal_nx+proposal);
iou_1 = sum(inter)/(sum(union)+1);

inter = GT_proposal_nx.*pred_proposal';
union = logical(GT_proposal_nx+pred_proposal');
iou_2 = sum(inter)/(sum(union)+1);
end

function dof_score = compute_epe(input,proposal,GT_dof,pred_dof)

input = input(proposal==1,1:3);

GT_mat = Rotation3D(GT_dof,input);
temp_pred = Rotation3D(pred_dof,input);
temp = GT_mat-temp_pred;
dof_score = mean(sum(abs(temp).^2,2).^(1/2));
end

function dis = compute_dis(p,dof)
P = p;
Q1 = dof(1:3);
Q2 = dof(1:3)+dof(4:6);
dis = norm(cross(Q2-Q1,P-Q1))./norm(Q2-Q1);
end

function D_err = compute_D_err(x,y)
D_err = acos(dot(x,y)/(norm(x)*norm(y)));
end


function points_rot=Rotation3D(Dof_para,Points)
if Dof_para(7)==1
    theta=pi;
    points_rot=rot3d(Points,Dof_para(1:3),Dof_para(4:6),theta);
elseif Dof_para(7)==2
    scale=1;
    points_rot=trans3d(Points,Dof_para(4:6),scale);
elseif Dof_para(7)==3
    scale=1;
    theta=pi;
    points_rot=trans3d(Points,Dof_para(4:6),scale);
    points_rot=rot3d(points_rot,Dof_para(1:3),Dof_para(4:6),theta);
end
end


function Pr=rot3d(P,origin,dirct,theta)

dirct=dirct(:)/norm(dirct);
A_hat=dirct*dirct';
A_star=[0,         -dirct(3),      dirct(2)
    dirct(3),          0,     -dirct(1)
    -dirct(2),   dirct(1),            0];
I=eye(3);
M=A_hat+cos(theta)*(I-A_hat)+sin(theta)*A_star;
origin=repmat(origin(:)',size(P,1),1);
Pr=(P-origin)*M'+origin;

end

function point_trans=trans3d(point,direct,scale)
direct=direct(:)/norm(direct);
direct1=scale*direct;
point_trans=point+repmat(direct1',size(point,1),1);
end