function [dist] = hausdorff(A, B) 
if(size(A,2) ~= size(B,2)) 
    fprintf( 'WARNING: dimensionality must be the same\n' ); 
    dist = []; 
    return; 
end
dist = max(compute_dist(A, B), compute_dist(B, A));
end


function dist = compute_dist(A, B) 
    dist_matrix = Distance_Points1_Points2(A,B);
    dist_vector = min(dist_matrix,[],2);
    dist = max(dist_vector);
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






