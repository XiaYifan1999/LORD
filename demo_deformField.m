clc;clear;close all;

addpath(genpath('./'));

%% 

file1 = './data/mesh020.ply';
file2 = './data/mesh070.ply';

k = 500;
S1 = MESH.preprocess(file1, 'IfComputeLB', true, 'numEigs', k,'IfFindNeigh',true,'IfFindEdge',true,'IfComputeGeoDist',true,'IfComputeNormals',true);
S2 = MESH.preprocess(file2, 'IfComputeLB', true, 'numEigs', k,'IfFindNeigh',true,'IfFindEdge',true,'IfComputeGeoDist',true,'IfComputeNormals',true);

[S1, S2] = surfaceNorm(S1, S2);

S1.area = sum(calc_tri_areas(S1.surface));
S2.area = sum(calc_tri_areas(S2.surface));

GT = 1:S1.nv;
% GT1 = load('david0.gt');
% GT2 = load('david1.gt');
% GT = nan(length(GT1),1);
% for ij=1:length(GT1)
%     ind= find(GT2==GT1(ij));
%     if ~isempty(ind)
%         GT(ij)=ind(1);
%     end
% end
GT = fast_pMap_NNinterp(GT, S1);
% GT2 = load('../Dataset/TOPKIDS/low resolution/kid05_ref.txt'); GT2 = GT2(:,2);
% GT1 = 1:S1.nv; 
% GT = zeros(length(GT1),1);
% for ij=1:length(GT1)
%     [~,GT(ij)] = min(abs(GT2-GT1(ij)));
% end

D = S2.Gamma;

%% SHOT matching
opts.shot_num_bins = 10;
opts.shot_radius = 5;
shot1 = calc_shot(S1.surface.VERT', S1.surface.TRIV', 1:S1.nv, opts.shot_num_bins, opts.shot_radius*sqrt(S1.area)/100, 3)';
shot2 = calc_shot(S2.surface.VERT', S2.surface.TRIV', 1:S2.nv, opts.shot_num_bins, opts.shot_radius*sqrt(S2.area)/100, 3)';
T0 = knnsearch(shot2, shot1, 'NSMethod','kdtree'); 
T0_21 = knnsearch(shot2, shot1, 'NSMethod','kdtree'); 

rgb=coord2rgb([S1.surface.X,S1.surface.Y,S1.surface.Z]);
mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb); title('Source')
rgb=coord2rgb([S2.surface.X,S2.surface.Y,S2.surface.Z]);
mplot_mesh_rgb([S2.surface.X,S2.surface.Y,S2.surface.Z],S2.surface.TRIV,rgb); title('Reference')
mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(GT,:)); title('Ground Truth')
mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(T0,:));title('Shot Matching')

%% Discrete Optimization
% T21 = T0_21; B1 = S1.evecs(:,1:500);  B2 = S2.evecs(:, 1:500);
% C12 = B2\B1(T21,:);
% T21 = knnsearch(B1*C12', B2);
% for k = 400:20:500
%     B11 = S1.evecs(:,1:k); B22 = S2.evecs(:, 1:k);
%     Ev11 = S1.evals(1:k);  Ev22 = S2.evals(1:k);
%     Ev11 = Ev11/sum(Ev11); Ev22 = Ev22/sum(Ev22); % normalize the delta to enforce the isometry
%     for iter = 1:5
%         C12 = B22\B11(T21,:);
%         T21 = knnsearch(B11*diag(Ev11)*C12', B22*diag(Ev22));
%         
%         C12 = B22\B11(T21,:);
%         T21 = knnsearch(B11*C12', B22);
%     end
% end
% T = knnsearch(gpuArray(B2),gpuArray(B1*C12'));
% Tdiscreteop=gather(T);

%% DiscreteOp - LORD
% T21 = T0_21; B1 = S1.evecs(:,1:500);  B2 = S2.evecs(:, 1:500);
% C12 = B2\B1(T21,:);
% T21 = knnsearch(B1*C12', B2);
% for k = 400:25:500
%     B11 = S1.evecs(:,1:k); B22 = S2.evecs(:, 1:k);
%     Ev11 = S1.evals(1:k);  Ev22 = S2.evals(1:k);
%     Ev11 = Ev11/sum(Ev11); Ev22 = Ev22/sum(Ev22); % normalize the delta to enforce the isometry
%     for iter = 1:5
%         C12 = B22\B11(T21,:);
%         T21 = knnsearch(gpuArray(B11*diag(Ev11)*C12'), gpuArray(B22*diag(Ev22)));
%         T21 = gather(T21);
%         C12 = B22\B11(T21,:);
%         T21 = knnsearch(gpuArray(B11*C12'), gpuArray(B22));
%         T21=gather(T21);
%         [T21,~] = LPO_DeformField(T21,S2,S1,S2.vtx_neigh,S1.vtx_neigh,10,0.1);
%     end
% end
% T = knnsearch(gpuArray(B2),gpuArray(B1*C12'));
% TdiscreteopLORD=gather(T);

%% LOPR 
Nf = 5;
max_iters = 3; 

[Tlopr, ~] = MWP_LOPR(S1, S2, 500, T0, Nf, max_iters,0.2);

mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(Tlopr,:)); title('LOPR ');

%% LOPR - DeformField
Nf = 5;
max_iters = 3; 

[Tlopr_DF1, ~] = MWP_LOPR_DeformField(S1, S2, 500, T0, Nf, max_iters,0.15);

mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(Tlopr_DF1,:)); title('LOPR-DeformField');

%% LOPR - DeformField
Nf = 5;
max_iters = 3; 

[Tlopr_DF2, ~] = MWP_LOPR_DeformField(S1, S2, 500, T0, Nf, max_iters,0.3);

mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(Tlopr_DF2,:)); title('LOPR-DeformField');

%% MWP
% f = 5;
% num_iters = 3;
% 
% [Tmwp, ~] = MWP2(S1, S2, 500, T0, f, num_iters);
% 
% mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(Tmwp,:)); title('MWP');

%% Sinkhorn
% f = 5;
% num_iters = 5;
% 
% [Tsinkhorn,~]=MWP_SinkHorn(S1,S2,T0,f,num_iters);
%
% mplot_mesh_rgb([S1.surface.X,S1.surface.Y,S1.surface.Z],S1.surface.TRIV,rgb(Tsinkhorn,:)); title('Sinkhorn');

%% ICP
% B1 = S1.evecs(:,1:k); Ev1 = S1.evals(1:k);
% B2 = S2.evecs(:,1:k); Ev2 = S2.evals(1:k);
% num_iters = 10;
% C_fmap=S1.evecs\S2.evecs(T0,:);
% C = fMAP.icp_refine(B2,B1,C_fmap,num_iters);
% T_ICP = fMAP.fMap2pMap(B2,B1,C);

%% PC - Gau_STAG

% S1.surface.n = S1.nv; S1.surface.m = S1.nf;
% S2.surface.n = S2.nv; S2.surface.m = S2.nf;
% 
% S1.surface = mesh.transform.normalize(S1.surface);
% [S1.surface.S, ~, S1.surface.A] = mesh.proc.FEM_higher(S1.surface, 1, 'Dirichlet');
% 
% S2.surface = mesh.transform.normalize(S2.surface);
% [S2.surface.S, ~, S2.surface.A] = mesh.proc.FEM_higher(S2.surface, 1, 'Dirichlet');
% 
% S1.Gamma = MESH.compute_geodesic_dist_matrix(S1);
% S2.Gamma = MESH.compute_geodesic_dist_matrix(S2);

%% 
% n_points = 1000;
% sigma = 0.05;
% para.n_atoms = 500; 
% S1.our_basis = compute_PC_Gau(S1.surface, n_points, sigma, true, para,S1.Gamma);
% S2.our_basis = compute_PC_Gau(S2.surface, n_points, sigma, true, para,S2.Gamma);
% 
% k = 500;
% C_ours = S1.our_basis(:, 1:k)\S2.our_basis(Tmwp, 1:k);  
% T_PC_GAU = fMAP.fMap2pMap(S2.our_basis(:,1:k), S1.our_basis(:,1:k),C_ours);

% Plot
figure;
plot_err_curve(1:S1.nv,Tlopr_DF1,GT,D);hold on;
plot_err_curve(1:S1.nv,Tlopr_DF2,GT,D);hold on;
legend('LORD 0.15','LORD 0.3','Location','southeast');hold off;
