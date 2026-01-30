function [T,M_num] = LPO_DeformField(Tn,s1_shape,s2_shape,neighbors1,neighbors2, k2,lambda)

% Here lambda = (1-lambda in paper)/2

if nargin < 7
    lambda = 0.15;
end

tol=1e-5; theta = 0.95; maxiter=10;

vector1 = s1_shape.surface.VERT;
vector2 = s2_shape.surface.VERT;

M_lp = LoPres_filter_v1(neighbors1,neighbors2,Tn(:,1),lambda);
% rgb=coord2rgb([s2_shape.surface.X,s2_shape.surface.Y,s2_shape.surface.Z]);
% rgb(M_lp(:,1),:) = ones(size(M_lp,1),1)*[255,255,255]; T = Tn(:,1);
% mplot_mesh_rgb([s1_shape.surface.X,s1_shape.surface.Y,s1_shape.surface.Z],s1_shape.surface.TRIV,rgb(T,:));
% title('Inliers-LPO-0.15');

% M_lp = LoPres_filter_v1(neighbors1,neighbors2,Tn(:,1),0.30);
% rgb=coord2rgb([s2_shape.surface.X,s2_shape.surface.Y,s2_shape.surface.Z]);
% rgb(M_lp(:,1),:) = ones(size(M_lp,1),1)*[255,255,255]; T = Tn(:,1);
% mplot_mesh_rgb([s1_shape.surface.X,s1_shape.surface.Y,s1_shape.surface.Z],s1_shape.surface.TRIV,rgb(T,:));
% title('Inliers-LPO-0.30');

M_num=size(M_lp,1); 
if M_num==0
    T=Tn(:,1);
    return
end

N = size(vector1,1);
P = zeros(size(vector1,1),1); P(M_lp(:,1)) = 1;
P = spdiags(P, 0, N, N);

% compute weight matrix
iter=1;
while iter<=maxiter
    Tau1 = pinv(s1_shape.evecs(M_lp(:,1),:))*(vector2(M_lp(:,2),:)-vector1(M_lp(:,1),:));
    Deform1 = vector1+s1_shape.evecs*Tau1;
    
    E = vector2(Tn(:,1),:) - Deform1;
    sigma2 = trace(E'*P*E)/(3*trace(P));
    numcorr = length(find(P > theta));
    gamma = numcorr/size(vector1,1);
    
    temp1 = exp(-sum(E.^2,2)/(2*sigma2));
    temp2 = (2*pi*sigma2)^(3/2)*(1-gamma)/(gamma*20);
    P = temp1./(temp1+temp2); 
    P = spdiags(P, 0, N, N);
    index2 = find(diag(P)>theta);
    if gamma > 0.95, gamma = 0.95; end
    if gamma < 0.05, gamma = 0.05; end
    if ~isempty(index2)
        M_lp = [index2, Tn(index2,1)];
    else
        index2 = M_lp(:,1);break;
    end
    iter=iter+1;
end
rem_index1 = 1:s1_shape.nv; rem_index1(index2)=[];

% rgb=coord2rgb([s2_shape.surface.X,s2_shape.surface.Y,s2_shape.surface.Z]);
% rgb(M_lp(:,1),:) = ones(size(M_lp,1),1)*[255,255,255]; T = Tn(:,1);
% mplot_mesh_rgb([s1_shape.surface.X,s1_shape.surface.Y,s1_shape.surface.Z],s1_shape.surface.TRIV,rgb(T,:));
% title('Inliers-DF');

Tau2 = pinv(s1_shape.evecs(index2,:))*(vector2(Tn(index2,1),:)-vector1(index2,:));
Deform2 = vector1+s1_shape.evecs*Tau2;

% T12 = knnsearch(vector2, Deform2); rgb=coord2rgb([s2_shape.surface.X,s2_shape.surface.Y,s2_shape.surface.Z]);
% mplot_mesh_rgb([s1_shape.surface.X,s1_shape.surface.Y,s1_shape.surface.Z],s1_shape.surface.TRIV,rgb(T12,:));
% title('LOPR-Deform2');

M_rem = (zeros(length(rem_index1),2));
k1 = ceil(M_num./100);

neighborhood = knnsearch(gpuArray(vector1(M_lp(:,1),:)),gpuArray(vector1(rem_index1,:)),'k',k1);
neighborhood = gather(neighborhood);

Ineighborhood = reshape(M_lp(neighborhood,1),k1,size(M_rem,1))';
Tneighborhood = reshape(M_lp(neighborhood,2),k1,size(M_rem,1))';
W = zeros(k1,s1_shape.nv);

for i = 1:length(rem_index1)
    
    z = vector1(Ineighborhood(i,:),:) - repmat(vector1(rem_index1(i),:),k1,1);
    G = z*z';
    G = G + eye(k1,k1)* tol * trace(G);
    W(:,i) = G\ones(k1,1);
    W(:,i) = W(:,i)/sum(W(:,i));
    LL_rem = sum(vector2(Tneighborhood(i,:),:)'*W(:,i),2);
    LL_dist = sum((repmat(LL_rem',k2,1) - vector2(Tn(rem_index1(i),:),:)).^2,2);
    
    LL_dist2 = sum((vector2(Tn(rem_index1(i),:),:) -repmat(Deform2(rem_index1(i),:),k2,1)).^2,2);
    if min(LL_dist2)>1
        [~,LL_ind] = min(LL_dist+0*LL_dist2);
    else
        [~,LL_ind] = min(LL_dist+LL_dist2);
    end
    M_rem(i,:) = [rem_index1(i), Tn(rem_index1(i),LL_ind)];
    
end

M = sortrows([M_lp;M_rem]); 
T =M(:,2);

end
