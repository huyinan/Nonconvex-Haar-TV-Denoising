function [x, cost] = prox_Haar_ncvx(y, bfmu, pen, Nit, a)
% [x, cost] = TVD_ncvx(y, lam, pen, Nit)
% Total variation denoising with non-convex penalty.
%
% INPUT
%   y - noisy signal
%   lam - regularization parameter (lam > 0)
%   pen - penalty ('log', 'atan', 'L1','rat','expo','mcp')
%   Nit - number of iterations
%
% OUTPUT
%   x - denoised signal
%   cost - cost function history

% Reference: 
% Convex 1-D Total Variation Denoising with Non-convex Regularization
% Ivan W. Selesnick, Ankit Parekh, and Ilker Bayram
% IEEE Signal Processing Letters, 2014

cost = zeros(1, Nit);                                 % Cost function history
[N,M] = size(y);

if nargin < 5
    a = 0.9 / (16 * mean(bfmu));
end

switch pen
    case 'L1'
        
        phi = @(x) abs(x);
        dphi = @(x) sign(x);
        wfun_inv = @(x) abs(x);
        
    case 'log'
        
        phi = @(x) 1/a * log(1 + a*abs(x));
        dphi = @(x) 1 ./(1 + a*abs(x)) .* sign(x);
        wfun_inv = @(x) abs(x) .* (1 + a*abs(x));
        prox_pen = @(x) prox_log(x,bfmu,a);
        
    case 'atan'
        
        phi = @(x) 2./(a*sqrt(3)) .* (atan((2*a.*abs(x)+1)/sqrt(3)) - pi/6);
        wfun_inv = @(x) abs(x) .* (1 + a.*abs(x) + a.^2.*abs(x).^2);
        dphi = @(x) 1 ./(1 + a*abs(x) + a.^2.*abs(x).^2) .* sign(x);
    
    case 'rat'
        phi = @(x) abs(x)./(1+a*abs(x)/2);
        wfun_inv = @(x) abs(x) .* (1 + a*abs(x)/2).^2;
        
    case 'expo'
        phi = @(x) 1/a * (1 - exp(- a * abs(x)));
        wfun_inv = @(x) exp(a * abs(x)) .* x;
        
    case 'mcp'        
        phi = @(t) (abs(t) - a/2 * t.^2).*(abs(t) <= 1/a) + 1/(2*a) * (abs(t) > 1/a);
        wfun_inv = @(x) abs(x) ./ max(1 - a*abs(x), 0); 

end

H_mat = Generate_matrix_H(y,bfmu);
H_mat = sparse(H_mat);
HHT = H_mat * H_mat.';       % H*H' (banded matrix)
H = @(x) HaarT(x,bfmu);                                     % H (operator)
HT = @(x) HaarT_t(x,bfmu);                  % H'

x = y;      % Initialization
Hx = vec(H(x));
Hy = vec(H(y));
vec_y = vec(y);
x_prev = x;

for k = 1:Nit
    Hx = Hx ;
    F = spdiags(wfun_inv(Hx)+ 1e-7, 0, (N-1)*(M-1)*3, (N-1)*(M-1)*3) + HHT;     % F : Sparse matrix structure
    FHy = reshape(F\Hy,N-1,M-1,3);
    vec_HT = vec(HT(FHy));
    vec_x = vec_y - vec_HT;  
    x = reshape(vec_x,N,M);    
    Hx = vec(H(x));
    cost(k) = 0.5*sum(vec(x-y).^2) + sum(abs(phi(Hx)));      % Save cost function history
   if (norm(vec(x-x_prev),2)/norm(vec(x),2) < 1e-4)
        fprintf('f\n');
        break;
    else
        x = x_prev;
    end
end
    x = max(min(x,255),0);
end
