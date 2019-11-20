function [ out ] = HaarT_t( x, lam )

    h1 = lam(1) * [1   1; -1 -1];
    h2 = lam(2) * [1  -1;  1 -1];
    h3 = lam(3) * [1  -1; -1  1];
   
    % Here we assume x is of three dimensional
    % dims refers to the indices of the last dimension
    slice1 = squeeze(x(:,:,1));
    slice2 = squeeze(x(:,:,2));
    slice3 = squeeze(x(:,:,3));

    out =  conv2t(h1,slice1,'full') + conv2t(h2,slice2,'full') + conv2t(h3,slice3,'full');
end