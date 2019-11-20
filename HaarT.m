function [ g ] = HaarT( x, lam)
    
    % x is 2-dimensional, and g is 3-dimensional
h1 = lam(1) * [1  1;-1 -1];
h2 = lam(2) * [1 -1; 1 -1];
h3 = lam(3) * [1 -1;-1  1]; 

a1 = conv2(x,h1,'valid');
a2 = conv2(x,h2,'valid');
a3 = conv2(x,h3,'valid');

[row,col] = size(a1);
spin = 3;

g = zeros(row,col,spin);

g(:,:,1) = a1;
g(:,:,2) = a2;
g(:,:,3) = a3;

end