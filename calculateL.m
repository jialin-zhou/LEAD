function [LA]=calculateL(A)
% DA
% LA
n = size(A,1);
DA = zeros(n,n);
for i = 1:n
    tmp = 0;
    for j = 1:n
        tmp = tmp + (A(i,j) + A(j,i))/2;
    end
    DA(i,i) = tmp;
end
LA = DA - (A + A.')/2;
end