function [S] = makeSparseS(Ssize,sparse)
% Function to create a random sparsity structure matrix S with dimensions Ssize*Ssize and
% sparsity level of 0-1

    S = zeros(Ssize);
    ind = [];
    for i = 1:Ssize
        for j = i+1:Ssize
            ind = [ind; sub2ind(size(S), i, j)];
        end
    end
    
    ind = ind(randperm(length(ind)));
    S(ind(1:round(sparse*(Ssize^2)/2))) = ones(length(1:round(sparse*(Ssize^2)/2)),1);
    for i = 1:Ssize
        S(i,i) = 0;
    end
    
    S = S+S';
end

