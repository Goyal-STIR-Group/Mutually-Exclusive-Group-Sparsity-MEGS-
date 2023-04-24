% Solves the proximal operator of MEGS |x|^T S |x| using an iterative
% method
function [u] = proxMEGS(z,lambda,S)
    u = (z);
    stepsize = 0.002;
    tol = 1e-5;
    c = 2;
    
    SS = ((S+S')/2);
    
    for it = 1:750
        grad = 2*(u-z) + lambda*(SS)*abs(u).*sign(u);
        tmp = u - stepsize*grad;
        tmp(u<0) = min(tmp(u<0),0);
        tmp(u>0) = max(tmp(u>0),0);
        u = tmp;
        
        % Early stopping
        if mod(it,5) == 0
            cost(c) = norm(u-z)^2 + abs(u)'*S*abs(u);
            c=c+1;
            
            if it>10
                if abs(cost(c-1) - cost(c-2))/(5*cost(c-2)) < tol
                    return;
                end
            end
        end
        
    end
end

