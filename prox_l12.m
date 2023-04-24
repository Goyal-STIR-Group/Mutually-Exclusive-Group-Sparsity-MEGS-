% Proximal operator for l12 norm
function [p] = prox_l12(x,lambda)
    lambda = lambda*2;
    y = sort(abs(x),'descend');
    ysum = cumsum(y);
    
    jmax = length(y);
    tau = ysum(end)*lambda/(1+jmax*lambda);
    
    for j = 1:length(y)
       tau = ysum(j)*lambda/(1+j*lambda);
       if y(j) - tau<=0
           p = max(abs(x) - lt, 0).*sign(x);
           return;
       end
       lt = tau;
    end
    
    p = max(abs(x) - ysum(end)*lambda/(1+lambda*length(y)), 0).*sign(x);
end

