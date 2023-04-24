% Makes a vector x that ensures |x|^T S |x| is approx 0
function [vec] = makeVec(S)
    vec = randn(length(S),1);
    
    while abs(vec)'*S*abs(vec)>1e-8
        vec = proxMEGS(vec,1,S);
    end
    
    vec(abs(vec)<1e-6) = 0;
    vec(vec~=0) = (0.5+rand(length(vec(vec~=0)),1)/2).*sign(randn(length(vec(vec~=0)),1));
end

