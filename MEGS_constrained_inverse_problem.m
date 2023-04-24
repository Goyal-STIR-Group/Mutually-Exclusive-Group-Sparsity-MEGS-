% Example showing how to solve a generic MEGS-constrained inverse problem
% argmin_x 0.5||A*x - y||_2^2  s.t. |x|^T S|x| = 0

% Make a random sparsity matrix
S = round(makeSparseS(60,0.5));
Sn = 1 - S;

% Make a ground truth vector that fits the sparsity structure
true_x = makeVec(S);

% Make a random compressive matrix
A = randn(30,length(S));

% Noisy measurement
y = awgn(A*true_x,25,'measured');

%% Solve MEGS-constrained problem to recover true_x
% Uses a proximal subgradient algorithm outlined in the paper

pA = pinv(A'*A)*A';

x_MEGS = pA*y;
x_pi = x_MEGS;
step_size = 0.0005;
x_MEGS = pA*y;
lambda = 0.01;

for it = 1:1000
    x_MEGS = prox_l12(x_MEGS - step_size*(A'*(A*x_MEGS-y) + lambda*(- 2*Sn*abs(x_MEGS).*sign(x_MEGS))), step_size*lambda); 
    lambda = lambda + step_size*abs(x_MEGS)'*S*abs(x_MEGS);
    cost(it) = 0.5*norm(A*x_MEGS-y)^2 + abs(x_MEGS)'*S*abs(x_MEGS);
    step_size = step_size*0.99975;
end

%% Plot results
figure
plot(true_x)
hold on
plot(x_MEGS)
plot(x_pi)

legend('Ground truth','Recovered','Pseudoinverse recovery')

