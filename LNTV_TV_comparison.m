% Denoising a noisy piecewise constant signal with LN-TV or regular TV 
rng(999999)

% Make an arbitrary noisy piecewise constant signal y
% Ground truth
x = zeros(100,1);
x(10:40) = 1;
x(60:80) = -1;
x(80:100) = -2;

% Noisy
y = awgn(x,10,'measured');

% Construct finite differencing matrix
D = diag(ones(1,length(x)),0) - diag(ones(1,length(x)-1),1);
D(end,:) = [];
D = sparse(D);

%% LN-TV MEGS
% Create structured sparsity matrix for local-neighborhood TV
S = conv2(eye(length(x)-1),ones(35,1),'same') - eye(length(x)-1);

lambda = 35;
step_size = 0.000001;

x_est = y;

% Here we use a subgradient method to solve.
% Could also use ADMM for a more efficient solution, see paper for details.
for i = 1:75000
    grad = (x_est - y);
    Dx = D*x_est;
    grad = grad + lambda*2*D' * ((S*abs(Dx)).*sign(Dx));
    
    x_est = x_est - step_size*grad;
    
    cost_lntv(i) = norm(x_est - y) + abs(Dx)'*S*abs(Dx);
    step_size = step_size*0.99999;
end

%% Total Variation only
step_size = 0.00005;
x_est_tv = y;
lambda = 3.52;

for i = 1:75000
    grad = (x_est_tv - y);
    grad = grad + lambda*D'*sign(D*x_est_tv);
    
    x_est_tv = x_est_tv - step_size*grad;
    
    cost_tv(i) = norm(x_est_tv - y) + norm(D*x_est_tv,1);
    step_size = step_size*0.999999;
end

%% Plot results
figure 
imagesc(S)
title('Structured sparsity matrix')

figure
subplot(3,1,1)
plot(y)
hold on
plot(x)
legend('Noisy','Ground truth')
title('Input')

subplot(3,1,2)
plot(x_est)
hold on
plot(x_est_tv)
legend('LN-TV','TV')
title('Denoised')

subplot(3,1,3)
title('Absolue error')
plot(abs(x-x_est))
hold on
plot(abs(x-x_est_tv))
legend('LN-TV error','TV error')
title('Absolute error')
