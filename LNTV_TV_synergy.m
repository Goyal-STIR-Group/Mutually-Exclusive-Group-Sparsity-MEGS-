% Denoising a noisy piecewise constant signal with both LN-TV and TV combined
% Demonstrates synergy.
rng(1)

% Ground truth signal
x = zeros(100,1);
x(40:70) = -3;
x(90:100) = 3;

% Noisy
y = awgn(x,10,'measured');

%% LN-TV (MEGS) and TV combined

% Create structured sparsity matrix for local-neighborhood TV
row = [linspace(0.6,1,5),1,linspace(1,0.6,5)];
S = conv2(eye(length(x)-1),row,'same') - eye(length(x)-1);

% Construct finite differencing matrix
D = diag(ones(1,length(x)),0) - diag(ones(1,length(x)-1),1);
D(end,:) = [];
D = sparse(D);

lambda = 0.675; % LN-TV strength
lambdatv = 3; % TV strength

step_size = 0.00005;
x_est = y;

% Here we use a subgradient method to solve
% Could also use ADMM for a more efficient solution, see paper for details.
for i = 1:150000
    grad = (x_est - y);
    Dx = D*x_est;
    grad = grad + lambda*2*D' * ((S*abs(Dx)).*sign(Dx));
    grad = grad + lambdatv*D'*sign(D*x_est);
    x_est = x_est - step_size*grad;
    
    cost(i) = norm(x_est - y) + abs(Dx)'*S*abs(Dx)+ norm(D*x_est,1);
    step_size = step_size*0.999985;
end

%% Total Variation ONLY

step_size = 0.00005;
x_est_tv = y;
lambda = 5.25; % TV strength

for i = 1:150000
    grad = (x_est_tv - y);
    grad = grad + lambda*D'*sign(D*x_est_tv);
    
    x_est_tv = x_est_tv - step_size*grad;
    
    cost_tv(i) = norm(x_est_tv - y) + norm(D*x_est_tv,1);
    step_size = step_size*0.99999;
end


%% Plot results 

subplot(3,1,1)
plot(y)
hold on
plot(x)
legend('Noisy','Ground truth')
title('Input')

subplot(3,1,2)
% plot(x)
hold on
plot(x_est)
plot(x_est_tv)
legend('LN-TV','TV')
title('Estimates')

subplot(3,1,3)
plot(abs(x-x_est))
hold on
plot(abs(x-x_est_tv))
legend('LN-TV error','TV error')
title('Absolute error')