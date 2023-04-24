% Apply 2D local-neighborhood TV (based on MEGS) to regularize a deblurring
% problem, compare and combine with regular TV to see synergy.

% Create a ground truth image
xx=(1:70).'; yy=1:70;
x = double((xx-35).^2+(yy-35).^2 <=25^2);
x(25:45,25:45) = 2;
x(35,10) = 0;
x(10,35) = 0;
x(35,60) = 0;
x(60,35) = 0;
x(32:38,10:60) = 2;
x(10:60,32:38) = 2;
x((xx-60).^2+(yy-60).^2 <=5^2) = 0.5;
x(10:15,10:15) = 1;
x(60,55) = 0;
x(55,60) = 0;
x(60,65) = 0;
x(65,60) = 0;

%%

% Create a blur kernel and blur matrix, and pseudoinverse
h = fspecial('gaussian',5,4);
H = convmtx2(h,size(x));
Hpinv = pinv(full(H));

% Apply blur to image
y = H*x(:);

% Add noise
y_noise = reshape(awgn(y,45,'measured'),74,74);

% Construct structured sparsity matrices for horizontal and vertical
% sliding windows ('local neighborhood TV')
Ss = convmtx([0.5,1,1,1,1,1,1,1,0.5],70);
Ss(:,1:4) = [];
Ss(:,end-3:end) = [];
Ss = Ss-eye(size(Ss));
S = Ss;
for i = 1:69
    S = blkdiag(S,Ss);
end
Sv = sparse(S);

c=1;
S = zeros(70*70);
for i = 1:70
    for j = 1:70
        binaryMap=(xx-j)==0 & abs(yy-i)<=3;
        binaryMap(j,i) = 0;
        S(:,c) = double(binaryMap(:));
        c = c+1;
    end
end
Sh = sparse(S);

% Construct horizontal and vertical finite differencing matrices
Dv = diag(ones(1,70*70),0) - diag(ones(1,70*70-1),1);
Dv(70:70:end,:) = 0;

Dh = zeros(70*70);
for i = 1:70*70
   Dh(i,i) = 1;
   Dh(i,i+70) = -1;
end

Dh(:,4901:end) = [];
Dh(end-69:end,:) = 0;

Dv = sparse(Dv);
Dh = sparse(Dh);

%% Perform optimizations

%% MEGS (2D LN-TV)
x_est = Hpinv*y_noise(:);

lambda = 1.1;
n = 0.75;
stepsize = 0.00015;

for i = 1:1000
    grad = H'*(H*x_est - y_noise(:));
    grad = grad + lambda* (3*Dh'*((Sh*abs(Dh*x_est).^n) .*(abs(Dh*x_est).^(1-n)).*sign(Dh*x_est)) );
    grad = grad + lambda* (3*Dv'*((Sv*abs(Dv*x_est).^n) .*(abs(Dv*x_est).^(1-n)).*sign(Dv*x_est)) );
   
    x_est_new = x_est - stepsize*grad;
    
    c(i) = 0.5*norm(H*x_est_new - y_noise(:))^2 + lambda * (abs(Dv*x_est_new(:)).^n'*Sv*abs(Dv*x_est_new(:)).^n + abs(Dh*x_est_new(:)).^1.5'*Sh*abs(Dh*x_est_new(:)).^1.5);
    err(i) = 0.5*norm(x(:) - x_est_new(:));
    
    x_est = x_est_new;
    
    if i>2
    if c(i)>c(i-1)*1.01
        stepsize  = stepsize*0.995;
    end
    end
    
    if mod(i,10) == 0
        subplot(2,3,1)
        imagesc(reshape(x_est,70,70))
        title('LN-TV Estimate')

        subplot(2,3,2)
        imagesc(reshape(H*x_est-y_noise(:),74,74))
        title('Error')
        title(num2str(i))
        
        subplot(2,3,3)
        imagesc(reshape(abs(Dv*x_est(:)),70,70))
        title('Vertical difference in est')
        subplot(2,3,4)
        plot(c)

        subplot(2,3,5)
        plot(err)
        title('error')
        imagesc(reshape(abs(Dh*x_est(:)),70,70))
        title('Horizontal difference in est')
        drawnow
    end
end

x_est_LNTV = x_est;

%% Total Variation

figure
clear c
clear err
x_est = Hpinv*y_noise(:);

stepsize = 0.0035;
tvlamb = 0.35;

for i = 1:1500
    grad = H'*(H*x_est - y_noise(:));
    grad = grad + tvlamb* Dh'*(sign(Dh*x_est));
    grad = grad + tvlamb* Dv'*(sign(Dv*x_est));
   
    x_est_new = x_est - stepsize*grad;
    
    c(i) = 0.5*norm(H*x_est_new - y_noise(:))^2 + lambda * sum((abs(Dv*x_est_new(:)) + abs(Dh*x_est_new(:))));
    err(i) = 0.5*norm(x(:) - x_est_new(:));
    
    x_est = x_est_new;
    
    if i>2
    if c(i)>c(i-1)
        stepsize  = stepsize*0.99;
    end
    end
    
    if mod(i,10) == 0
        subplot(2,3,1)
        imagesc(reshape(x_est,70,70))
        title('TV est')

        subplot(2,3,2)
        imagesc(reshape(H*x_est-y_noise(:),74,74))
        title('Error')
        title(num2str(i))
        
        subplot(2,3,3)
        imagesc(reshape(abs(Dv*x_est(:)),70,70))
        title('Vertical difference in est')
        
        subplot(2,3,4)
        plot(c)

        subplot(2,3,5)
        plot(err)
        title('error')
        imagesc(reshape(abs(Dh*x_est(:)),70,70))
        title('Horizontal difference in est')
        drawnow
    end
end

x_est_tv = x_est;

%% LN-TV combined with TV

figure
clear c
clear err
x_est = Hpinv*y_noise(:);

stepsize = 0.00275;

tvlamb = 0.17;
lamda = 0.005;
n = 0.75;

for i = 1:1750
    grad = H'*(H*x_est - y_noise(:));

    grad = grad + tvlamb* Dh'*(sign(Dh*x_est));
    grad = grad + tvlamb* Dv'*(sign(Dv*x_est));
    
    grad = grad + lamda* (3*Dh'*((Sh*abs(Dh*x_est).^n) .*(abs(Dh*x_est).^(1-n)).*sign(Dh*x_est)) );
    grad = grad + lamda* (3*Dv'*((Sv*abs(Dv*x_est).^n) .*(abs(Dv*x_est).^(1-n)).*sign(Dv*x_est)) );    
   
    x_est_new = x_est - stepsize*grad;
    
    c(i) = 0.5*norm(H*x_est_new - y_noise(:))^2 + lambda * (abs(Dv*x_est_new(:)).^n'*Sv*abs(Dv*x_est_new(:)).^n + abs(Dh*x_est_new(:)).^1.5'*Sh*abs(Dh*x_est_new(:)).^1.5) + lambda * sum((abs(Dv*x_est_new(:)) + abs(Dh*x_est_new(:))));
    err(i) = 0.5*norm(x(:) - x_est_new(:));
    
    x_est = x_est_new;
    
    if i>2
    if c(i)>c(i-1)*1.005
        stepsize  = stepsize*0.99;
    end
    end
    
    lambda = lambda*1.00015;
    
    if mod(i,10) == 0
        subplot(2,3,1)
        imagesc(reshape(x_est,70,70))
        title('LN-TV + TV Estimate')

        subplot(2,3,2)
        imagesc(reshape(H*x_est-y_noise(:),74,74))
        title('Error')
        title(num2str(i))
        
        subplot(2,3,3)
        imagesc(reshape(abs(Dv*x_est(:)),70,70))
        title('Vertical differences in est')
        subplot(2,3,4)
        plot(c)

        subplot(2,3,5)
        plot(err)
        title('error')
        imagesc(reshape(abs(Dh*x_est(:)),70,70))
        title('Horizontal differences in est')
        drawnow
    end
end

x_est_both = x_est;

%% 
disp(['LN-TV PSNR: ',num2str(psnr(x_est_LNTV, x(:)))]);
disp(['TV PSNR: ',num2str(psnr(x_est_tv, x(:)))]);
disp(['BOTH PSNR: ',num2str(psnr(x_est_both, x(:)))]);

%% Plot the final results

figure
subplot(2,2,1)
imagesc(y_noise)
title('Blurred, 45 dB SNR')
subplot(2,2,2)
imagesc(reshape(x_est_tv,70,70))
xlabel(['PSNR: ',num2str(psnr(x_est_tv, x(:)))])
title('TV')
subplot(2,2,3)
imagesc(reshape(x_est_LNTV,70,70))
xlabel(['PSNR: ',num2str(psnr(x_est_LNTV, x(:)))])
title('2D LN-TV (MEGS)')
subplot(2,2,4)
imagesc(reshape(x_est_both,70,70))
xlabel(['PSNR: ',num2str(psnr(x_est_both, x(:)))])
title('TV + 2D LN-TV (MEGS)')

function a = div0(x,y)
    a = x./y;
    a(y==0) = 0;
end