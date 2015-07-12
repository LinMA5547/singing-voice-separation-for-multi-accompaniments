function [ d_theta loss ] = rbm_cdn( theta,x,numhid,actfun,actfun_vis,cdn)


batch_size = size(x,1);
V = x;
if actfun =='logistic'
    H = sigmoid(x*theta.w_vh'+repmat(theta.b',batch_size,1 ));
elseif actfun =='RELU'
    H = max(0,x*theta.w_vh'+repmat(theta.b',batch_size,1));
end
pos_vb = sum(V);
pos_hb = sum(H);
pos_w_vh = V'*H;
for cd = 1:cdn
     H = H > rand(batch_size,numhid);
     if actfun_vis =='logistic'
         V = sigmoid(H*theta.w_vh+repmat(theta.a',batch_size,1));
     elseif acrfun_vis == 'RELU'
         V = max(0,H*theta.w_vh+repmat(theta.a',batch_size,1));
     end
     if actfun == 'logistic'
         H = sigmoid (V*theta.w_vh'+ repmat(theta.b',batch_size,1));
     elseif actfun == 'RELU'
         H = max (0,V*theta.w_vh'+ repmat(theta.b',batch_size,1));
     end
end
     neg_vb = sum(V);
     neg_hb = sum(H);
     neg_w_vh = V'*H;
     d_theta.vb = ((pos_vb-neg_vb)/batch_size)';
     d_theta.hb = ((pos_hb-neg_hb)/batch_size)';
     d_theta.w_vh = ((pos_w_vh - neg_w_vh)/batch_size)';
    
     loss = sum(sum((V-x).^2));
     
end

