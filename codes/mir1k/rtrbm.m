function theta = rtrbm( data_cell,theta,numdims,numhid,maxepoch,actfun,actfun_vis,rnn,cdn )
%%------------------------------------------------------------------------%%
theta.a             = zeros(numdims,1);
lr_wvh              = 0.1;
lr_whh              = 0.1;
lr_vb               = 0.1;
lr_hb               = 0.1;
weightcost          = 0.2;
initialmomentum     = 0.9;
finalmomentum       = 0.6;
vishidinc           = zeros(numhid,numdims);
vvshidinc           = zeros(numhid,numhid);
hidbiasinc          = zeros(numhid,1);
visbiasinc          = zeros(numdims,1);

%%----------------------for epoch = 1:maxepoch----------------------------%%
for epoch = 1:maxepoch
    loss = 0;
    dtheta.w_vh = zeros(numhid,numdims);
    dtheta.w_hh = zeros(numhid,numhid);
    dtheta.a  = zeros(numdims,1);
    dtheta.b  = zeros(numhid,1);
    
    if epoch>20
        cdn = 10;
    elseif epoch > 100
            cdn = 25;
    end
    
    if epoch>5,
      momentum=finalmomentum;
    else
      momentum=initialmomentum;
    end;
    %%-----------------for c = 1:numel(data_cell)--------------------%%    
    for c = 1:numel(data_cell)
%         mbsz=min(size(data_cell{c},2), floor(200*1024^2/size(data_cell{c},1)/8)); 
%         if mod(size(data_cell{c},2),mbsz) == 0
%             nbat = floor(size(data_cell{c},2)/mbsz);
%         else
%             nbat = floor(size(data_cell{c},2)/mbsz)+1;
%         end
        mbsz=min(size(data_cell{c},2), floor(200*1024^2/size(data_cell{c},1)/8)); % 200mb max
        if mbsz==0, continue; end
        nbat = floor(size(data_cell{c},2)/mbsz)+1;
        %%----------for bat = 1:nbat---------------%
        for bat = 1:nbat
            data = data_cell{c}(:,1+(bat-1)*mbsz: min(size(data_cell{c},2), bat*mbsz));
            if size(data,2)==0
                continue;
            end
            T =size(data,1) / numdims;
            V = cell (T,1);
            H = cell (T,1);
            B = cell (T,1);
          
            d_B   = cell (T,1);
            d_BL  = cell (T,1);
            F_t   = zeros (numhid,1);
            vishidinc1           = zeros(numhid,numdims);
            vvshidinc1           = zeros(numhid,numhid);
            hidbiasinc1          = zeros(numhid,1);
            visbiasinc1          = zeros(numdims,1);
            for t = 1:T
                V{t,1} = data((t-1)*numdims+1:t*numdims,:)';
            end
   
            if actfun == 'logistic'
                H{1,1} = sigmoid(V{1,1}*theta.w_vh'+ repmat(theta.b',size(V{1,1},1),1));  %H (c,1000)
            elseif actfun == 'RELU'
                 H{1,1} = max(0,V{1,1}*theta.w_vh'+ repmat(theta.b',size(V{1,1},1),1));
            end
            for t = 2:T
               B{t,1} = H{t-1,1}*theta.w_hh';
               if actfun =='logistic'
                    H{t,1} = sigmoid(V{t,1}*theta.w_vh' + B{t,1}+repmat(theta.b',size(V{t,1},1),1));
               elseif actfun == 'RELU'
                    H{t,1} = max(0,V{t,1}*theta.w_vh' + B{t,1}+repmat(theta.b',size(V{t,1},1),1));
               end
            end
            
            VH_t=theta;
            for t = T:-1:1
                d_B{t,1} = (sum(H{t,1}.*(1-H{t,1}))'/size(H{t,1},2)).*F_t;% d_B(1000,1)
%                 if T>1&&t>1
%                     dtheta.b = d_B{t,1} + dtheta.b;
%                 end
                [dVH_t,loss1] = rbm_cdn(VH_t,V{t,1},size(H{t,1},2),actfun,actfun_vis,cdn);
                 loss = loss+loss1; 
                
                d_BL{t,1} = dVH_t.hb;
               
%                 vishidinc1 = momentum*vishidinc1 + ...
%                     lr_wvh*( dVH_t.w_vh - weightcost*VH_t.w_vh);
%                 VH_t.w_vh = VH_t.w_vh+ vishidinc1;
%                 visbiasinc1 = momentum*visbiasinc1 + lr_vb*dVH_t.vb;
%                 VH_t.a = VH_t.a + visbiasinc1;
%                 hidbiasinc1 = momentum*hidbiasinc1 + lr_hb*dVH_t.hb;
%                 VH_t.b = VH_t.b +hidbiasinc1;
                dtheta.w_vh = dtheta.w_vh+dVH_t.w_vh; dtheta.a = dtheta.a+dVH_t.vb; dtheta.b = dtheta.b+dVH_t.hb;
                F_t = VH_t.w_hh*(d_B{t,1}+d_BL{t,1});%%%%”–Œ Ã‚
            end
%             theta = VH_t;
            
            for t=2:T
                dtheta.w_hh = dtheta.w_hh + ((sum(H{t-1,1})'/size(H{t-1,1},1))*(d_B{t,1})')';
                dtheta.b = dtheta.b + sum(d_B{t,1});
                dtheta.w_vh = dtheta.w_vh + ((sum(V{t,1})'/size(V{t,1},1))*d_B{t,1}')';
%                 dtheta.b = dtheta.b + sum(d_B{t,1});
            end
             dtheta.w_vh = dtheta.w_vh + (sum(V{1,1})'/size(V{1,1},1)*d_B{1,1}')';
             
             dtheta.b = dtheta.b + sum(d_B{1,1});
             
             vvshidinc = momentum*vvshidinc + ...
                lr_whh*( dtheta.w_hh - weightcost*theta.w_hh);
             theta.w_hh = theta.w_hh+ vvshidinc;
             vishidinc = momentum*vishidinc + ...
                lr_wvh*( dtheta.w_vh - weightcost*theta.w_vh);
             theta.w_vh = theta.w_vh+ vishidinc;
             visbiasinc = momentum*visbiasinc + lr_vb*dtheta.a;
             theta.a = theta.a + visbiasinc;
             hidbiasinc = momentum*hidbiasinc + lr_hb*dtheta.b;
             theta.b = theta.b +hidbiasinc;
        end
        %%--------end for bat = 1:nbat---------------%
    end
    %%----------end for c = 1:numel(data_cell)----------------------%%
   fprintf(1, 'epoch %4i error %6.1f  \n', epoch, loss); 
end
%%--------------end for epoch = 1:maxepoch--------------------------------%%
  
    
end

