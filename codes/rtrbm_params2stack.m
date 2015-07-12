function [stack] = rtrbm_params2stack(theta,numdims,numhid)
p = 1;
wsize = numdims*numhid;

stack.w_vh = reshape(theta(p:p+wsize-1),numhid,numdims);
p = p+wsize; 

wsize = numhid*numhid;
stack.w_hh = reshape(theta(p:p+wsize-1),numhid,numhid);
p = p+wsize;

wsize = numdims;
stack.a = reshape(theta(p:p+wsize-1),numdims,1);
p = p+wsize;

wsize = numhid;
stack.b = reshape(theta(p:p+wsize-1),numhid,1);
end

