load('res1.mat');
res1 = res;
load('res2.mat');
res2 = res;

res3 = res2-res1;
k = 0;
for i = 1:numel(res3)
    if res3(i)>= 0
       k = k+1;
    end
end