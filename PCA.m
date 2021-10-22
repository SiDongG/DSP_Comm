load('/Users/guostone/Downloads/hw11digits.mat')
m=mean(X);
xm=X-m;
Q = xm*xm';
[V,D] = eig(Q);
[d,ind] = sort(diag(D),'descend');
D = D(ind,ind);
V = V(:,ind);
W = V(:,1:10);
g = W'*xm;
g = g';
count=1;
while count<11
    v=g(:,count);
    figure
    imagesc(reshape(v, 20, 50));
    count=count+1;
end
