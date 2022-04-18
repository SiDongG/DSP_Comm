function ak=LP(y)
p=10;
N=26;
X=zeros(N-p,p);
count_r=1;
while count_r<N-p+1
    X(count_r)=y(count_r);
    count_c=1;
    while count_c<p
        X(count_r,count_c+1)=y(count_r+count_c);
        count_c=count_c+1;
    end
    count_r=count_r+1;
end
x=y(p+1:N);
a=-X\x;
e=X*a+x;
ak=a;
