function a=CrossCov(data,p)
L=155;
R=zeros(p);
r=zeros(p,1);
count_r=1;
while count_r<p+1
    count_c=1;
    while count_c<p+1
        count=max(count_r,count_c)+1;
        sum=0;
        while count<L
            sum=sum+data(count-count_r)*data(count-count_c);
            count=count+1;
        end
        R(count_r,count_c)=sum;
        count_c=count_c+1;
    end
    count_r=count_r+1;
end
count_r=1;
while count_r<p+1
    count=count_r+1;
    sum=0;
    while count<L
        sum=sum+data(count)*data(count-count_r);
        count=count+1;
    end
    r(count_r)=sum;
    count_r=count_r+1;
end
a=-R\r;
