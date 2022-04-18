function error=Cross_predict(a,data,p)
L=182;
a=[0;a];
data_pre=filter(-a,1,data);
count=p+1;
sum=0;
while count<L+1
    sum=sum+(data(count)-data_pre(count))^2;
    count=count+1;
end
error=sum;
