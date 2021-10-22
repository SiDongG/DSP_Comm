x=0;
sum=0;
while x<21
    sum=sum+nchoosek(120,x)*(0.1^x)*0.9^(120-x);
    x=x+1;
end
result=1-sum;