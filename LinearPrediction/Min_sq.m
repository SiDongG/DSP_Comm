function e=Min_sq(data)
count=1;
error_sum=0;
while count<length(data)
    error_sum=error_sum+data(count)^2;
    count=count+1;
end
e=error_sum;
    