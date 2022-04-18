function data_pre=reverser(ak,data)
count=1;
ak_r=zeros(length(ak),1);
while count<length(ak)+1
    ak_r(count)=ak(length(ak)-count+1);
    count=count+1;
end
ak_r=[0;ak_r];
data_pre=filter(-ak_r,1,data);

