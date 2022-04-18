function M=Invest(data,data_p)
N=521;
count=1;
Initial=1000;
bank=1.0006;
while count<N
    if data_p(count+1)/data_p(count)>bank
        Initial=Initial*(data(count+1)/data(count));
    else
        Initial=Initial*bank;
    end
    count=count+1;
end
M=Initial;