SNR=-2;
Q1=0;
Ber=0;
count=1;
while count<7
    SNR1=10^(SNR/10);
    Q1(count)=0.5*erfc(sqrt(SNR1));
    Ber(count)=transceiver(100000,SNR,2);
    SNR=SNR+2;
    count=count+1;
end
semilogy(-2:2:8,Q1)
hold on
semilogy(-2:2:8,Ber)

