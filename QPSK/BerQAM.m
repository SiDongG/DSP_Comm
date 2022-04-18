SNR=2;
Q1=0;
Ber=0;
count=1;
while count<6
    SNR1=10^(SNR/10);
    Q1(count)=0.5*erfc(sqrt(SNR1/2));
    Ber(count)=transceiver(400000,SNR,4);
    SNR=SNR+2;
    count=count+1;
end
semilogy(2:2:10,Q1)
hold on
semilogy(2:2:10,Ber)

