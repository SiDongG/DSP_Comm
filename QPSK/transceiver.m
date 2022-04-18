function BER=transceiver(length, SNR, mod_size)
SNR=10^(SNR/10);
Input=DataGeneration(length);
Symbols=Mod(Input, mod_size);
Rec=NoiseGeneration(Symbols,mod_size,SNR);
est_sym=receiver(Rec, mod_size);
Output=Demod(est_sym, mod_size);
count=1;
Err_count=0;
while count<length+1
    if Input(count)~=Output(count)
        Err_count=Err_count+1;
    end
    count=count+1;
end
BER=Err_count/length;
