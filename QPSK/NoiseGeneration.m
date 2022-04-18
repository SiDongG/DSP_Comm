function Rec=NoiseGeneration(Symbols,mod_size,SNR)
%SNR1=10^(SNR/10);
if mod_size==2
    L=length(Symbols);
    Noise=randn(1,L);
    Rec=Noise+sqrt(2*SNR)*Symbols;
end
if mod_size==4
    L=length(Symbols);
    nr=randn(1,L);
    ni=randn(1,L);
    Noise=(sqrt(2)/2)*(nr+1i*ni);
    Rec=Noise+sqrt(SNR)*Symbols;
end

