function Symbols=Mod(Bits, mod_size)
count=1;
if mod_size==2
    while count<length(Bits)+1
        if Bits(count)==0
            Symbols(count)=-1;
        end
        if Bits(count)==1
            Symbols(count)=1;
        end
        count=count+1;
    end
end
if mod_size==4
    while count<length(Bits)
        if (Bits(count)==0 && Bits(count+1)==0)
            Symbols((count+1)/2)=(1+1i)/sqrt(2);
        end
        if (Bits(count)==0 && Bits(count+1)==1)
            Symbols((count+1)/2)=(-1+1i)/sqrt(2);
        end
        if (Bits(count)==1 && Bits(count+1)==1)
            Symbols((count+1)/2)=(-1-1i)/sqrt(2);
        end
        if (Bits(count)==1 && Bits(count+1)==0)
            Symbols((count+1)/2)=(1-1i)/sqrt(2);
        end
        count=count+2;
    end
end
Symbols;