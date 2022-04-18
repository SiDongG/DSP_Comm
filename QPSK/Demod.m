function est_bits=Demod(est_sym, mod_size)
count=1;
if mod_size==2
    while count<length(est_sym)+1
        est_bits(count)=(est_sym(count)+1)/2;
        count=count+1;
    end
end
if mod_size==4
    while count<length(est_sym)+1
        if est_sym(count)==1+1i
            est_bits(2*count-1)=0;
            est_bits(2*count)=0;
        end
        if est_sym(count)==-1+1i
            est_bits(2*count-1)=0;
            est_bits(2*count)=1;
        end
        if est_sym(count)==-1-1i
            est_bits(2*count-1)=1;
            est_bits(2*count)=1;
        end
        if est_sym(count)==1-1i
            est_bits(2*count-1)=1;
            est_bits(2*count)=0;
        end
        count=count+1;
    end
end
