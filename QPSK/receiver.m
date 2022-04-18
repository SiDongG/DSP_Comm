function est_sym=receiver(y, mod_size)
count=1;
if mod_size==2
    while count<length(y)+1
        if y(count)<=0
            est_sym(count)=-1;
        end
        if y(count)>0
            est_sym(count)=1;
        end
        count=count+1;
    end
end
if mod_size==4
    while count<length(y)+1
        if (real(y(count))>=0 && imag(y(count))>0)
            est_sym(count)=1+1i;
        end
        if (real(y(count))<=0 && imag(y(count))>0)
            est_sym(count)=-1+1i;
        end
        if (real(y(count))<0 && imag(y(count))<=0)
            est_sym(count)=-1-1i;
        end
        if (real(y(count))>0 && imag(y(count))<=0)
            est_sym(count)=1-1i;
        end
        count=count+1;
    end
end