function p=sparsify(p,t)
for i=1:length(p)
    if abs(p(i))<t
        p(i)=0;
    else
        p(i)=p(i)*(abs(p(i))-t)/abs(p(i));
    end
end