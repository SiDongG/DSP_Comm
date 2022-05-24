t=0;
P=5;
NDFT=zeros(P);
tau=[1,3,5,6,9];
f=[1,2,3,4,5];
for a=1:P
    for b=1:P
        NDFT(a,b)=exp(-1i*2*pi*tau(a)*f(b)/P);
    end 
end
NDFT=NDFT*1/sqrt(P);
INDT=conj(NDFT);

l=1/norm(NDFT);

h=[-2.9271+1.2205i, 1.3416+2.0474i, 2.3292 + 0.3078i,-2.9271+1.2205i,-3.1507-1.6180i].';
Truep=[-0.6382 + 0.2629i
   0.8618 + 0.4253i
   6.2610 + 0.0000i
  -0.6382 + 0.2629i
  -0.6382 - 0.2629i];
p=[2,3,4,1,2+1i].';
alpha=3;
C=1;
converged=false;
while converged==false
    p1=sparsify(p-l*conj(NDFT)*(NDFT*p-h),l*alpha);
    disp(norm(p1-p))
    if norm(p1-p)<C
        converged=true;
        p=p1;
    else
        t=t+1;
        p=p1;
    end
end
p