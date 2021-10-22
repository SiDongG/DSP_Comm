B=0;
A=0;
C=0;
D=0;
count=1;
syms k
while A<15
   B=A^15/(factorial(15)*symsum(A^k/factorial(k),0,15));
   D(count)=B;
   C(count)=A;
   A=A+1;
   count=count+1;
end
figure()
semilogy(C,D)