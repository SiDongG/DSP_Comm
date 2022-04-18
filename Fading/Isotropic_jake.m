%% 2-D Isotropic Scattering with Jake's deterministic model
n2=1:200;
scale=0.83;   %Scaling Factor
t=n2*T;
[Value,Value2]=SUM(t);
gi=scale*sqrt(2/N)*(2*Value+sqrt(2)*cos(2*pi*fm*t));
gq=scale*sqrt(2/N)*2*Value2;
g=sqrt(gi.^2+gq.^2);

plot(n2,20*log(g));
xlabel('Time t/T');
ylabel('Envelop Power dB');

W=[50,100,150,200,250,300];
% Find the mean-square matrix
Mean=[];
for count=1:6
    Inc=0;
    total_omega=0;
    for Estimate=1:1000
        total=0;
        for Window=1:W(count)
            t=(Window+Inc)*T;
            [Value,Value2]=SUM(t);
            gi=scale*sqrt(2/N)*(2*Value+sqrt(2)*cos(2*pi*fm*t));
            gq=scale*sqrt(2/N)*2*Value2;
            g=sqrt(gi^2+gq^2);
            total=total+g;
        end
        total_omega=total_omega+total/W(count);
        Inc=Inc+W(count);
    end
    Mean(count)=total_omega/1000;
end
% Find the Variance Matrix
Var=[];
for count=1:6
    Inc=0;
    sum=0;
    for Estimate=1:1000
        Window=1;
        total=0;
        for Window=1:W(count)
            t=(Window+Inc)*T;
            [Value,Value2]=SUM(t);
            gi=scale*sqrt(2/N)*(2*Value+sqrt(2)*cos(2*pi*fm*t));
            gq=scale*sqrt(2/N)*2*Value2;
            g=sqrt(gi^2+gq^2);
            total=total+g;
        end
        sum=sum+(abs(Mean(count)-total/W(count)))^2;  % Sum of square difference
        Inc=Inc+W(count);
    end
    Var(count)=sum/1000;
end
figure()
plot(W,Var)

xlabel('Window Size');
ylabel('Sample variance');




