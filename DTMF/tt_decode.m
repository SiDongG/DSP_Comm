
function digits =tt_decode(x);
tt.keys  = [ '1','2','3','A';        %DTMF key tones in a 4*4 table%
            '4','5','6','B';
            '7','8','9','C';
            '*','0','#','D' ];
tt.colTones = [1209,1336,1477,1633];
tt.rowTones = [697,770,852,941];
Low_num=load('Lowband.mat');         %Load the bandpass filters%
Low=Low_num.Low_num;         
High_num=load('Highband.mat');
High=High_num.High_num;
load(x);                             %load the mat file%
signal1=signal(8000:length(signal)-7000);  %Shorten the signal to samples of interest%
%compute the noise average%
noise=signal(8100:8400);
noise_sum=0;
n=1;
while n<300
    noise_sum=noise_sum+noise(n)^2;
    n=n+1;
end
noise_avg=noise_sum/300;
%......
Lowband=conv(Low,signal1);        %convolving signal with a passband filter for row tones%
Highband=conv(High,signal1);      %convolving signal with a passband filter for column tones%
%......
pre_mode=0;       %Previous mode%
mode=0;           %mode=0 means noise, mode=1 means valid signal segment%
digit_num=0;      %integer that keeps track of number of digits decoded%
index_norm=0;     %normalised index value for each valid loop%
index_sum=0;      %sum of indexes over a window%
index_num=0;      %number of indexes added over a window%
rows=0;           %row list%
cols=0;           %column list%
sample1_index=0;  %index of sample%
pre_index=1;
row=0;            %row value, update every digit%
while sample1_index<length(Lowband)-212     %loop the signal with a window of 212%
    sample1_index=sample1_index+212;        %increment 212 every loop%
    %lowband%
    sample=Lowband(pre_index:sample1_index);
    y=goertzel(sample,1:212/323:212);       %Perform DFT with N=323%
    count=1;
    count1=1;
    index=0;
    peak=0;
    sum=0;
    %obtain the index number corresponding to the highest peak in DFT%
    while count<1/2*length(y)
        if sqrt(real(y(count))^2+imag(y(count))^2)>peak
            index=count;
            peak=sqrt(real(y(count))^2+imag(y(count))^2);
        end
        count=count+1;
    end
    %Compute average signal power%
    while count1<212
        sum=sum+sample(count1)^2;
        count1=count1+1;
    end
    sum_avg=sum/212;
    if abs(sum_avg-noise_avg)<0.023    %if considered noise%
        mode=0;
        if pre_mode==1                 %previous window was a non-noise segment%
            index_avg=(index_sum/index_num)*8000/length(y);%average and normalise the index%
            index_sum=0;
            index_num=0;
            %determine the row number%
            if (672<index_avg)&&(index_avg<721)
                row=1;
            end
            if (743<index_avg)&&(index_avg<797)
                row=2;
            end
            if (822<index_avg)&&(index_avg<882)
                row=3;
            end
            if (908<index_avg)&&(index_avg<974)
                row=4;
            end
            digit_num=digit_num+1;
            rows(digit_num)=row;      %add the row to the row list% 
        end
        pre_mode=mode;                %set the previous mode to the current mode for next loop%
    else   %if not noise%
        mode=1;
        index_sum=index_sum+index;    %add the new index to old index%
        index_num=index_num+1;
        pre_mode=mode;                %set the previous mode to the current mode for next loop%
    end
    pre_index=pre_index+212;
end
%.....%                               %Repeat the loop for high band%
pre_mode=0;
mode=0;
digit_num=0;
index_norm=0;
index_sum=0;
index_num=0;
sample1_index=0;
pre_index=1;
col=0;
while sample1_index<length(Highband)-106
    sample1_index=sample1_index+106;
    %highband%
    sample=Highband(pre_index:sample1_index);
    y=goertzel(sample,1:106/332:106);
    count=1;
    count1=1;
    index=0;
    peak=0;
    sum=0;
    while count<1/2*length(y)
        if sqrt(real(y(count))^2+imag(y(count))^2)>peak
            index=count;
            peak=sqrt(real(y(count))^2+imag(y(count))^2);
        end
        count=count+1;
    end
    while count1<106
        sum=sum+sample(count1)^2;
        count1=count1+1;
    end
    sum_avg=sum/106;
    if abs(sum_avg-noise_avg)<0.023
        mode=0;
        if pre_mode==1
            index_avg=(index_sum/index_num)*8000/length(y);
            index_sum=0;
            index_num=0;
            if (1167<index_avg)&&(index_avg<1251)
                col=1;
            end
            if (1289<index_avg)&&(index_avg<1383)
                col=2;
            end
            if (1425<index_avg)&&(index_avg<1529)
                col=3;
            end
            digit_num=digit_num+1;
            cols(digit_num)=col;
        end
        pre_mode=mode;
    else
        mode=1;
        index_sum=index_sum+index;
        index_num=index_num+1;
        pre_mode=mode;
    end
    pre_index=pre_index+106;
end
count2=1;
%determine the digit by finding the corresponding coordinate in the matrix%
while count2<11 
    digits(count2)=tt.keys(rows(count2),cols(count2));
    count2=count2+1;
end
digits=strcat(digits(1:3),'-',digits(4:6),'-',digits(7:10)); %concatenate the strings and formatting%
end
 
