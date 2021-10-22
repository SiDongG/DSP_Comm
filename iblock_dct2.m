function X = iblock_dct2 ( a )
[m , n ] = size ( a ) ;
X = zeros (m , n ) ;
for i = 1:8: m
for j = 1:8: n
X ( i : i +7 , j : j +7) = idct2 ( a ( i : i +7 , j : j +7) ) ;
end
end
end