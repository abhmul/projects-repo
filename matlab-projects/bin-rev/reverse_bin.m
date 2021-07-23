function y=reverse_bin(x)
y = zeros(size(x));
for i=1:length(x)
    y(i) = x(i) - bin2dec(reverse(dec2bin(x(i))));
end

    