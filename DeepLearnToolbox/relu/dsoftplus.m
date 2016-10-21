function result = dsoftplus(x)
    result = 1 ./ (1 + exp(-x)) ./ log(1 + exp(1));
end