function result = softplus(x)
    result = log(1+exp(x)) ./ log(1 + exp(1)) ;
end  