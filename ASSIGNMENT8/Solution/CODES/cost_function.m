function cost = cost_function(errors)
    cost   = sum(errors.*errors,'all')/size(errors,2);
end