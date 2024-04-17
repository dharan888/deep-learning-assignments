function cost = cost_function(errors)
    cost   = sqrt(errors*transpose(errors));
end