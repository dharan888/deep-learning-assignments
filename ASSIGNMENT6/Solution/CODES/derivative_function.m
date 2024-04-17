function derivatives = derivative_function(netinputs,ftype)
   switch ftype 
       case char('sig')
           activations = activation_function(netinputs,ftype);
           derivatives = activations.*(1-activations);
       case char('tan')
           derivatives = 1-(tanh(netinputs)).^2;
       case char('lin')
           derivatives = ones(size(netinputs));
       otherwise
           disp("No such activation functions are available.")
           disp("Type : for sigmoid fuction - 'sig', tan hyperbolic - 'tan' and  linear function - lin");
   end
end