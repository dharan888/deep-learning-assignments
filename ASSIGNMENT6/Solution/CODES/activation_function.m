function activations = activation_function(netinputs,ftype)
   switch ftype 
       case char('sig')
           activations = 1./(1+exp(-netinputs));
       case char('tan')
           activations = tanh(netinputs);
       case char('lin')
           activations = netinputs;
       otherwise
           disp("No such activation functions are available.")
           disp("Type : for sigmoid fuction - 'sig', tan hyperbolic - 'tan' and  linear function - lin");
   end
end