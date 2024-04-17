a=[1 .5 .5;.2 .3 .4;.6 .7 .8]
b=[1;2;3]
I=sub2ind(size(a),b,b)
a(I)=a(I)-1
