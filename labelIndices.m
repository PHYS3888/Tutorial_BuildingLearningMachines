function labelIndices()
c = 1;
for i = 1:5
    for j = 1:5
        text(i-0.1,j-0.1,num2str(c),'Color','r');
        c = c+1;
    end
end
end
