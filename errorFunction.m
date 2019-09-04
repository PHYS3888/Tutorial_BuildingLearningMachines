function totError = errorFunction(y,w,dataMat,isModel)

numDataPoints = size(dataMat,1);

G = zeros(numDataPoints,1);
for i = 1:numDataPoints
    G(i) = isModel(i)*log(y(dataMat(i,:)',w)) + (1-isModel(i))*(1-y(dataMat(i,:)',w));
end

totError = -sum(G);

end
