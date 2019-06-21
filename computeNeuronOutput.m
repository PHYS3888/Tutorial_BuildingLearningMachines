function neuronOutput = computeNeuronOutput(w,x1,x2)

% Define the neuron's input-output transformation:
y = @(x,w) 1/(1 + exp(-w*x));

% Compute the output values:
numPoints = length(x1);
neuronOutput = zeros(numPoints,numPoints);
for i = 1:numPoints
    for j = 1:numPoints
        neuronOutput(i,j) = y([x1(i);x2(j)],w);
    end
end

end
