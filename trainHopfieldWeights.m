function w = trainHopfieldWeights(memoryMatrix)

% Learning rate (unimportant for binary networks):
eta = 1;

% Preliminaries:
[numNeurons,numMemories] = size(memoryMatrix);
fprintf(1,'Trying to store %u memories in a network of %u neurons\n',...
                numMemories,numNeurons);

% Initialize weights:
w = zeros(numNeurons,numNeurons);

% Loop through memories to update weights:
for i = 1:numNeurons
    for j = 1:numNeurons
        if i~=j
            w(i,j) = eta * sum(memoryMatrix(i,:).*memoryMatrix(j,:));
        end
    end
end


end
