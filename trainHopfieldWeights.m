function w = trainHopfieldWeights(memoryMatrix,eta)

if nargin < 1
    % Learning rate (unimportant for binary networks):
    eta = 1;
end
%-------------------------------------------------------------------------------

% Preliminaries:
[numNeurons,numMemories] = size(memoryMatrix);
fprintf(1,'Trying to store %u memories in a network of %u neurons\n',...
                numMemories,numNeurons);

%-------------------------------------------------------------------------------
% Nice matrix multiplication way:
%-------------------------------------------------------------------------------
w = memoryMatrix * memoryMatrix';
% Set diagonal to zero:
w(logical(eye(size(w)))) = 0;

%-------------------------------------------------------------------------------
% Laborious loop way:
%-------------------------------------------------------------------------------
% w = zeros(numNeurons,numNeurons);
% % Loop through memories to update weights:
% for i = 1:numNeurons
%     for j = 1:numNeurons
%         if i~=j
%             w(i,j) = eta * sum(memoryMatrix(i,:).*memoryMatrix(j,:));
%         end
%     end
% end


end
