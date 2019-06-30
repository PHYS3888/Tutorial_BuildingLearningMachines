function [xNext,numIterations] = runHopfield(w,x0)
% Implements the activity rules of a Hopfield network
% defined by weights w, from an initial state, x0.

maxIters = 10;
numNeurons = length(x0);
thresholdFun = @(x) 2*(x>=0)-1; % Theta

% ---Asynchronous updates until converged---
xPrev = nan(size(x0));
xNext = x0;
numIterations = 0;
while ~all(xNext == xPrev) && (numIterations < maxIters)
    numIterations = numIterations + 1;
    fprintf(1,'Iteration %u\n',numIterations);
    
    xPrev = xNext;
    % Update order:
    updateOrder = randperm(numNeurons);
    for k = 1:numNeurons
        myNeuron = updateOrder(k);
        % Compute this neuron's activation, a:
        a = w(myNeuron,:)*xNext;
        % Threshold to update the state:
        xNext(myNeuron) = thresholdFun(a);
    end
    % f = figure('color','w');
    % imagesc(reshape(xNext,5,5))
    % keyboard
end
numIterations = numIterations - 1;

end
