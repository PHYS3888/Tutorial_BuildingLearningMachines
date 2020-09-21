function memoryRestore(memoryMatrix,w,startNearMemoryNumber)

if nargin < 3
    startNearMemoryNumber = 5;
end
%-------------------------------------------------------------------------------

numRepeats = 5;
startPointPure = memoryMatrix(:,startNearMemoryNumber);
f = figure('color','w');
colormap(flipud(gray))
for k = 1:numRepeats
    startPoint = flipALittle(startPointPure);

    % Simulate network dynamics until an equilibrium is found:
    [finalPoint,numIters] = runHopfield(w,startPoint);

    % Plot the initial (corrupted) state:
    subplot(numRepeats,2,(k-1)*2+1);
    imagesc(reshape(startPoint,5,5));
    axis('square')
    title('Initial condition')

    % Plot the final (equilibrium) state:
    subplot(numRepeats,2,(k-1)*2+2);
    imagesc(reshape(finalPoint,5,5));
    axis('square')
    title(sprintf('%u iterations',numIters))
end

f.Position(3:4) = [400 700];

end
