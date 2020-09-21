function InspectMemories()
% Plot the five 5x5 memories with the 25 neurons (pixels) labeled
%-------------------------------------------------------------------------------

f = figure('color','w');
theMemories = {'P','H','Y','S','checker'};
numMemories = length(theMemories);
for i = 1:numMemories
    ax = subplot(1,numMemories,i);
    defineMemories(theMemories{i},true);
    axis('square')
    title(sprintf('memory %u',i))
    labelIndices();
    ax.XTick = [];
    ax.YTick = [];
end
f.Position(3:4) = [850 250];

end
