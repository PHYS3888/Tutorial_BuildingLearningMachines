
#### TASK: The One-dimensional Neuron

```matlab
x = linspace(-2,2,100);
y = @(x,w) 1./(1 + exp(-w*x));

f = figure('color','w');
hold('on')
plot(x,y(x,-2),'r')
plot(x,y(x,0),'k')
plot(x,y(x,2),'b')
plot(x,y(x,5),'g')
xlabel('Neuron input')
ylabel('Neuron output')
```

#### Task: plot network

```matlab
G = graph(w>2); % construct a graph object
p = plot(G); % plot the graph
```

### Task: brain damage

```matlab
propCorrupt = 0.4;

wVector = squareform(w);
numUniqueWeights = length(wVector);
numCorrupt = round(propCorrupt*numUniqueWeights);
rp = randperm(numUniqueWeights);
wCorruptVector = wVector;
wCorruptVector(rp(1:numCorrupt)) = 0;
wCorrupt = squareform(wCorruptVector);

numRepeats = 5;
startNearMemoryNumber = 2;
startPointPure = memoryMatrix(:,startNearMemoryNumber);


f = figure('color','w');
for k = 1:numRepeats
    startPoint = startPointPure;
    corruptMe = randi(25);
    startPoint(corruptMe) = -startPoint(corruptMe); % corrupted
    [finalPoint,numIters] = runHopfield(wCorrupt,startPoint);
    subplot(numRepeats,2,(k-1)*2+1);
    imagesc(reshape(startPoint,5,5));
    axis('square')
    title('Initial condition')
    subplot(numRepeats,2,(k-1)*2+2);
    imagesc(reshape(finalPoint,5,5));
    axis('square')
    title(sprintf('%u iterations',numIters))
    colormap(flipud(gray))
end
```
