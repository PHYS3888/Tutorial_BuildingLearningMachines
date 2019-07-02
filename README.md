# Artificial Intelligence

This tutorial will walk you through some core concepts in artificial intelligence.

## Weight space of a single neuron

Recall that a sigmoidal function can be used to map a neuron's activation to an output value.

### The one-dimensional neuron
Let's first get an intuition for neuronal activation for a neuron with a single input, `x`.
The neuron's output, `y`, can be computed by defining an inline function as:

```matlab
y = @(x,w) 1/(1 + exp(-w*x));
```

#### TASK: one-dimensional neuron
Plot this function for some different values of the weight parameter, `w`.
What happens when `w = 0`?
How does a higher weight magnitude shape the neuron's response to the same input?
For the same weight magnitude, what does flipping its sign do to the neuron's response to the same input?

_(Hint: You can define a range for the input `x`, using `linspace`.)_

### The two-dimensional neuron

The same function, `y`, defined above can be used for multiple inputs, through the use of the inner product between the weight vector, `w`, and the input vector, `x`.

Above we had a one-dimensional weight space (we set a single parameter to define the neuron's behavior, the scalar `w`) and a one-dimensional input space (the single input `x`).
But now we have a two-dimensional weight space (we can set/learn two numbers, `w = [w1,w2]`, which define the neuron's response to two inputs, `x = [x1,x2]`).
Thus, each point in weight space (defined by two numbers, `[w1,w2]`) defines a function of the two inputs, `[x1,x2]`.

Let's try looking at some points in weight space:

#### Setting equal weight makes each input count the same in the neuron's response

(Note that the for loop can be avoided in the below using meshgrid and reshaping, but it reads clearly in a for loop):

```matlab
% Parameters:
w = [1,-1];
plotAsSurface = true;
numPoints = 20;

% Compute the input space and output values:
x1 = linspace(-5,5,numPoints);
x2 = linspace(-5,5,numPoints);
neuronOutput = zeros(numPoints);
for i = 1:numPoints
    for j = 1:numPoints
        neuronOutput(i,j) = y([x1(i);x2(j)],w);
    end
end

% Plot:
f = figure('color','w');
if plotAsSurface
    surf(x1,x2,neuronOutput')
else
    imagesc(x1,x2,neuronOutput)
end
xlabel('x1')
ylabel('x2')
zlabel('y')
```

QUESTION: When does the neuron have minimal output? Maximal output?
QUESTION: Repeat the above with `w = [1,0]`. What input is the neuron sensitive to?


### Training a single neuron to distinguish fashion models from sports stars

Imagine we have collected two pieces of information (number of instagram followers and resting heart rate) about a group of people, some of which are sports stars and some of which are fashion models.

We can represent the data, `X`, as a person by feature matrix, and a binary vector, `t`, to label each row as representing either a sports star (`0`) or a fashion models (`1`).

Load the data:

```matlab
load('ModelSportData.mat','dataMat','isModel')
```

Plot the data as a scatter, coloring each individual by their `sport`/`model` label:
```matlab
f = figure('color','w');
scatter(dataMat(:,1),dataMat(:,2),50,isModel,'filled');
colormap(jet)
xlabel('Number of instagram followers')
ylabel('Resting heart rate')
```

Let's see if our neuron can learn the difference between the two types of people by moving around in weight space.

```matlab
% Set learning rate
eta = 0.25;
numIterations = 100;

% Let's start assuming both measurements are equally important:
w = [-0.1,-0.1];
dataMatNorm = zscore(dataMat);
x1range = linspace(min(dataMatNorm(:,1)),max(dataMatNorm(:,1)),20);
x2range = linspace(min(dataMatNorm(:,2)),max(dataMatNorm(:,2)),20);

% Update weight using error on a given individual:
f = figure('color','w'); hold('on'); colormap('jet')
subplot(1,2,1); hold('on'); axis('square')
title('Neuron parameters')
S = scatter(w(1),w(2),50,1/1000,'filled');
xlabel('w1 (insta followers)'); ylabel('w2 (resting heart rate)')
subplot(1,2,2); axis('square'); hold on;
scatter(dataMatNorm(:,1),dataMatNorm(:,2),50,isModel,'filled');
H = plot(dataMatNorm(1,1),dataMatNorm(1,2),'xr','MarkerSize',20)
[~,C] = contour(x1,x2,neuronOutput',5);
xlabel('Instagram followers (normalized)')
ylabel('Resting heart rate (normalized)')
for i = 1:numIterations
    thePersonID = randi(40,1);

    % Compute the error estimate for this individual:
    xInd = dataMatNorm(thePersonID,:);
    yInd = isModel(thePersonID);
    yPred = y(xInd',w);
    yErr = yInd - yPred;

    % Adjust the weight vector according to this learning
    w = w + eta * yErr * xInd;

    % S.XData = w(1);
    % S.YData = w(2);
    % S.CData = i/1000;
    subplot(1,2,1);
    S = scatter(w(1),w(2),50,i/100,'filled');
    neuronOutput = computeNeuronOutput(w,x1range,x2range);
    C.ZData = neuronOutput';
    H.XData = dataMatNorm(thePersonID,1);
    H.YData = dataMatNorm(thePersonID,2);
    drawnow()
    pause(1)
end
```

Where in weight space did the neuron end up?
Between the two variables, what did the neuron learn to pay more attention to?

## Hopfield networks


### Training
Let's try to store four memories, corresponding to the letters `P`, `H`, `Y`, and `S` into a Hopfield network.
Each neuron is going to be a pixel in 5x5 grid, making 25 neurons in total.
Working with a binary Hopfield network, we note that each neuron's state can be either `-1` (inactive) or `1` (active).

Our first step is to define the memories on our 5x5 grid.
Check out the first few:
```matlab
plotMemory = true;
defineMemories('P',plotMemory);
defineMemories('H',plotMemory);
defineMemories('Y',plotMemory);
```
#### Question
Go into the code and see if you can define a new memory.
Give it a label, so that you can access the memory using `myMemory = defineMemories('myLabel')`.
It might help you to print out the index of the neurons in the grid you need to set to be active using:
```matlab
display(reshape(1:25,5,5))
```

Now let's train a Hopfield network to learn these five memories.

Define the memories:
```matlab
plotMatrix = false;
theMemories = {'P','H','Y','S','checker'};
% theMemories = {'D','J','C','M'};
numNeurons = 25;
numMemories = length(theMemories);
memoryMatrix = zeros(numNeurons,numMemories);
for i = 1:numMemories
    memoryMatrix(:,i) = defineMemories(theMemories{i},plotMatrix);
end
```

Compute weights under Hebbian learning rule:
```matlab
w = trainHopfieldWeights(memoryMatrix);
```

Plot the weights as a matrix:

```matlab
f = figure('color','w');
imagesc(w)
axis('square')
xlabel('neuron');
ylabel('neuron');
colormap([flipud(BF_getcmap('blues',numMemories));1,1,1;BF_getcmap('reds',numMemories)])
colorbar()
```

#### Question:
Can you plot the strongest weights as a graph?
HINT: You can make a graph object with a threshold on which weights to keep.
E.g., `strongGraph = graph(w>1);`

Look at what happens when you increase the threshold?
Do the most strongly connected groups of neurons make sense?
Repeat for the most strongly negatively correlated pairs of neurons.


#### Question
Weights are high between neurons that 'fire together' (i.e., pixels that tend to be on together or off together across the memories).

Inspect plots of the memories (displayed as a 5x5 grid):
```matlab
f = figure('color','w');
for i = 1:numMemories
    subplot(1,numMemories,i);
    defineMemories(theMemories{i},true);
    axis('square')
    title(sprintf('memory %u',i))
end
```

Find:
1. A pair of neurons that are always on together.
2. A pair of neurons that are always off together.
3. A pair of neurons that tend to be anticorrelated (when one is on the other is off)
4. A pair of neurons with no particular synchronization.

For each of these three pairs, predict what the weight will be in the trained Hopfield network (high positive, high negative, or zero).
Now confirm your intuition by checking the corresponding trained weight in `w`.

### Exploring stable states

Ok, no we have trained a binary Hopfield network with a set of memories, with the result stored in the weight matrix, `w`.

Let's look at some of the stable states of the trained network, by feeding it lots of random initial states and see what state the neurons settle down to.

```matlab
numRepeats = 20;
f = figure('color','w');
for i = 1:numRepeats
    startPoint = defineMemories('random',false);
    [finalPoint,numIters] = runHopfield(w,startPoint);
    subplot(4,5,i);
    imagesc(reshape(finalPoint,5,5));
    axis('square')
    colormap(flipud(gray))
end
```

Do any of these states resemble the memories we were trying to store?

### Restoring memories

Let's start this trained network near one of our memories.
We'll retrieve a memory, corrupt a random neuron, and see if feeding that corrupted memory into the network can restore us back to the trained memory:

```matlab
numRepeats = 5;
startNearMemoryNumber = 1;
startPointPure = memoryMatrix(:,startNearMemoryNumber);
f = figure('color','w');
for k = 1:numRepeats
    startPoint = startPointPure;
    corruptMe = randi(25);
    startPoint(corruptMe) = -startPoint(corruptMe); % corrupted
    [finalPoint,numIters] = runHopfield(w,startPoint);
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

How did it do?
What sort of stable states did you find? How do they relate to your memories?
Did you find any stable states that are inverses of your memories?
How did it get confused?

### Brain Damage
How robust is the network to some damage?
Start by having a guess as to the proportion of network weights that can be set to zero before the network's function breaks down.

Write a function `wCorrupted = brainDamage(w,propCorrupt)` that takes as input a trained weight matrix, `w`, and a proportion of weights to randomly set to zero `propCorrupt`, such that setting `propCorrupt = 0.1` sets 10% of the weights to zero.
HINT: The `squareform` function will help you unravel the upper triangle weights into a vector: `wVector = squareform(w)`, and can also be used to transform back to a zero-diagonal matrix, `w = squareform(wVector);`.
You might want to use the `randperm` function to randomly select elements to delete.

Now you can repeat the above exercise on memory restoration, but using the corrupted network defined by `wCorrupted`.
Qualitatively explore the robustness of the memory restoration capability as a function of the proportion of weights you set to zero.
At what proportion does the network start to break down?
How does this compare to a computer circuit?

ADVANCED (OPTIONAL): for each value of `propCorrupt`, quantify the associative memory performance, `meanMatch`, of the network by the average proportion of neurons that don't match the desired state.
Summarize the network's performance as a scatter plot of `propCorrupt` against `meanMatch`.
How is this curve dependent on the target memory?

### Task (optional): An overloaded network
How many memories can we squeeze into our 25-neuron network?
Repeat the above, adding new memories (e.g., by adding new cases to `defineMemories`), and see how an overtrained network can affect the stability of the desired memories.

# Improving on the Hebb rule

In lectures we recast the problem such that memories define a classification problem for each neuron.

Below is some simple code to implement this variant on the Hebb rule given a neuron x memory matrix of memories, `memoryMatrix`:

```matlab
% Set learning rate:
eta = 1;
% Convert memoryMatrix (-1/1) to binary memories (0/1):
binaryMemories = memoryMatrix;
binaryMemories(binaryMemories==-1) = 0;
% Define sigmoidal function:
sigmoid = @(x) 1/(1 + exp(-x));
% Start weights as they would be according to the Hebbian rule:
w0 = trainHopfieldWeights(memoryMatrix);

w = w0;
numIterations = 100;
for i = 1:numIterations
    % Ensure self-weights are zero:
    w(logical(eye(size(w)))) = 0;

    % Activations:
    a = w'*memoryMatrix;
    % Pass through a sigmoid:
    predictedOutputs = arrayfun(sigmoid,a);
    % Compute errors:
    neuronErrors = binaryMemories - predictedOutputs;
    % Compute gradients:
    wGrad = memoryMatrix*neuronErrors';
    % Symmetrize:
    wGradSym = wGrad + wGrad';
    % Update weights in direction of the gradient:
    w = w + eta*wGradSym;
end
```

Let's see if the weights have changed much from this process:
```matlab
f = figure('color','w');
subplot(1,3,1);
imagesc(w0)
title('Hebbian weights')

subplot(1,3,2);
imagesc(w)
title('Classification-trained weights')

subplot(1,3,3);
imagesc(w-w0)
title('Weight differences')

colormap([flipud(BF_getcmap('blues',numMemories));1,1,1;BF_getcmap('reds',numMemories)])
```

Question: are the weight changes focused on any particular neurons? Why might this be the case?
