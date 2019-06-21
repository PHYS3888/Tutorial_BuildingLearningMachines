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
