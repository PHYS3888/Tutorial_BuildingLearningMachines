function plotNeuronResponse(y,w,plotAsSurface,numPoints)
% plotNeuronResponse plots the response, y, of a single neuron to two inputs for
%                    a given set of weights, w.

%-------------------------------------------------------------------------------
% Check input parameters:
if nargin < 2
    w = [1,-1];
end
if nargin < 3
    plotAsSurface = true;
end
if nargin < 4
    numPoints = 20;
end
%-------------------------------------------------------------------------------

% (Note that the for loop can be avoided in the below using meshgrid and
% reshaping, but it reads more clearly in a for loop)

%-------------------------------------------------------------------------------
% Compute the input space and output values:
x1 = linspace(-5,5,numPoints);
x2 = linspace(-5,5,numPoints);
neuronOutput = zeros(numPoints);
for i = 1:numPoints
    for j = 1:numPoints
        neuronOutput(i,j) = y([x1(i);x2(j)],w);
    end
end

%-------------------------------------------------------------------------------
% Plot:
% f = figure('color','w');
if plotAsSurface
    surf(x1,x2,neuronOutput')
else
    imagesc(x1,x2,neuronOutput')
    axis('square')
    cB = colorbar();
    cB.Label.String = 'y';
end
xlabel('x1')
ylabel('x2')
zlabel('y')

% Set rainbow colormap:
giveMeTurboMap();

end
