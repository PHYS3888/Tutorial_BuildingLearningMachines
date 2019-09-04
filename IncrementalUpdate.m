function IncrementalUpdate(y,dataMat,outputLabel,eta,w,numIterations,delayTime)

if nargin < 5
    % Default: assume that both measurements are equally important
    w = [-0.1,-0.1];
end
if nargin < 6
    numIterations = 100;
end
if nargin < 7
    delayTime = 1;
end

x1range = linspace(min(dataMat(:,1)),max(dataMat(:,1)),20);
x2range = linspace(min(dataMat(:,2)),max(dataMat(:,2)),20);

% Update weight using error on a given individual:
f = figure('color','w'); hold('on');
giveMeTurboMap()
subplot(1,2,1); hold('on'); axis('square')
title('Neuron parameters (weight space)')
S = scatter(w(1),w(2),50,1/1000,'filled');
xlabel('w1 (insta followers)'); ylabel('w2 (resting heart rate)')
subplot(1,2,2); axis('square'); hold on;
scatter(dataMat(:,1),dataMat(:,2),50,(outputLabel+0.5)/2,'filled');
H = plot(dataMat(1,1),dataMat(1,2),'xr','MarkerSize',20);
neuronOutput = computeNeuronOutput(w,x1range,x2range);
[~,C] = contour(x1range,x2range,neuronOutput',5);
xlabel('Instagram followers (normalized)')
ylabel('Resting heart rate (normalized)')
for i = 1:numIterations
    thePersonID = randi(40,1);

    % Compute the error estimate for this individual:
    xInd = dataMat(thePersonID,:);
    yInd = outputLabel(thePersonID);
    yPred = y(xInd',w);
    yErr = yInd - yPred;

    % How to adjust weights after learning from this individual:
    wAdjust = eta * yErr * xInd;

    % Adjust the weight vector according to this learning
    w = w + wAdjust;

    % S.XData = w(1);
    % S.YData = w(2);
    % S.CData = i/1000;
    subplot(1,2,1);
    S = scatter(w(1),w(2),50,i/100,'filled');
    plot([w(1)-wAdjust(1),w(1)],[w(2)-wAdjust(2),w(2)],':k')
    neuronOutput = computeNeuronOutput(w,x1range,x2range);
    C.ZData = neuronOutput';
    H.XData = dataMat(thePersonID,1);
    H.YData = dataMat(thePersonID,2);
    drawnow()
    pause(delayTime)
end

end
