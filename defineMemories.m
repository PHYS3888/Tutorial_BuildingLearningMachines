function myGrid = defineMemories(whatLetter,doPlot)

if nargin < 2
    doPlot = true;
end
%-------------------------------------------------------------------------------

% Define a baseline -1 (binary off) grid for the memory:
myGrid = -ones(5,5);
index = reshape(1:25,5,5);

% Make the letter:
switch whatLetter
case 'P'
    myGrid([1:5,6,8,11,13,16,18,21:23]) = 1;
case 'H'
    myGrid([6:10,13,16:20]) = 1;
case 'Y'
    myGrid([1,5,7,9,13,17,21]) = 1;
case 'S'
    myGrid([2,5,6,8,10,11,13,15,16,18,20,21,24]) = 1;
case {'checker','checkerboard'}
    % Make a checkerboard:
    myGrid(1:2:end) = 1;
case 'D'
    myGrid([1,6:10,11,15,16,20,22:24]) = 1;
case 'C'
    myGrid([2:4,6,10,11,15,16,20,21,25]) = 1;
case 'J'
    myGrid([1,4,6,10,11,15,16:19,21]) = 1;
case 'M'
    myGrid([1:5,7,13,17,21:25]) = 1;
case 'random'
    turnMeOn = (rand(25,1) > 0.5);
    myGrid(turnMeOn) = 1;
otherwise
    error('Unknown letter ''%s''',whatLetter);
end

% Plot it:
if doPlot
    imagesc(myGrid)
    colormap(flipud(gray))
end

% Expand out to vector:
myGrid = myGrid(:);

end
