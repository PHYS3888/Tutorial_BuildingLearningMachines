function wCorrupted = brainDamage(w,propCorrupt)

if nargin < 2
    propCorrupt = 0.5;
end

% Reshape the w matrix upper diagonal into a vector:
wVector = squareform(w);
numUniqueWeights = length(wVector);

% Corrupt a given proportion of the weights to zero:
numCorrupt = round(propCorrupt*numUniqueWeights);
rp = randperm(numUniqueWeights);
wCorruptVector = wVector;
wCorruptVector(rp(1:numCorrupt)) = 0;

% Reshape back to a matrix:
wCorrupted = squareform(wCorruptVector);

end
