function corruptedMemory = flipALittle(pureMemory)
% flipALittle flips a random neuron's state from the pureMemory

% Count the number of neurons making up the memory:
numNeurons = length(pureMemory);

% Pick a random neuron:
corruptThisNeuron = randi(numNeurons);

% Copy the pure memory into the corrupted memory, and then
% flip the state of corruptThisNeuron:
corruptedMemory = pureMemory;
% XXX FILL IN THIS LINE XXX

end
