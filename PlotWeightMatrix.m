function PlotWeightMatrix(w)

numGrads = 5;

f = figure('color','w');
imagesc(w)
axis('square')
xlabel('neuron');
ylabel('neuron');
colormap([flipud(BF_getcmap('blues',numGrads));1,1,1;BF_getcmap('reds',numGrads)])
colorbar()

end
