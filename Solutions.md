
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

#### Task:
