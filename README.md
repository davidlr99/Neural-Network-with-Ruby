# Neural-Network-with-Ruby
A simple Neural Network written in Ruby.
Usage:
```
nnt = NNetwork.new

netStructure = [2,20,2] #The network structure e.g. 2 input neurons, 20 hidden neurons, 2 output neurons

learnrate = 0.1  # The learnrate
untilError = 0.0001 #The network is trained until it reaches this total error.
bias = nnt.createRandomBias(netStructure.length) #The bias for the neurons.

input = [Matrix[[0.05,0.10]],Matrix[[0.5,0.1]]] #The input array (type of elements: Matrix).
expected = [[0.5,0.5],[0.3,0.7]] #The expected values array (e.g for the Matrix[[0.05,0.10]] the expected output is [0.5,0.5]).

nntFrame = nnt.createNetwork(netStructure) #Generates random weights.

netTrained = nnt.learn(nntFrame,bias,input,expected,untilError,learnrate) #The network is trained with backpropagation.

layerOutput,errors,out = nnt.run(input[1],netTrained,bias,expected[1])  #Run the trained network, with one input and one expected output.
puts "Output: #{out.to_a}" #Print the output
```
