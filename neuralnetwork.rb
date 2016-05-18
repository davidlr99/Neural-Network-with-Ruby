require 'matrix'
require 'json'


class Matrix
  def []=(row, column, value)
    @rows[row][column] = value
  end
end

class NNetwork

  def saveNNet(name,net,bias)
    data = Hash.new
    data.store("net",net)
    data.store("bias",bias)

    serialized = Marshal.dump(data)
    File.open("#{name}", "w+") { |file| file.write(serialized) }
  end
  def readSavedNNet(name)
    data = Marshal.load File.read("#{name}")
    return data
  end
  def error(output,expected)
    output = output.values[-1].to_a[0]

    i = 0
    totalError = 0.0
    sum = 0.0

    for value in output
      e = 0.5*((expected[i]-value)**2)
      sum+=e
      i+=1
    end
    return sum
  end
  def sigmoid(x,reverse)
    y = 1.0 / ( 1.0 + Math::exp( -x ) )
    return y
  end

  def createNetwork(struct)
    net = Hash.new
    i = 0
    while i < struct.length-1
      thise = struct[i]
      nexte = struct[i+1]
      m = Matrix.build(thise, nexte) {|row, col| (rand(-10.0..10.0)/10.0).to_f }
      net.store(i+1,m)
      i+=1
    end
    return net
  end

  def run(input,net,bias,expected)
    lC  = 0.0
    allOutputsMatrix = Hash.new
    allOutputsMatrix.store(-1,[input])
    for layer in net.values

      l = input*layer
      i = 0
      eOfOutput1 = Array.new
      while i < l.column_count
        w  = l[0,i]
        l[0,i] = w+bias[lC]
        l[0,i] = sigmoid(l[0,i],false)
        z = l[0,i]
        e = z*(1.0-z)
        eOfOutput1.push(e)
        i+=1
      end

      allOutputsMatrix.store(lC,[l,eOfOutput1])

      input = l
      lC+=1
    end
    m = allOutputsMatrix[allOutputsMatrix.keys[-1]]
    e = m[0].to_a[0]
    i = 0
    errors = Array.new
    for value in e
      eOfOutput = -(expected[i]-value)
      errors.push(eOfOutput)
      i+=1
    end

    return allOutputsMatrix,errors,l
  end

  def createRandomBias(layerCount)
    bias = Array.new
    for v in 1..layerCount
      bias.push(rand(1..10).to_f/10.0)
    end
    return bias
  end

  def learn(net,bias,inputs,expecteds,untilError,learnrate)
    time = Time.now
    counter = 0.0
    while true
      total_error = 0.0
      inputN = 0.0
      for input in inputs
        expected = expecteds[inputN]
        output_of_neuron,errors = run(input,net,bias,expected)
        total_error += error(output_of_neuron,expected)

        value = errors
        l = 0
        for layer in net.values.reverse
          neurons = layer.to_a
          inputOfLayer = output_of_neuron.values.reverse[l+1][0]
          outputOfLayer = output_of_neuron.values.reverse[l][1]

          n = 0
          nextValue = Array.new

          for neuron in neurons
            inputOfNeuron = inputOfLayer.to_a[0][n]
            k = 0
            for kante in neuron
              outputOfNeuron = outputOfLayer[k]
              error = value[k]
              newKante = kante-learnrate*(inputOfNeuron*outputOfNeuron*error)
              neurons_chan = net.values.reverse[l]
              neurons_chan[n,k] = newKante
              net.values.reverse[l] = neurons_chan
              e1 = outputOfNeuron*error*kante

              old = nextValue[n].to_f
              newv = old+e1
              nextValue[n] = newv

              k+=1
            end
            n+=1
          end
          l+=1
          value = nextValue
        end
        counter+=1.0
        inputN+=1

      end
      print "Error: #{total_error}  Iterations: #{counter}\r"

      if total_error <= untilError
        time2 = Time.now
        puts "\n#{(time2-time)/60.0} Minutes.\r"
        return net
        break
      end
    end

    puts "#{counter} Iterations.                                              \r"
    return net
  end

end
