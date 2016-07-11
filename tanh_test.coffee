# Test by creating a function that approximates tanh

fs = require 'fs'
_ = require 'lodash'
NeuralNet = require './neural_net.coffee'

## Test
teacher = (data) ->
	return (Math.tanh(_.sum(data)) + 1) / 2

nn = new NeuralNet.NeuralNet([5, 10, 1])

data = ((Math.random() * 2 - 1 for [0...5]) for [0...500])
diffs = []

for [0 .. 100]
	for row in data
		prediction = nn.predict(row)
		diffs.push Math.abs (prediction - teacher(row))
		nn.backpropagate(prediction, [teacher(row)])

# use the model to graph tanh
test_data = ((Math.random() * 2 - 1 for [0...5]) for [0...100])
myTanh = ([_.sum(row), nn.predict(row)] for row in test_data)

reportSuccess = (str="") -> (err) ->
	console.log "Error writing file", err if err
	console.log "Success writing to file #{str}"

fs.writeFile("./diffs.txt", diffs.join("\n"), reportSuccess "diffs")
fs.writeFile("./model.txt", nn.toString(), reportSuccess "model")
fs.writeFile("./test.txt", myTanh.join('\n'), reportSuccess "tanh")
