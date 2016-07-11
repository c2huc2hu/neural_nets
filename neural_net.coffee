# First implementation of a neural net
# Objective: predict output of various functions

# Backpropagation algorithm from wikipedia and from
# https://web.archive.org/web/20150317210621/https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

_ = require('lodash')
fs = require('fs')

LAYERS = 3
NEURONS = [3,4,1]

class Neuron
	constructor: (@parents) ->
		@children = undefined
		@output = undefined
		@weights = (Math.random() * 2 - 1 for i in @parents)
		@delta = 0 # derivative of error w.r.t weighted sum of outputs

	forward: ->
		@output = 0
		for parent, index in @parents
			@output += @weights[index] * parent.output
		@output = @activation(@output)

	activation: (x) -> # logistic activation function
		return 1 / (1 + Math.exp(-x))

class NeuralNet
	constructor: (numNeurons=[3,4,1]) ->
		@inputs = ({output: 0} for i in [0...numNeurons[0]])
		parents = @inputs

		# order must be preserved
		@neurons = []
		for n in numNeurons
			layer = (new Neuron(parents) for i in [0...n])
			for par in parents
				par.children = layer
			@neurons.push layer
			parents = layer

	predict: (inputs) ->
		for inp, index in inputs
			@inputs[index].output = inp.output or inputs[index]
		for layer in @neurons
			for neuron in layer
				neuron.forward()
		return @neurons[@neurons.length-1].map (x) -> x.output

	backpropagate: (outputs, desiredOutputs, stepSize=0.05) ->
		weightChanges = (((undefined for w in neuron.weights) for neuron in layer) for layer in @neurons)

		# final layer
		for [neuron, desiredOut], neuronIndex in _.zip(@neurons[@neurons.length-1], desiredOutputs)
			neuron.delta = (neuron.output - desiredOut) * neuron.output * (1 - neuron.output)
			for [weight, parent], weightIndex in _.zip(neuron.weights, neuron.parents)
				weightChanges[@neurons.length-1][neuronIndex][weightIndex] = -stepSize * parent.output * neuron.delta
				# neuron.output = O_j; parent.output = o_i
		# other layers
		for layer, layerIndex in @neurons[... -1] by -1
			for neuron, neuronIndex in layer
				neuron.delta = _.sum(child.delta * child.weights[neuronIndex] for child in neuron.children) * neuron.output * (1 - neuron.output)
				for [weight, parent], weightIndex in _.zip(neuron.weights, neuron.parents)
					weightChanges[layerIndex][neuronIndex][weightIndex] = -stepSize * parent.output * neuron.delta
					loop
						break if not _.isNaN(weight)

		# apply weight changes
		for layer, layerIndex in @neurons
			for neuron, neuronIndex in layer
				for weight, weightIndex in neuron.weights
					neuron.weights[weightIndex] += weightChanges[layerIndex][neuronIndex][weightIndex]

	toString: ->
		result = ""
		for layer, layerIndex in @neurons
			result += "Layer #{layerIndex} \n"
			for neuron, neuronIndex in layer
				result += "  Neuron #{layerIndex}-#{neuronIndex}\n"
				for weight, weightIndex in neuron.weights
					result += "    #{weight} \n"
				result += "    Outputs: #{neuron.output} \n"
		return result

module.exports = {NeuralNet}

