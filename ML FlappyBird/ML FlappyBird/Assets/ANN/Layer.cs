using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Layer {

	public int numNeurons;
	public List<Neuron> neurons = new List<Neuron>();

    /// <summary>
    /// Creates a Neural Network layer.
    /// </summary>
    /// <param name="nNeurons">The number of neurons in the layer.</param>
    /// <param name="numNeuronInputs">The number of inputs to each neuron.</param>
	public Layer(int nNeurons, int numNeuronInputs)
	{
		numNeurons = nNeurons;
		for(int i = 0; i < nNeurons; i++)
		{
			neurons.Add(new Neuron(numNeuronInputs));
		}
	}
}
