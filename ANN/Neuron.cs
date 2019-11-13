using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron {

	public int numInputs;
	public double bias;
	public double output;
	public double errorGradient;
	public List<double> weights = new List<double>();
	public List<double> inputs = new List<double>();

    /// <summary>
    /// Creates and Artificial Neuron.
    /// </summary>
    /// <param name="nInputs">The number of inputs to the neuron.</param>
	public Neuron(int nInputs)
	{
		float weightRange = (float) 2.4/(float) nInputs;
		bias = UnityEngine.Random.Range(-weightRange,weightRange); // Create a random bias.
		numInputs = nInputs;

		for(int i = 0; i < nInputs; i++)
			weights.Add(UnityEngine.Random.Range(-weightRange,weightRange)); // Sets random weights to initialise it.
	}
}
