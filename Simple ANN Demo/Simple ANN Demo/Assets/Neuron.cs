using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Neuron
{
    public int numInputs;
    public double bias;
    public double output;
    public double errorGradient;
    public List<double> weights = new List<double>();
    public List<double> inputs = new List<double>();

    /// <summary>
    /// Creates a neuron with the given number of inputs with random bias and weights
    /// </summary>
    /// <param name="nInputs">The number of inputs</param>
    public Neuron(int nInputs)
    {
        bias = Random.Range(-1.0f, 1.0f);
        numInputs = nInputs;
        for (int i = 0; i < nInputs; i++)
            weights.Add(Random.Range(-1.0f, 1.0f));
    }
}
