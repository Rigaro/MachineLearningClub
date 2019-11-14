using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class ANN{

    public enum ActivationFunctionTypes
    {
        Step,
        Sigmoid,
        TanH,
        ReLu,
        LeakyReLu,
        Sinusoid,
        ArcTan,
        SoftSign
    }

	private int numInputs;
    private int numOutputs;
    private int numHidden;
    private int numNPerHidden;
    private double alpha;
    private List<Layer> layers = new List<Layer>();
    private ActivationFunctionTypes hiddenActFunType;
    private ActivationFunctionTypes outputActFunType;

    /// <summary>
    /// Creates a new Neural Network with the given number of inputs, outputs, hidden layers, number of neurons per layer, and learning rate.
    /// Allows to select the type of activation function for the hidden and output layers.
    /// </summary>
    /// <param name="nI">The number of inputs.</param>
    /// <param name="nO">The number of outputs.</param>
    /// <param name="nH">The number of hidden layers.</param>
    /// <param name="nPH">The number of neurons per layer.</param>
    /// <param name="alpha">The learning rate alpha.</param>
    /// <param name="hiddenType">The type of activation function for the hidden layers.</param>
    /// <param name="outputType">The type of activation function for the output layer.</param>
	public ANN(int nI, int nO, int nH, int nPH, double alpha, ActivationFunctionTypes hiddenType = ActivationFunctionTypes.Sigmoid, ActivationFunctionTypes outputType = ActivationFunctionTypes.Sigmoid)
	{
		numInputs = nI;
		numOutputs = nO;
		numHidden = nH;
		numNPerHidden = nPH;
		this.alpha = alpha;
        this.hiddenActFunType = hiddenType;
        this.outputActFunType = outputType;

		if(numHidden > 0)
		{
            // Create first hidden layer that receives inputs
            layers.Add(new Layer(numNPerHidden, numInputs));

            // Create hidden layers
			for(int i = 0; i < numHidden-1; i++)
			{
				layers.Add(new Layer(numNPerHidden, numNPerHidden));
			}

            // Create output layers
			layers.Add(new Layer(numOutputs, numNPerHidden));
		}
        // Create single I/O layer
		else
		{
			layers.Add(new Layer(numOutputs, numInputs));
		}
	}

    /// <summary>
    /// Trains the Neural Network with the given input-output dataset.
    /// </summary>
    /// <param name="inputValues">The inputs.</param>
    /// <param name="desiredOutput">The desired outputs.</param>
    /// <returns></returns>
	public List<double> Train(List<double> inputValues, List<double> desiredOutput)
	{
        List<double> outputValues = CalculateOutput(inputValues);
		UpdateWeights(outputValues, desiredOutput);
		return outputValues;
	}

    /// <summary>
    /// 
    /// </summary>
    /// <param name="inputValues"></param>
    /// <returns></returns>
	public List<double> CalculateOutput(List<double> inputValues)
	{
		List<double> inputs = new List<double>();
		List<double> outputs = new List<double>();
		int currentInput = 0; // Debug variable

        // Check input
		if(inputValues.Count != numInputs)
		{
			Debug.Log("ERROR: Number of Inputs must be " + numInputs);
			return outputs;
		}

		inputs = new List<double>(inputValues); // init inputs
        // Loop through all layers
		for(int lN = 0; lN < numHidden + 1; lN++)
		{
            // Set the output of the previous layer as the input to the new one
            if (lN > 0)
			{
				inputs = new List<double>(outputs);
			}
            // Clear outputs
			outputs.Clear();

            // Loop through all neurons in each layer
            for (int nN = 0; nN < layers[lN].numNeurons; nN++)
			{
                // Clear neuron inputs and input potential to populate with new data
                double inputPotential = 0;
				layers[lN].neurons[nN].inputs.Clear();

                // Loop through all inputs in each neuron
                for (int iN = 0; iN < layers[lN].neurons[nN].numInputs; iN++)
				{
                    // Set all inputs from the output of the previous layer
                    layers[lN].neurons[nN].inputs.Add(inputs[currentInput]);
                    // Update the inputPotential (dot product)
                    inputPotential += layers[lN].neurons[nN].weights[iN] * inputs[currentInput];
					currentInput++;
				}
                // Add bias
                inputPotential -= layers[lN].neurons[nN].bias;

                // Calculate neuron output and append to list
                if (lN == numHidden)
					layers[lN].neurons[nN].output = ActivationFunction(inputPotential, outputActFunType);
				else
					layers[lN].neurons[nN].output = ActivationFunction(inputPotential, hiddenActFunType);
					
				outputs.Add(layers[lN].neurons[nN].output); // Append output
				currentInput = 0; // Some debug stuff
			}
		}
		return outputs;
	}

    /// <summary>
    /// Prints the weights of the Neural Network to a string.
    /// </summary>
    /// <returns>The comma separated weights as a string.</returns>
	public string PrintWeights()
	{
		string weightStr = "";
		foreach(Layer l in layers)
		{
			foreach(Neuron n in l.neurons)
			{
				foreach(double w in n.weights)
				{
					weightStr += w + ",";
				}
				weightStr += n.bias + ",";
			}
		}
		return weightStr;
	}

    /// <summary>
    /// Loads the weights of the Neural Network from a string. Use after Parse from file.
    /// </summary>
    /// <param name="weightStr">The string containing the comma separated weights.</param>
	public void LoadWeights(string weightStr)
	{
		if(weightStr == "") return; // Return if empty
		string[] weightValues = weightStr.Split(','); // Split values
		int w = 0;
        // Loop through layers
		foreach(Layer l in layers)
		{
            // Loop through neurons.
			foreach(Neuron n in l.neurons)
			{
                // Loop through weights.
				for(int i = 0; i < n.weights.Count; i++)
				{
					n.weights[i] = System.Convert.ToDouble(weightValues[w]); //Write weights.
					w++;
				}
				n.bias = System.Convert.ToDouble(weightValues[w]); // Write bias.
				w++;
			}
		}
	}
	
    /// <summary>
    /// Updates the weights of the Neural Network through back-propagation.
    /// </summary>
    /// <param name="outputs">The computed output of the Neural Netowrk.</param>
    /// <param name="desiredOutput">The desired output.</param>
	void UpdateWeights(List<double> outputs, List<double> desiredOutput)
	{
		double error;
        // Loop backwards through all layers
		for(int lN = numHidden; lN >= 0; lN--)
		{
            // Loop through all neurons in each layer
			for(int nN = 0; nN < layers[lN].numNeurons; nN++)
			{
                // When reaching the output layer, use the Delta Rule: https://en.wikipedia.org/wiki/Delta_rule
                if (lN == numHidden)
				{
					error = desiredOutput[nN] - outputs[nN]; // Calculate output error
					layers[lN].neurons[nN].errorGradient = outputs[nN] * (1-outputs[nN]) * error; // Calculate error gradient
				}
                // Propagate through hidden layers
                else
                {
					layers[lN].neurons[nN].errorGradient = layers[lN].neurons[nN].output * (1-layers[lN].neurons[nN].output); // Calculate the error gradient (forward phase)
					double errorGradientSum = 0;
                    // (Backward Phase)
                    // Propagate the error from the next layer and accumulate it.
                    // Compute the sum of error gradients by looping through neurons in the next layer
                    for (int nNN = 0; nNN < layers[lN+1].numNeurons; nNN++)
					{
						errorGradientSum += layers[lN+1].neurons[nNN].errorGradient * layers[lN+1].neurons[nNN].weights[nN]; // Combine gradients
					}
					layers[lN].neurons[nN].errorGradient *= errorGradientSum;
				}	
                // Loop through inputs to each neuron to update weights
				for(int k = 0; k < layers[lN].neurons[nN].numInputs; k++)
				{
                    // Output layer
					if(lN == numHidden)
					{
						error = desiredOutput[nN] - outputs[nN]; // Calculate error
						layers[lN].neurons[nN].weights[k] += alpha * layers[lN].neurons[nN].inputs[k] * error; // Update weight
					}
                    // Hidden layers
					else
					{
						layers[lN].neurons[nN].weights[k] += alpha * layers[lN].neurons[nN].inputs[k] * layers[lN].neurons[nN].errorGradient; // Update weight
					}
				}
				layers[lN].neurons[nN].bias += alpha * -1 * layers[lN].neurons[nN].errorGradient; // Update bias
			}

		}

	}

    #region Activation functions

    /// <summary>
    /// Neural network activation function.
    /// https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
    /// </summary>
    /// <param name="input">The input to the activation function.</param>
    /// <param name="type">The type of activation function.</param>
    /// <returns>The output of the activation function.</returns>
    private double ActivationFunction(double input, ActivationFunctionTypes type)
    {
        switch (type)
        {
            case ActivationFunctionTypes.Step:
                return Step(input);
            case ActivationFunctionTypes.Sigmoid:
                return Sigmoid(input);
            case ActivationFunctionTypes.TanH:
                return TanH(input);
            case ActivationFunctionTypes.ReLu:
                return ReLU(input);
            case ActivationFunctionTypes.LeakyReLu:
                return LeakyReLU(input);
            case ActivationFunctionTypes.Sinusoid:
                return Sinusoid(input);
            case ActivationFunctionTypes.ArcTan:
                return ArcTan(input);
            case ActivationFunctionTypes.SoftSign:
                return SoftSign(input);
            default:
                return Sigmoid(input);
        }
    }

    private double Step(double value)
    {
        if (value < 0) return 0;
        return 1;
    }

    private double Sinusoid(double value)
    {
        return System.Math.Sin(value);
    }

    private double ArcTan(double value)
    {
        return System.Math.Atan(value);
    }

    private double SoftSign(double value)
    {
        return value / (1 + System.Math.Abs(value));
    }

    /// <summary>
    /// Hyperbolic Tangent.
    /// https://reference.wolfram.com/language/ref/Tanh.html
    /// </summary>
    /// <param name="input">The input.</param>
    /// <returns>The output.</returns>
	double TanH(double input)
	{
		double k = (double) System.Math.Exp(-2*input);
    	return 2 / (1.0f + k) - 1;
	}

    /// <summary>
    /// Rectifier Linear Unit (ReLU)
    /// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    /// https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
    /// https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7
    /// </summary>
    /// <param name="input">The input to the ReLU.</param>
    /// <returns>The output of the Leaky ReLU.</returns>
	double ReLU(double input)
	{
		if(input > 0) return input;
		else return 0;
	}

    /// <summary>
    /// Linear function.
    /// </summary>
    /// <param name="input">The input.</param>
    /// <returns>The output.</returns>
	double Linear(double input)
	{
		return input;
	}

    /// <summary>
    /// Leaky Rectifier Linear Unit (Leaky ReLU)
    /// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    /// https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/
    /// https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7
    /// </summary>
    /// <param name="input">The input to the Leaky ReLU.</param>
    /// <returns>The output of the Leaky ReLU.</returns>
	double LeakyReLU(double input)
	{
		if(input < 0) return 0.01*input;
   		else return input;
	}

    /// <summary>
    /// Sigmoid function:
    /// https://en.wikipedia.org/wiki/Sigmoid_function
    /// </summary>
    /// <param name="input">The input to the sigmoid function.</param>
    /// <returns>The output of the sigmoid function.</returns>
	double Sigmoid(double input) 
	{
    	double k = (double) System.Math.Exp(input);
    	return k / (1.0f + k);
	}

    #endregion

    #region File management

    public void SaveWeightsToFile()
    {
        string path = Application.dataPath + "/weights.txt";
        StreamWriter wf = File.CreateText(path);
        wf.WriteLine(PrintWeights());
        wf.Close();
    }

    public void LoadWeightsFromFile()
    {
        string path = Application.dataPath + "/weights.txt";
        StreamReader wf = File.OpenText(path);

        if (File.Exists(path))
        {
            string line = wf.ReadLine();
            LoadWeights(line);
        }
    }

    #endregion
}
