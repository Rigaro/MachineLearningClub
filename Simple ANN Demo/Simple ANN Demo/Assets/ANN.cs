using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ANN 
{
    public int numInputs;
    public int numOutputs;
    public int numHidden;
    public int numNPerHidden;
    public double alpha; // learning rate kinda parameter
    public ActivationFunctionTypes hiddenType;
    public ActivationFunctionTypes outputType;

    private List<Layer> layers = new List<Layer>();

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

    /// <summary>
    /// 
    /// </summary>
    /// <param name="nI"></param>
    /// <param name="nO"></param>
    /// <param name="nH"></param>
    /// <param name="nPH"></param>
    /// <param name="a"></param>
    public ANN(int nI, int nO, int nH, int nPH, double a, ActivationFunctionTypes hiddenType, ActivationFunctionTypes outputType)
    {
        numInputs = nI;
        numOutputs = nO;
        numHidden = nH;
        numNPerHidden = nPH;
        alpha = a;
        this.hiddenType = hiddenType;
        this.outputType = outputType;


        if (numHidden > 0)
        {
            // Create first hidden layer that receives inputs from input layer
            layers.Add(new Layer(numNPerHidden, numInputs));
            // Create hidden layers
            for (int i = 0; i < numHidden - 1; i++)
            {
                layers.Add(new Layer(numNPerHidden, numNPerHidden));
            }
            // Create output layer
            layers.Add(new Layer(numOutputs, numNPerHidden));
        }
        else
        {
            // Just create a single layer
            layers.Add(new Layer(numOutputs, numInputs));
        }


    }

    public List<double> ProcessData(List<double> inputValues, List<double> desiredOutput, bool isTraining = false)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();

        // Check input
        if (inputValues.Count != numInputs)
            throw new System.ArgumentException("ERROR: The number of training data inputs must be " + numInputs);

        inputs = new List<double>(inputValues);
        // Loop through all layers
        for (int lN = 0; lN < numHidden + 1; lN++)
        {
            // Set the output of the previous layer as the input to the new one.
            if (lN > 0)
            {
                inputs = new List<double>(outputs);
            }
            // Clear outputs
            outputs.Clear();

            // Loop through all neurons in each layer
            for (int nN = 0; nN < layers[lN].numNeurons; nN++)
            {
                // Clear neuron inputs and input potential to populate with training data
                double inputPotential = 0;
                layers[lN].neurons[nN].inputs.Clear();

                // Loop through all inputs in each neuron
                for (int iN = 0; iN < layers[lN].neurons[nN].numInputs; iN++)
                {
                    // Set all inputs from the output of the previous layer
                    layers[lN].neurons[nN].inputs.Add(inputs[iN]);
                    // Update the inputPotential (dot product)
                    inputPotential += layers[lN].neurons[nN].weights[iN] * inputs[iN];
                }
                // Add bias
                inputPotential -= layers[lN].neurons[nN].bias;
                // Calculate neuron output and append to list
                if (lN == numHidden)
                    layers[lN].neurons[nN].output = ActivationFunction(inputPotential, outputType);
                else
                    layers[lN].neurons[nN].output = ActivationFunction(inputPotential, hiddenType);

                outputs.Add(layers[lN].neurons[nN].output);
            }
        }

        // Update Network when receiving training data
        if (isTraining)
            UpdateWeights(outputs, desiredOutput);

        return outputs;
    }


    /// <summary>
    /// Use back-propagation to update weights
    /// </summary>
    /// <param name="outputs"></param>
    /// <param name="desiredOutput"></param>
    private void UpdateWeights(List<double> outputs, List<double> desiredOutput)
    {
        double error;
        // Loop backwards through all layers
        for(int lN = numHidden; lN >= 0; lN--)
        {
            // Loop through all neurons
            for (int nN = 0; nN < layers[lN].numNeurons; nN++)
            {
                // If output layer then use Delta Rule
                if (lN == numHidden)
                {
                    // Compute the output error and its gradient
                    error = desiredOutput[nN] - outputs[nN];
                    layers[lN].neurons[nN].errorGradient = outputs[nN] * (1 - outputs[nN]) * error;
                }
                // Propagate through hidden layers
                else
                {
                    layers[lN].neurons[nN].errorGradient = layers[lN].neurons[nN].output * (1 - layers[lN].neurons[nN].output);
                    // Compute the sum of error gradients by looping through neurons in the next layer
                    double errorGradientSum = 0;
                    int nLN = lN + 1; // next layer number
                    for (int nNN = 0; nNN < layers[nLN].numNeurons; nNN++)
                    {
                        errorGradientSum += layers[nLN].neurons[nNN].errorGradient * layers[nLN].neurons[nNN].weights[nN];
                    }
                    layers[lN].neurons[nN].errorGradient *= errorGradientSum;
                }
                // Loop through all inputs in each neuron
                for (int iN = 0; iN < layers[lN].neurons[nN].numInputs; iN++)
                {
                    if (lN == numHidden)
                    {
                        // Compute the output error and update weights
                        error = desiredOutput[nN] - outputs[nN];
                        layers[lN].neurons[nN].weights[iN] += alpha * layers[lN].neurons[nN].inputs[iN] * error;
                    }
                    else
                    {
                        layers[lN].neurons[nN].weights[iN] += alpha * layers[lN].neurons[nN].inputs[iN] * layers[lN].neurons[nN].errorGradient;
                    }
                }
                // Update bias
                layers[lN].neurons[nN].bias -= alpha * layers[lN].neurons[nN].errorGradient;
            }
        }
    }

    private double ActivationFunction(double value, ActivationFunctionTypes type)
    {
        switch (type)
        {
            case ActivationFunctionTypes.Step:
                return Step(value);
            case ActivationFunctionTypes.Sigmoid:
                return Sigmoid(value);
            case ActivationFunctionTypes.TanH:
                return TanH(value);
            case ActivationFunctionTypes.ReLu:
                return ReLu(value);
            case ActivationFunctionTypes.LeakyReLu:
                return LeakyReLu(value);
            case ActivationFunctionTypes.Sinusoid:
                return Sinusoid(value);
            case ActivationFunctionTypes.ArcTan:
                return ArcTan(value);
            case ActivationFunctionTypes.SoftSign:
                return SoftSign(value);
            default:
                return Sigmoid(value);
        }
    }

    private double Step(double value)
    {
        if (value < 0) return 0;
        return 1;
    }

    private double Sigmoid(double value)
    {
        double k = (double)System.Math.Exp(-value);
        return 1 / (1.0f + k);
        /*
        double k = (double)System.Math.Exp(value);
        return k / (1.0f + k);
         */
    }

    private double TanH(double value)
    {
        return (2 * (Sigmoid(2 * value)) - 1);
    }

    private double ReLu(double value)
    {
        if (value > 0) return value;
        else return 0;
    }

    private double LeakyReLu(double value)
    {
        if (value < 0) return 0.01 * value;
        else return value;
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
}
