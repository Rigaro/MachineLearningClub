using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.Linq;

public class MNISTBrainTemplate : MonoBehaviour
{
    public MNISTDataSet dataset;
    public int batchSize = 100;
    public int epochs = 500;
    public TextMeshPro resultDisplay;

    private int testDataNumber = 1;
    private ANN ann;
    private double sumSquareError = 0;
    private int trainingDatasetSize = 60000;
    private int testDatasetSize = 10000;

    // Start is called before the first frame update
    void Start()
    {
        //
        // Write your code here! This template code does not work.
        // You'll need to rename the class and file for this to run in unity.
        // Just remove the "Template" part from the file name and the class name.
        //
        List<double> output; // 

        // Create your ANN
        ann = new ANN(2, 2, 1, 2, 0.1, ANN.ActivationFunctionTypes.SoftSign, ANN.ActivationFunctionTypes.Step);

        // Train network n times (epochs) with dataset. You can train in batchs as well, just use the batch variable inside.
        for (int e = 0; e < epochs; e++)
        {
            sumSquareError = 0; // Used to get the error from your net

            // Pick a random image from the training data-set
            int dataNum = (int)Mathf.Floor(Random.Range(0.0f, trainingDatasetSize));
            dataset.SetActiveData(dataNum, true);

            // Create training lists here
            List<double> inputs = new List<double>();
            List<double> desiredOutput = new List<double>();
            inputs.AddRange(dataset.values);
            desiredOutput.AddRange(dataset.desiredOutput);

            // Run the training routine
            output = ann.ProcessData(inputs, desiredOutput, true);

            // Calculate estimation error for debug
            for (int o = 0; o < output.Count; o++)
            {
                sumSquareError += Mathf.Pow((float)output[o] - (float)desiredOutput[o], 2);
            }
        }
        Debug.Log("Final SSE: " + sumSquareError);
    }

    private void Update()
    {
        // This is used to display the results, no need to change.

        // Update selected image number
        dataset.SetActiveData(testDataNumber, false);
        // Create lists for processing (output not needed but this is not the final ANN implementation).
        List<double> inputs = new List<double>();
        List<double> desiredOutput = new List<double>();
        inputs.AddRange(dataset.values);
        desiredOutput.AddRange(dataset.desiredOutput);
        // Get output from network
        List<double> output = ann.ProcessData(inputs, desiredOutput, true);

        // Get result and display it
        double max = output.Max(); // Get the maximum (choice) from output
        int maxIndex = output.ToList().IndexOf(max); // Get the index of the maximum value, which is the result!
        resultDisplay.text = maxIndex.ToString();
    }

    /// <summary>
    /// Handles test image selection from GUI
    /// </summary>
    /// <param name="value"></param>
    public void ImageSelection(string value)
    {
        int imageNumber = System.Convert.ToInt32(value); // Convert to int
        // Clamp to dataset size
        if (imageNumber < 1)
            imageNumber = 1;
        else if (imageNumber > 10000)
            imageNumber = 10000;

        // Selector!
        testDataNumber = imageNumber;
    }
}
