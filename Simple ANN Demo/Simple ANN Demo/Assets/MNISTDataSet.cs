using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

/// <summary>
/// MNIST dataset object, modified from:
/// https://github.com/JoseLuisRojasAranda/Unity3D-MNIST-NN
/// </summary>
public class MNISTDataSet : MonoBehaviour
{
    public float[] values; // The active values
    public int label; // The active label
    public int[] desiredOutput; // The active desiredOutput

    private float[][] trainingValues; // Extracted image pixel data
    private float[] trainingLabels; // Labels
    private float[][] testValues;
    private float[] testLabels;
    private Texture2D texture;


    // Start is called before the first frame update
    void Start()
    {
        trainingLabels = new float[60000];
        trainingValues = new float[60000][];

        testLabels = new float[10000];
        testValues = new float[10000][];

        desiredOutput = new int[10];

        for (int v = 0; v < trainingValues.Length; v++) // initialise training array
            trainingValues[v] = new float[28 * 28];

        for (int v = 0; v < testValues.Length; v++) // initialise test array
            testValues[v] = new float[28 * 28];

        LoadTrainingData("train-labels.idx1-ubyte", "train-images.idx3-ubyte");
        LoadTestData("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte");

        // Create the image to display the active number
        texture = new Texture2D(28, 28);
        GetComponent<SpriteRenderer>().material.mainTexture = texture;

        SetActiveData(100, true);
    }

    /// <summary>
    /// Sets the active data that is accessible externally
    /// </summary>
    /// <param name="number">The data-set entry number.</param>
    /// <param name="isTraining">Trainin dataset (true) or test dataset (false)</param>
    public void SetActiveData(int number, bool isTraining)
    {
        if(isTraining)
        {
            values = trainingValues[number-1];
            label = (int)trainingLabels[number-1];
            UpdateDesiredOutput();
            UpdateTexture();
        }
    }

    /// <summary>
    /// Coverts label to output and updates active.
    /// </summary>
    private void UpdateDesiredOutput()
    {
        for (int i = 0; i < 10; i++)
        {
            if (i == label)
                desiredOutput[i] = 1;
            else
                desiredOutput[i] = 0;
        }
    }

    /// <summary>
    /// Updates the texture that displays the current number
    /// </summary>
    private void UpdateTexture()
    {
        for(int x = 0; x < 28; x++)
        {
            for (int y = 0; y < 28; y++)
            {
                float pixelValue = values[(28 * x) + y];
                texture.SetPixel(x, y,new Color(pixelValue, pixelValue, pixelValue));
            }
        }
        
        texture.Apply();
        Sprite sprite = Sprite.Create(texture, new Rect(0.0f, 0.0f, texture.width, texture.height), new Vector2(0.5f, 0.5f));
        GetComponent<SpriteRenderer>().sprite = sprite;
    }

    /// <summary>
    /// Loads the training data.
    /// </summary>
    /// <param name="labelsFilename">The name of the file containing training labels.</param>
    /// <param name="imagesFilename">The name of the file containing training images.</param>
    void LoadTrainingData(string labelsFilename, string imagesFilename)
    {
        // Open the files
        FileStream ifsLabels = File.Open(Application.dataPath + "/MNIST/" + labelsFilename, FileMode.Open);
        FileStream ifsImages = File.Open(Application.dataPath + "/MNIST/" + imagesFilename, FileMode.Open);
        // Create the binary readers for each data file
        BinaryReader brLabels = new BinaryReader(ifsLabels);
        BinaryReader brImages = new BinaryReader(ifsImages);

        int magic1 = brImages.ReadInt32(); // Discard the first value in the images binary and go to the next value
        int numImages = brImages.ReadInt32(); // Read the number of images
        int numRows = brImages.ReadInt32(); // Read the number of rows
        int numCols = brImages.ReadInt32(); // Read the number of columns

        int magic2 = brLabels.ReadInt32(); // Discard the first value in the labels binary
        int numLabels = brLabels.ReadInt32(); // Read the number of labels (should be 60k)

        // Initialise image pixel array
        byte[][] pixels = new byte[28][];
        for (int i = 0; i < pixels.Length; ++i)
            pixels[i] = new byte[28];

        // Loop through all data
        for (int di = 0; di < 60000; ++di)
        {
            // loop in the x direction
            for (int i = 0; i < 28; ++i)
            {
                // Loop in the y direction
                for (int j = 0; j < 28; ++j)
                {
                    byte b = brImages.ReadByte(); // Read pixel
                    pixels[i][j] = b;
                    trainingValues[di][(28 * i) + j] = pixels[i][j]; // Flatten to be able to use it as input to neural net.
                }
            }

            byte lbl = brLabels.ReadByte(); // Read the label
            trainingLabels[di] = lbl; // Set the label value
        } // each image
    }

    /// <summary>
    /// Loads test data.
    /// </summary>
    /// <param name="labelFilename">The name of the file containing test labels.</param>
    /// <param name="imagesFilename">The name of the file containing test images.</param>
    void LoadTestData(string labelFilename, string imagesFilename)
    {

        FileStream ifsLabels = File.Open(Application.dataPath + "/MNIST/" + labelFilename, FileMode.Open);
        FileStream ifsImages = File.Open(Application.dataPath + "/MNIST/" + imagesFilename, FileMode.Open);

        BinaryReader brLabels = new BinaryReader(ifsLabels);
        BinaryReader brImages = new BinaryReader(ifsImages);

        int magic1 = brImages.ReadInt32(); // discard
        int numImages = brImages.ReadInt32();
        int numRows = brImages.ReadInt32();
        int numCols = brImages.ReadInt32();

        int magic2 = brLabels.ReadInt32();
        int numLabels = brLabels.ReadInt32();

        byte[][] pixels = new byte[28][];
        for (int i = 0; i < pixels.Length; ++i)
            pixels[i] = new byte[28];

        for (int di = 0; di < 10000; ++di)
        {
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    byte b = brImages.ReadByte();
                    pixels[i][j] = b;
                    testValues[di][(28 * i) + j] = pixels[i][j];
                }
            }

            byte lbl = brLabels.ReadByte();
            testLabels[di] = lbl;
        } // each image
    }
}
