using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Brain : MonoBehaviour
{
    private ANN ann;
    private double sumSquareError = 0;

    // Start is called before the first frame update
    void Start()
    {
        ann = new ANN(2, 1, 1, 2, 0.8, ANN.ActivationFunctionTypes.SoftSign, ANN.ActivationFunctionTypes.Sigmoid);

        List<double> result;

        // Train network n times to do XOR
        for(int i = 0; i < 100000; i++)
        {
            sumSquareError = 0;
            result = Train(1, 1, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
            result = Train(1, 0, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
            result = Train(0, 1, 0);
            sumSquareError += Mathf.Pow((float)result[0] - 0, 2);
            result = Train(0, 0, 1);
            sumSquareError += Mathf.Pow((float)result[0] - 1, 2);
        }
        Debug.Log("SSE: " + sumSquareError);

        result = Process(1, 1, 0);
        Debug.Log(" 1 1 " + result[0]);
        result = Process(1, 0, 1);
        Debug.Log(" 1 0 " + result[0]);
        result = Process(0, 1, 1);
        Debug.Log(" 0 1 " + result[0]);
        result = Process(0, 0, 0);
        Debug.Log(" 0 0 " + result[0]);
    }

    private List<double> Train(double i1, double i2, double o)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();
        inputs.Add(i1);
        inputs.Add(i2);
        outputs.Add(o);
        return (ann.ProcessData(inputs, outputs, true));
    }

    private List<double> Process(double i1, double i2, double o)
    {
        List<double> inputs = new List<double>();
        List<double> outputs = new List<double>();
        inputs.Add(i1);
        inputs.Add(i2);
        outputs.Add(o);
        return (ann.ProcessData(inputs, outputs));
    }
}
