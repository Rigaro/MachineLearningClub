using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TrafficLightController : MonoBehaviour
{
    public Light redLight;
    public Light yellowLight;
    public Light greenLight;

    public bool redOn;
    public bool yellowOn;
    public bool greenOn;


    private void Update()
    {
        if (redLight != null && yellowLight != null && greenLight != null)
        {
            if (redOn)
            {
                redLight.enabled = true;
                yellowLight.enabled = false;
                greenLight.enabled = false;
                redOn = false;
                yellowOn = false;
                greenOn = false;
            }
            else if (yellowOn)
            {
                redLight.enabled = false;
                yellowLight.enabled = true;
                greenLight.enabled = false;
                redOn = false;
                yellowOn = false;
                greenOn = false;
            }
            else if (greenOn)
            {
                redLight.enabled = false;
                yellowLight.enabled = false;
                greenLight.enabled = true;
                redOn = false;
                yellowOn = false;
                greenOn = false;
            }
        }
    }
}
