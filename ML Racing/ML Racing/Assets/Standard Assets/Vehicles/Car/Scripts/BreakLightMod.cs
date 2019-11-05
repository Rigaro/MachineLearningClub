using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityStandardAssets.Vehicles.Car;

public class BreakLightMod : MonoBehaviour
{
    public CarController car; // reference to the car controller, must be dragged in inspector

    public Light rightLight;
    public Light leftLight;


    private void Update()
    {
        if (rightLight != null & leftLight != null)
        {
            // enable the light when the car is braking, disable it otherwise.
            rightLight.enabled = car.BrakeInput > 0f;
            leftLight.enabled = car.BrakeInput > 0f;
        }
    }
}
