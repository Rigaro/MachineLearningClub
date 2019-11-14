using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 
/// </summary>
public interface IExternalSensor
{
    /// <summary>
    /// Returns the sensor data at the current time-step
    /// </summary>
    /// <returns>The sensor data</returns>
    double[] GetSensorData();
}
