using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LapTimer : MonoBehaviour
{
    private List<float> lapTimes = new List<float>();
    private float timer = 0.0f;
    private bool started = false;

    /// <summary>
    /// Returns the running lap time.
    /// </summary>
    /// <returns></returns>
    public float GetCurrentLapTime()
    {
        return timer;
    }

    /// <summary>
    /// Returns the lap times so far.
    /// </summary>
    /// <returns></returns>
    public List<float> GetLapTimes()
    {
        return new List<float>(lapTimes);
    }


    private void OnTriggerEnter(Collider other)
    {
        // Check that the car crosses the start line
        if (other.tag == "Car")
        {
            // Check whether it's the first lap
            if (started)
            {
                // Save lap time
                lapTimes.Add(timer);
                // Reset Timer to start a new lap
                timer = 0.0f;
            }
            else
            {
                started = true;
            }
        }
    }

    private void FixedUpdate()
    {
        if (started)
            timer += Time.fixedDeltaTime;
    }
}
