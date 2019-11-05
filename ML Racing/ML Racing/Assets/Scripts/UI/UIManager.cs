using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class UIManager : MonoBehaviour
{
    public TextMeshProUGUI lapTextArea;

    private LapTimer lapTimer;
    // Start is called before the first frame update
    void Start()
    {
        lapTimer = GameObject.FindGameObjectWithTag("Start").GetComponent<LapTimer>();
    }

    // Update is called once per frame
    void Update()
    {
        UpdateLapInformation();
    }

    private void UpdateLapInformation()
    {
        string lapInfo = "Lap data:\n";
        int currentLapNumber = lapTimer.GetLapTimes().Count + 1;
        lapInfo += currentLapNumber + ": " + System.Math.Round(lapTimer.GetCurrentLapTime(), 3) + "\n";
        for (int i = lapTimer.GetLapTimes().Count - 1; i >= 0; i--)
        {
            lapInfo += (i+1) + ": " + System.Math.Round(lapTimer.GetLapTimes()[i], 3) + "\n";
        }
        lapTextArea.text = lapInfo;
    }
}
