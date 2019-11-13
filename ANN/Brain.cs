using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class Replay
{
    public List<double> states;
    public double reward;

    public Replay(double xr, double ballx, double ballvx, double r)
    {
        states = new List<double>();
        states.Add(xr);
        states.Add(ballx);
        states.Add(ballvx);
        reward = r;
    }
}

public class Brain : MonoBehaviour
{
    public GameObject ball;                                          // Balance ball monitor

    private ANN ann;

    private float _reward = 0.0f;                                    // Reward to associate to actions
    private List<Replay> _replayMemory = new List<Replay>();         // Memory - list of past actions and rewards.
    private int _mCapacity = 10000;                                  // Memory capacity

    private float _discount = 0.99f;                                 // How much future states affect rewards
    private float _exploreRate = 100.0f;                             // Chance of picking a random action
    private float _maxExploreRate = 100.0f;                          // Max chance value
    private float _minExploreRate = 0.01f;                           // Min chance value
    private float _exploreDecay = 0.0001f;                           // Chance decay amount per iteration/update

    private Vector3 _ballStartPos;                                   // Save ball start position
    private int _failCount = 0;                                      // The number of failed attempts (ball dropped)
    /// <summary>
    /// The max angle to add to the tilt at each update.
    /// This mus be large enough so that the Q-value multiplied
    /// by it is enough to recover balance when the ball speeds up
    /// </summary>
    private float _tiltSpeed = 0.5f;

    private float _timer = 0.0f;                                      // Timer to keep track of time balancing ball
    private float _maxBalanceTime = 0.0f;                              // The best time.

    // Start is called before the first frame update
    void Start()
    {
        // Create the ANN, get ball position, and speed up simulation.
        ann = new ANN(3, 2, 1, 6, 0.2f);
        _ballStartPos = ball.transform.position;
        Time.timeScale = 5.0f;
    }

    private GUIStyle _guiStyle = new GUIStyle();
    private void OnGUI()
    {
        _guiStyle.fontSize = 25;
        _guiStyle.normal.textColor = Color.white;
        GUI.BeginGroup(new Rect(10, 10, 600, 150));
        GUI.Box(new Rect(0, 0, 140, 140), "Stats", _guiStyle);
        GUI.Label(new Rect(10, 25, 500, 30), "Fails: " + _failCount, _guiStyle);
        GUI.Label(new Rect(10, 50, 500, 30), "Explore rate: " + _exploreRate, _guiStyle);
        GUI.Label(new Rect(10, 75, 500, 30), "Last best time: " + _maxBalanceTime, _guiStyle);
        GUI.Label(new Rect(10, 100, 500, 30), "Balance time: " + _timer, _guiStyle);
        GUI.EndGroup();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Space))
            ResetBall();
        
    }

    private void ResetBall()
    {
        ball.transform.position = _ballStartPos;
        ball.GetComponent<Rigidbody>().velocity = Vector3.zero;
        ball.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
    }

    private void FixedUpdate()
    {
        // Initialise local variables and update balance time
        _timer += Time.deltaTime;
        List<double> states = new List<double>();
        List<double> qs = new List<double>();

        // Update states list
        states.Add(this.transform.rotation.x); // platform rotation
        states.Add(ball.transform.position.z); // ball position
        states.Add(ball.GetComponent<Rigidbody>().angularVelocity.x); // ball angular velocity

        // Q data
        qs = SoftMax(ann.CalculateOutput(states)); // SoftMax formats the output
        double maxQ = qs.Max(); // Find maximum Q out of states
        int maxQIndex = qs.ToList().IndexOf(maxQ); // Get the index of the maximum value
        _exploreRate = Mathf.Clamp(_exploreRate - _exploreDecay, _minExploreRate, _maxExploreRate);

        // Choose whether to randomly explore
        //if (Random.Range(0, 100) < _exploreRate)
        //    maxQIndex = Random.Range(0, 2);

        // Determine behaviour/action
        if (maxQIndex == 0)
            this.transform.Rotate(Vector3.right, _tiltSpeed * (float)qs[maxQIndex]);
        else if (maxQIndex == 1)
            this.transform.Rotate(Vector3.right, -_tiltSpeed * (float)qs[maxQIndex]);

        // Determine reward
        if (ball.GetComponent<BallState>().dropped)
            _reward = -1.0f;
        else
            _reward = 0.1f;

        // Save into a replay
        Replay lastMemory = new Replay(this.transform.rotation.x, ball.transform.position.z,
                                       ball.GetComponent<Rigidbody>().angularVelocity.x, _reward);

        // Dequeue if reached memory capacity and then add
        if (_replayMemory.Count > _mCapacity)
            _replayMemory.RemoveAt(0);

        _replayMemory.Add(lastMemory);

        // Train the network every time the ball is dropped.
        if(ball.GetComponent<BallState>().dropped)
        {
            // Replay all memories backwards and train with each memory
            for(int i = _replayMemory.Count - 1; i > 0; i--)
            {
                // Initialise lists and extract outputs for current memory
                List<double> toutputsOld = new List<double>(); // Q values with current memory
                List<double> toutputsNew = new List<double>(); // Q values with next memory (later in time)
                toutputsOld = SoftMax(ann.CalculateOutput(_replayMemory[i].states)); // Compute Q values with current memory

                // Get Q data
                double maxQOld = toutputsOld.Max(); // The "stronger" output
                int action = toutputsOld.ToList().IndexOf(maxQOld); // Tilt direction (best action)

                // Determine feedback
                double feedback;
                if (i == _replayMemory.Count - 1 || _replayMemory[i].reward == -1) // End of memory or last action that cause fail
                    feedback = _replayMemory[i].reward;
                else
                {
                    toutputsNew = SoftMax(ann.CalculateOutput(_replayMemory[i + 1].states)); // Get the action from the next memory
                    maxQ = toutputsNew.Max();
                    feedback = _replayMemory[i].reward + _discount * maxQ; // Compute feedback (bellman equation)
                }

                toutputsOld[action] = feedback; // Set the reward/feedback only for the relevant action
                ann.Train(_replayMemory[i].states, toutputsOld); // Re-train network
            }

            // Update best time
            if (_timer > _maxBalanceTime)
                _maxBalanceTime = _timer;

            _timer = 0.0f;

            // Reset world
            ball.GetComponent<BallState>().dropped = false;
            this.transform.rotation = Quaternion.identity;
            ResetBall();
            _replayMemory.Clear();
            _failCount++;
        }
    }

    /// <summary>
    /// Normalises values on an exponential scale:
    /// Sum of values = 1.
    /// Values 0-1.
    /// </summary>
    /// <param name="values">The values to soften.</param>
    /// <returns>The softened values.</returns>
    private List<double> SoftMax(List<double> values)
    {
        double max = values.Max();

        float scale = 0.0f;
        for (int i = 0; i < values.Count; i++)
            scale += Mathf.Exp((float)(values[i] - max));

        List<double> result = new List<double>();
        for (int i = 0; i < values.Count; i++)
            result.Add(Mathf.Exp((float)(values[i] - max)) / scale);

        return result;
    }
}
