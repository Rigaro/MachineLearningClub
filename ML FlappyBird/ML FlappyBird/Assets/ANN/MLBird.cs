using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

public class MLBird : MonoBehaviour 
{
    [Header("Bird configuration")]
    [SerializeField]
    private float upForce = 200;					//Upward force of the "flap".
    [SerializeField]
    private float visibleDistance = 200.0f;
    [SerializeField]
    private float flapDelay = 0.5f;
    private bool isDead = false;            //Has the player collided with a wall?

    private Animator anim;                  //Reference to the Animator component.
    private Rigidbody2D rb2d;				//Holds a reference to the Rigidbody2D component of the bird.

    // Unity stuff
    [Header("Brain configuration")]
    [SerializeField]
    private List<GameObject> _externalSensors;  // The external sensors for this brain
    [SerializeField]
    private bool _debug = true;
    [SerializeField]
    private bool _training = true;

    [Header("ANN configuration")]
    // Neural Network
    private ANN _ann;
    [SerializeField]
    private int _numInputs;
    [SerializeField]
    private int _numOutputs;
    [SerializeField]
    private int _numHidden;
    [SerializeField]
    private int _numNPerHidden;
    [SerializeField]
    private double _alpha = 0.2f;
    [SerializeField]
    private ANN.ActivationFunctionTypes hiddenActFunType = ANN.ActivationFunctionTypes.Sigmoid;
    [SerializeField]
    private ANN.ActivationFunctionTypes outputActFunType = ANN.ActivationFunctionTypes.Sigmoid;
    [SerializeField]
    private bool loadWeightsFromFile = false;

    [Header("Reinforcement learning configuration")]
    // Memory variables used for reinforcement learning
    private float _reward = 0.0f;                                    // Reward to associate to actions
    private List<Replay> _replayMemory = new List<Replay>();         // Memory - list of past actions and rewards.
    [SerializeField]
    private int _mCapacity = 10000;                                  // Memory capacity

    // Reinforcement learning configuration
    [SerializeField]
    private float _discount = 0.99f;                                 // How much future states affect rewards
    [SerializeField]
    private float _exploreRate = 100.0f;                             // Chance of picking a random action
    [SerializeField]
    private float _maxExploreRate = 100.0f;                          // Max chance value
    [SerializeField]
    private float _minExploreRate = 0.01f;                           // Min chance value
    [SerializeField]
    private float _exploreDecay = 0.0001f;                           // Chance decay amount per iteration/update

    // Help variables
    private float _timer = 0.0f;                                   // Timer to keep track of running time
    private float _maxRunTime = 0.0f;                              // The best time.
    private float _distanceTravelled = 0.0f;
    private bool inTraining = false;

    /// <summary>
    /// Problem specific configurations, e.g. States, object initial condition.
    /// </summary>
    private int _failCount = 0;                                      // The number of failed attempts

    /// <summary>
    /// Behavioural configurations that affect the outputs
    /// </summary>
    private Vector3 _startPosition;
    private enum ISee { bottom, gap, top};
    private ISee see = ISee.bottom;
    private bool isFlapping = false;
    private bool scored = false;

    void Start()
	{
		//Get reference to the Animator component attached to this GameObject.
		anim = GetComponent<Animator> ();
		//Get and store a reference to the Rigidbody2D attached to this GameObject.
		rb2d = GetComponent<Rigidbody2D>();
        // Set start position
        _startPosition = transform.position;

        // Create the ANN, get ball position, and speed up simulation.
        _ann = new ANN(_numInputs, _numOutputs, _numHidden, _numNPerHidden, _alpha, hiddenActFunType, outputActFunType);
        if (loadWeightsFromFile)
            _ann.LoadWeightsFromFile();

        // Write any specific initialisations here.
        // Find the eyes to set the visible distance.
        foreach(GameObject sensor in _externalSensors)
        {
            Eyes eyes = sensor.GetComponent<Eyes>();
            if (eyes != null)
                eyes.VisibleDistance = visibleDistance;
        }
        //

        // Speed up simulation when training
        if (_training)
            Time.timeScale = 15.0f;
    }

    // Debug GUI
    private GUIStyle _guiStyle = new GUIStyle();
    private void OnGUI()
    {
        if (_debug)
        {
            _guiStyle.fontSize = 25;
            _guiStyle.normal.textColor = Color.white;
            GUI.BeginGroup(new Rect(10, 10, 700, 150));
            GUI.Box(new Rect(0, 0, 140, 140), "Stats", _guiStyle);
            GUI.Label(new Rect(10, 25, 500, 30), "Fails: " + _failCount, _guiStyle);
            GUI.Label(new Rect(10, 50, 500, 30), "Explore rate: " + _exploreRate, _guiStyle);
            GUI.Label(new Rect(10, 75, 500, 30), "Last best time: " + _maxRunTime, _guiStyle);
            GUI.Label(new Rect(10, 100, 500, 30), "Run time: " + _timer, _guiStyle);
            GUI.Label(new Rect(10, 125, 500, 30), "I see: " + see.ToString(), _guiStyle);
            GUI.EndGroup();
        }
    }

    void Update()
	{
        // Force states to reset by pressing space key.
        if (Input.GetKeyDown(KeyCode.Space))
            ResetStates();
        // Save weights to a file by pressing S
        if (Input.GetKeyDown(KeyCode.S))
            _ann.SaveWeightsToFile();
        
        if (isDead) return;
    }

	void OnCollisionEnter2D(Collision2D other)
	{
		// Zero out the bird's velocity
		rb2d.velocity = Vector2.zero;
		// If the bird collides with something set it to dead...
		isDead = true;
        _distanceTravelled = 0.0f;
		//...tell the Animator about it...
		anim.SetTrigger ("Die");
		//...and tell the game control about it.
		GameControl.instance.BirdDied ();
	}

    /// <summary>
    /// Run deterministically
    /// </summary>
    private void FixedUpdate()
    {
        if (inTraining) return;
            // Initialise local variables and update run time
            List<double> states = new List<double>(); // The system states that will be the input to the ANN.
            List<double> qValues = new List<double>(); // Quality values for Q-Learning
            _timer += Time.deltaTime;
            _distanceTravelled = Vector3.Distance(_startPosition, transform.position);

            // Update states (input to ANN)
            // First update the internal states (for the object the brain is attached to)
            Vector2 vel = rb2d.velocity;
            states.Add(vel.y);
            // Now get data from the external sensors.
            foreach (GameObject sensorGO in _externalSensors)
            {
                IExternalSensor sensorManager = sensorGO.GetComponent<IExternalSensor>();
                states.AddRange(sensorManager.GetSensorData());
            }
            //Debug.Log(states[0] + ", " + states[1] + ", " + states[2] + ", " + states[3] + ", " + states[4]);
            see = (ISee)((int)states[6] + 1);

            // Q data
            qValues = SoftMax(_ann.CalculateOutput(states)); // use SoftMax to format the ANN output
            double maxQ = qValues.Max(); // Find maximum Q out of states
            int maxQIndex = qValues.ToList().IndexOf(maxQ); // Get the index of the maximum value
            _exploreRate = Mathf.Clamp(_exploreRate - _exploreDecay, _minExploreRate, _maxExploreRate); // Update the explore rate

            // Perform some actions according to the system behaviour
            UpdateActions(qValues, maxQIndex);

            // Update reward
            _reward = UpdateReward();

            // Update memory
            UpdateMemory(states);

            // Train the network when the training condition is met.
            if (_training && TrainingCondition() && !inTraining)
            {
                if(_timer > _maxRunTime)
                    _ann.SaveWeightsToFile();
                inTraining = true;
                QLearning(maxQ);

                // Reset world and system
                _replayMemory.Clear();
                _failCount++;
            }
    }

    /// <summary>
    /// Updates the system's behaviour by performing some action
    /// </summary>
    /// <param name="qs"></param>
    /// <param name="maxQIndex"></param>
    private void UpdateActions(List<double> qs, int maxQIndex)
    {
        // Choose whether to randomly explore
        if (Random.Range(0, 100) < _exploreRate && !_training)
            maxQIndex = Random.Range(0, _numOutputs); // Pick from a random action (dependent on number of outputs)

        //Debug.Log(maxQIndex + ", " + (float)qs[maxQIndex]);
        // Perform some action
        //Don't allow control if the bird has died.
        if (isDead == false)
        {
            // Choose to flap
            if (maxQIndex == 0 && !isFlapping)
            {
                //Debug.Log("Flap!");
                isFlapping = true;
                //...tell the animator about it and then...
                anim.SetTrigger("Flap");
                //...zero out the birds current y velocity before...
                rb2d.velocity = Vector2.zero;
                //	new Vector2(rb2d.velocity.x, 0);
                //..giving the bird some upward force.
                rb2d.AddForce(new Vector2(0, upForce));
                //rb2d.AddForce(new Vector2(0, upForce * (float)qs[maxQIndex]));
                StartCoroutine(ResetFlap());
            }
            // Choose to flap
            if (maxQIndex == 1)
            {
                //...tell the animator about it and then...
                //anim.SetTrigger("Flap");
                //...zero out the birds current y velocity before...
                //rb2d.velocity = Vector2.zero;
                //	new Vector2(rb2d.velocity.x, 0);
                //..giving the bird some upward force.
            }
        }
    }

    private IEnumerator ResetFlap()
    {
        yield return new WaitForSecondsRealtime(flapDelay);
        isFlapping = false;
    }

    /// <summary>
    /// Updates the reward to the system
    /// </summary>
    private float UpdateReward()
    {
        // Some reward function
        if (isDead)
            return -1.0f;
        else if (scored)
        {
            scored = false;
            return 0.1f;
        }
        else
            return 0.001f;
            //return 0.01f*_timer;
    }

    public void Scored()
    {
        scored = true;
    }

    private void UpdateMemory(List<double> states)
    {
        // Save into a replay
        Replay lastMemory = new Replay(states, _reward);

        // Dequeue if reached memory capacity and then add
        if (_replayMemory.Count > _mCapacity)
            _replayMemory.RemoveAt(0);

        _replayMemory.Add(lastMemory);
    }

    /// <summary>
    /// Checks that the brain training condition is met
    /// </summary>
    /// <returns></returns>
    private bool TrainingCondition()
    {
        return isDead;
    }

    /// <summary>
    /// Trains the brain using Q-learning.
    /// https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56
    /// </summary>
    /// <param name="maxQ"></param>
    private void QLearning(double maxQ)
    {
        // Replay all memories backwards and train with each memory
        for (int i = _replayMemory.Count - 1; i > 0; i--)
        {
            // Initialise lists and extract outputs for current memory
            List<double> toutputsOld = new List<double>(); // Q values with current memory
            List<double> toutputsNew = new List<double>(); // Q values with next memory (later in time)
            toutputsOld = SoftMax(_ann.CalculateOutput(_replayMemory[i].states)); // Compute Q values with current memory

            // Get Q data
            double maxQOld = toutputsOld.Max(); // The "stronger" output
            int action = toutputsOld.ToList().IndexOf(maxQOld); // The best action

            // Determine feedback
            double feedback;
            if (i == _replayMemory.Count - 1 || _replayMemory[i].reward == -1) // End of memory or last action that cause fail
                feedback = _replayMemory[i].reward;
            else
            {
                toutputsNew = SoftMax(_ann.CalculateOutput(_replayMemory[i + 1].states)); // Get the action from the next memory
                maxQ = toutputsNew.Max();
                feedback = _replayMemory[i].reward + _discount * maxQ; // Compute feedback (bellman equation)
            }

            toutputsOld[action] = feedback; // Set the reward/feedback only for the relevant action
            _ann.Train(_replayMemory[i].states, toutputsOld); // Re-train network
        }

        // Update best time
        if (_timer > _maxRunTime)
            _maxRunTime = _timer;

        _timer = 0.0f;
    }

    /// <summary>
    /// Resets the brain states
    /// </summary>
    private void ResetStates()
    {
        // Some resetting stuff
        transform.position = _startPosition;
        _reward = -1;
        isDead = false;
        inTraining = false;
        anim.SetTrigger("Reset");
    }

    public void ResetBird()
    {
        ResetStates();
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
