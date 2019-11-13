using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class PopulationManager : MonoBehaviour
{
    public List<GameObject> birdPrefabs;
    public int populationSize = 50;
    List<GameObject> population = new List<GameObject>();
    public static float elapsed = 0;
    public float trialTime = 5;
    int generation = 1;

    GUIStyle guiStyle = new GUIStyle();
    private void OnGUI()
    {
        guiStyle.fontSize = 20;
        guiStyle.normal.textColor = Color.white;
        GUI.BeginGroup(new Rect(10, 10, 250, 150));
        GUI.Box(new Rect(0, 0, 140, 140), "Stats", guiStyle);
        GUI.Label(new Rect(10, 25, 200, 30), "Gen: " + generation, guiStyle);
        GUI.Label(new Rect(10, 50, 200, 30), string.Format("Time: {0:0.00}", elapsed), guiStyle);
        GUI.Label(new Rect(10, 75, 200, 30), "Population: " + population.Count, guiStyle);
        GUI.EndGroup();
    }

    private void Start()
    {
        Time.timeScale = 5;
        // Create initial population
        for (int i = 0; i < populationSize; i++)
        {
            Vector3 startingPos = new Vector3(transform.position.x + Random.Range(-0.5f, 0.5f), transform.position.y,
                                              transform.position.z + Random.Range(-0.5f, 0.5f));

            GameObject b = Instantiate(birdPrefabs[Random.Range(0, birdPrefabs.Count)], startingPos, transform.rotation);
            b.GetComponent<Brain>().Init();
            population.Add(b);

        }
    }

    private GameObject Breed(GameObject mom, GameObject dad)
    {
        Vector3 startingPos = new Vector3(transform.position.x + Random.Range(-0.5f, 0.5f), transform.position.y,
                                          transform.position.z + Random.Range(-0.5f, 0.5f));
        
        // Get colour from parent
        string birdType = Random.Range(0, 10) < 5 ? mom.name : dad.name;
        GameObject birdPrefab = Resources.Load<GameObject>(birdType.Remove(birdType.Length - 7));
        GameObject child = Instantiate(birdPrefab, startingPos, transform.rotation);
        Brain b = child.GetComponent<Brain>();
        if(Random.Range(0,100) == 1) // Mutate 1/100
        {
            b.Init();
            b.dna.Mutate();
        }
        else
        {
            b.Init();
            b.dna.Combine(mom.GetComponent<Brain>().dna, dad.GetComponent<Brain>().dna);
        }
        return child;
    }

    private void BreedNewPopulation()
    {
        // Order the population by fittest.
        List<GameObject> sortedPopulation = population.OrderBy(o => (o.GetComponent<Brain>().distanceTravelled - o.GetComponent<Brain>().crash)).ToList<GameObject>();
        // Breed the top half
        population.Clear();
        // Breed the top half of the population
        for (int i = (int)(3*sortedPopulation.Count / 4.0f) - 1; i < sortedPopulation.Count - 1; i++)
        {
            population.Add(Breed(sortedPopulation[i], sortedPopulation[i + 1]));
            population.Add(Breed(sortedPopulation[i + 1], sortedPopulation[i]));
            population.Add(Breed(sortedPopulation[i], sortedPopulation[i + 1]));
            population.Add(Breed(sortedPopulation[i + 1], sortedPopulation[i]));
        }
        // Destroy old generation and move forward
        for (int i = 0; i < sortedPopulation.Count; i++)
        {
            Destroy(sortedPopulation[i]);
        }
        generation++;
    }

    private void Update()
    {
        elapsed += Time.deltaTime;
        if (elapsed >= trialTime)
        {
            BreedNewPopulation();
            elapsed = 0;
        }
    }
}
