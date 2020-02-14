using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Eyes : MonoBehaviour, IExternalSensor
{
    // Looks up, down, front, and tells whether it's a top or bottom obstacle.
    private List<double> outputs = new List<double>(4);
    private float visibleDistance;

    public float VisibleDistance { get => visibleDistance; set => visibleDistance = value; }

    void Start()
    {
        outputs.Add(0);
        outputs.Add(0);
        outputs.Add(0);
        outputs.Add(0);
        outputs.Add(0);
        outputs.Add(0);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        // Debug eye view
        Debug.DrawRay(transform.position, transform.right * visibleDistance, Color.red);
        Debug.DrawRay(transform.position, new Vector2(0.7f, 0.3f) * visibleDistance, Color.red);
        Debug.DrawRay(transform.position, new Vector2(0.7f, -0.3f) * visibleDistance, Color.red);

        // Top facing raycast
        RaycastHit2D hit = Physics2D.Raycast(transform.position, new Vector2(0.7f, 0.3f), visibleDistance, 1 << LayerMask.NameToLayer("Limits"));
        //Debug.Log(hit.transform.gameObject.tag);
        if (hit.collider != null)
        {
            if (hit.collider.gameObject.tag == "upwall")
            {
                outputs[0] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[1] = 1;
            }
            else if (hit.collider.gameObject.tag == "bottomwall")
            {
                outputs[0] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[1] = -1;
            }
            else if (hit.collider.gameObject.tag == "point")
            {
                outputs[0] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[1] = 0;
            }
            else if (hit.collider.gameObject.tag == "top")
            {
                outputs[0] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[1] = 2;
            }
            else if (hit.collider.gameObject.tag == "bottom")
            {
                outputs[0] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[1] = -2;
            }

        }
        else
        {
            outputs[0] = 1;
            outputs[1] = 0;
        }
        // Bottom facing raycast
        hit = Physics2D.Raycast(transform.position, new Vector2(0.7f, -0.3f), visibleDistance, 1 << LayerMask.NameToLayer("Limits"));
        //Debug.Log(hit.transform.gameObject.tag);
        if (hit.collider != null)
        {
            if (hit.collider.gameObject.tag == "upwall")
            {
                outputs[2] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[3] = 1;
            }
            else if (hit.collider.gameObject.tag == "bottomwall")
            {
                outputs[2] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[3] = -1;
            }
            else if (hit.collider.gameObject.tag == "point")
            {
                outputs[2] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[3] = 0;
            }
            else if (hit.collider.gameObject.tag == "top")
            {
                outputs[2] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[3] = 2;
            }
            else if (hit.collider.gameObject.tag == "bottom")
            {
                outputs[2] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[3] = -2;
            }
        }
        else
        {
            outputs[2] = 1;
            outputs[3] = 0;
        }
        // Front facing raycast
        hit = Physics2D.Raycast(transform.position, transform.right, visibleDistance);
        if (hit.collider != null)
        {
            if (hit.collider.gameObject.tag == "upwall")
            {
                outputs[4] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[5] = -1;
            }
            else if (hit.collider.gameObject.tag == "bottomwall")
            {
                outputs[4] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[5] = 1;
            }
            else if (hit.collider.gameObject.tag == "point")
            {
                outputs[4] = (hit.transform.position.x - transform.position.x) / visibleDistance;
                outputs[5] = 0;
            }
        }
        else
        {
            outputs[4] = 1;
            outputs[5] = 0;
        }

    }

    public double[] GetSensorData()
    {
        return outputs.ToArray();
    }
}
