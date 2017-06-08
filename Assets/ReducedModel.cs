using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.Text;
using System.IO;
using UnityEngine.Rendering;
using System.Runtime.InteropServices;
using System;

public class ReducedModel : MonoBehaviour {  
    public float m_frameSpeed = 5.0f;

    private int m_pointsCount = 61440;
    private Renderer m_pointRenderer;

    private int m_textureSideSize;
    private int m_textureSize;
    private int m_lookupTextureSize = 256; //Since the values in the value texture are only 0-255, it doesn't make sense to have more values here.

    private int m_textureSwitchFrameNumber = -1;

    private ComputeBuffer positionComputebuffer, velocityComputebuffer, nodesComputeBuffer, positionRecordComputebuffer, errorComputeBuffer;
    private int m_dimensionWidth, m_dimensionHeight, m_dimensionDepth;

    private float m_updateFrequency = 0.0333f;
    private float m_currentTime;
    private float m_lastFrameTime = 0.0f;
    private int m_iteration = 0;
    private int m_maxIterations = 100;

    public ComputeShader m_computeShader;
    public ComputeShader m_computeShaderSum;
    private int m_kernel, m_kernelSum;

    struct Node {
        Vector3 pos;
        float force;
        public Node(Vector3 pos, float force) {
            this.pos = pos;
            this.force = force;
        }
    };

    void initializeParticles() {
        m_dimensionWidth = 128;
        m_dimensionHeight = 1;
        m_dimensionDepth = 1;
        Vector3[] points = new Vector3[m_dimensionWidth * m_dimensionHeight * m_dimensionDepth];
        Vector3 startingPosition = new Vector3(20.0f, 0.0f, 0.0f);
        
        float deltaPos = 0.05f;

        for (int i = 0; i < m_dimensionWidth; i++) {
            for (int j = 0; j < m_dimensionHeight; j++) {
                for (int k = 0; k < m_dimensionDepth; k++) {
                    int index = i + m_dimensionWidth * (j + m_dimensionDepth * k);
                    //int index = i + j * m_dimensionWidth;
                    points[index] = new Vector3();
                    float randomVal = UnityEngine.Random.Range(0.0f, 1.0f);
                    points[index].x = i * deltaPos - m_dimensionWidth*deltaPos + m_dimensionWidth*deltaPos*0.5f + randomVal*1.0f + startingPosition.x; // + randomVal - 0.5f + startingPosition.x;
                    randomVal = UnityEngine.Random.Range(0.0f, 1.0f);
                    points[index].y = j * deltaPos - m_dimensionHeight*deltaPos + m_dimensionHeight*deltaPos*0.5f + randomVal*1.0f + startingPosition.y; //- dimensionHeight*deltaPos + dimensionWidth*deltaPos*0.5f + randomVal - 0.5f + startingPosition.y;
                    //points[index].z = k * deltaPos - m_dimensionDepth*deltaPos + m_dimensionDepth*deltaPos*0.5f/* + randomVal*10.0f*/; //- dimensionHeight*deltaPos + dimensionWidth*deltaPos*0.5f + randomVal - 0.5f + startingPosition.y;
                    
                }
            }
        }

        Vector3[] velocities = new Vector3[points.Length];
        for (int i = 0; i < m_dimensionWidth; i++) {
            for (int j = 0; j < m_dimensionHeight; j++) {
                for (int k = 0; k < m_dimensionDepth; k++) {
                    int index = i + m_dimensionWidth * (j + m_dimensionDepth * k);//i + j * m_dimensionWidth;
                    velocities[index] = new Vector3();
                }
            }
        }

        m_pointsCount = points.Length;

        positionComputebuffer = new ComputeBuffer (m_pointsCount, Marshal.SizeOf(typeof(Vector3)), ComputeBufferType.Default);
        positionComputebuffer.SetData(points);
        m_pointRenderer.material.SetBuffer ("_Positions", positionComputebuffer);
        m_computeShader.SetBuffer(m_kernel, "_Positions", positionComputebuffer);

        velocityComputebuffer = new ComputeBuffer (m_pointsCount, Marshal.SizeOf(typeof(Vector3)), ComputeBufferType.Default);
        velocityComputebuffer.SetData(velocities);
        m_pointRenderer.material.SetBuffer ("_Velocities", velocityComputebuffer);       
        m_computeShader.SetBuffer(m_kernel, "_Velocities", velocityComputebuffer);
    }

    void Start () {
        Screen.SetResolution(720, 1280, true);
        m_pointRenderer = GetComponent<Renderer>();
        m_kernel = m_computeShader.FindKernel("CSMain");
        m_kernelSum = m_computeShaderSum.FindKernel("reduce1");

        //nodes for training:
         Node[] trainingNodes = {
            new Node(new Vector3(0.0f, 0.0f, 0.0f), -0.005f),
            new Node(new Vector3(2.0f, 3.0f, 0.0f), -0.005f),
            new Node(new Vector3(-5.0f, 2.0f, 0.0f), 0.02f)
        };

        initializeParticles();

        nodesComputeBuffer = new ComputeBuffer(trainingNodes.Length, Marshal.SizeOf(typeof(Node)), ComputeBufferType.Default);
        nodesComputeBuffer.SetData(trainingNodes);

        positionRecordComputebuffer = new ComputeBuffer (m_pointsCount*m_maxIterations, Marshal.SizeOf(typeof(Vector3)), ComputeBufferType.Default);

        errorComputeBuffer = new ComputeBuffer(m_pointsCount*m_maxIterations, Marshal.SizeOf(typeof(float)), ComputeBufferType.Default);
        float[] errorData = new float[m_pointsCount*m_maxIterations];
        for (int i = 0; i < errorData.Length; i++) {
            errorData[i] = 1.0f;
        }
        errorComputeBuffer.SetData(errorData);

        m_computeShader.SetBuffer(m_kernel, "_Nodes", nodesComputeBuffer);
        m_computeShader.SetBuffer(m_kernel, "_PositionsRecord", positionRecordComputebuffer);
        m_computeShader.SetBuffer(m_kernel, "_Error", errorComputeBuffer);

        m_computeShaderSum.SetBuffer(m_kernelSum, "g_data", errorComputeBuffer);

        //m_pointRenderer.material.SetBuffer ("_Error", errorComputeBuffer);

        m_pointRenderer.material.SetInt("_PointsCount", m_pointsCount);
        float aspect = Camera.main.GetComponent<Camera>().aspect;
        m_pointRenderer.material.SetFloat("aspect", aspect);
        Vector4 trans = transform.position;
        m_pointRenderer.material.SetVector("trans", trans);
        m_pointRenderer.material.SetInt("_TextureSwitchFrameNumber", m_textureSwitchFrameNumber);   

        Debug.Log("Number of points: " + m_pointsCount);

        m_computeShader.SetInt("_particleCount", m_pointsCount);

        m_computeShader.SetBool("_recording", true);

        //Record simulation:
        for (int i = 0; i < m_maxIterations; i++) {
            m_computeShader.SetInt("_iteration", i);
            Dispatch();
        }

        m_computeShader.SetBool("_recording", false);

        m_iteration = 0;
        positionComputebuffer.Release();
        velocityComputebuffer.Release();
        initializeParticles();

        //Run validation:
        Debug.Log("Running validation:");
    

        for (int i = 0; i < m_maxIterations; i++) {
            m_computeShader.SetInt("_iteration", i);
            Dispatch();
        }

        errorComputeBuffer.GetData(errorData);

        Debug.Log("Error: " + errorData[0]);

        float sum = 0f;
        for (int i = 0; i < errorData.Length; i++) {
            float error = errorData[i];
            //if (error > 0.001f) {
            //    Debug.Log("Error: " + error);
            //}
            sum += error;
        }
        Debug.Log("Error Sum: " + sum);

        m_computeShaderSum.Dispatch(m_kernelSum, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
        m_computeShaderSum.Dispatch(m_kernelSum, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
        //m_computeShaderSum.Dispatch(m_kernelSum, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);

        errorComputeBuffer.GetData(errorData);

        Debug.Log("Error: " + errorData[0]);

    }
	
	// Update is called once per frame
	void Update () {
        float aspect = Camera.main.GetComponent<Camera>().aspect;
        m_pointRenderer.material.SetFloat("aspect", aspect);

        //Debug.Log("Support instancing: " + SystemInfo.supportsInstancing);

        //int a = pointRenderer.material.GetInt("_FrameTime");

        //Debug.Log(a);
        //m_currentTime += Time.deltaTime;
        //if (m_currentTime >= m_updateFrequency)
        {
            //m_currentTime = 0.0f;
            //m_computeShader.SetInt("_iteration", m_iteration);
           

            //m_lastFrameTime = Time.fixedTime;
        }

        
        m_iteration = m_iteration % m_maxIterations;
    }

    private void OnRenderObject()
    {
        m_computeShader.SetInt("_iteration", m_iteration);
        Dispatch();

        //m_currentTime += Time.deltaTime;
        //if (m_currentTime >= m_updateFrequency)
        {
            m_currentTime = 0.0f;
            m_pointRenderer.material.SetPass(0);
            m_pointRenderer.material.SetMatrix("model", transform.localToWorldMatrix);
            m_pointRenderer.material.SetInt("_iteration", m_iteration);
            //Graphics.DrawProcedural(MeshTopology.Points, 1, m_pointsCount);
            Graphics.DrawProcedural(MeshTopology.Points, m_pointsCount);  // index buffer.
            m_iteration++;
        }
    }

    /*void OnPostRender ()
    {
        Dispatch();
    }*/

    void OnDestroy() {
        positionComputebuffer.Release();
        velocityComputebuffer.Release();
        nodesComputeBuffer.Release();
        positionRecordComputebuffer.Release();
        errorComputeBuffer.Release();
    }
    private void Dispatch()
    {
        float deltaTime = Time.fixedTime - m_lastFrameTime;

        //Debug.Log("Delta time: " + deltaTime);

        m_computeShader.SetFloat("deltaTime", deltaTime*1.0f);

        m_computeShader.Dispatch(m_kernel, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
    }
}
