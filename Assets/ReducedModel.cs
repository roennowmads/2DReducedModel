﻿using UnityEngine;
using System.Runtime.InteropServices;

public class ReducedModel : MonoBehaviour {  
    public float m_frameSpeed = 5.0f;

    private int m_pointsCount = 61440;
    private Renderer m_pointRenderer;

    private int m_textureSideSize;
    private int m_textureSize;
    private int m_lookupTextureSize = 256; //Since the values in the value texture are only 0-255, it doesn't make sense to have more values here.

    private int m_textureSwitchFrameNumber = -1;

    private ComputeBuffer nodesComputeBuffer, recordedDataComputebuffer, sumDataComputeBuffer, deltaNodesComputeBuffer, bestNodesComputeBuffer;
    private ComputeBuffer[] particleInOutBuffers;

    private int m_dimensionWidth, m_dimensionHeight, m_dimensionDepth, m_threadGroupsX, m_threadGroupsY;

    private float m_updateFrequency = 0.0333f;
    private float m_currentTime;
    private float m_lastFrameTime = 0.0f;
    private int m_iteration = 0;
    private int m_iter = 0;
    private int m_maxIterations = 2000;//172 * 5;//1000;

    public ComputeShader m_computeShader;
    private int m_kernelValidate, m_kernelRecord, m_kernelRun, m_kernelRunField, m_kernelRecordField, m_kernelNodeError;

    private Particle[] m_particles;
    private float[] m_errorSingleData = new float[1];

    private int m_validationStepSize = 50;

    private int bufferSwitch = 0;

    public struct Node {
        public Vector3 pos;
        public float force;
        public Node(Vector3 pos, float force) {
            this.pos = pos;
            this.force = force;
        }
    };

    public struct Particle {
        public Vector3 position;
        public Vector3 velocity;
    };

    void incrementBufferSwitch() {
        bufferSwitch = (bufferSwitch + 1) % 2;
    }

    void loadPreGenData()
    {
        Texture2D tex = new Texture2D(64, 64, TextureFormat.RGFloat, false, false);
        tex.filterMode = FilterMode.Point;
        tex.wrapMode = TextureWrapMode.Repeat;
        tex.anisoLevel = 1;

        Texture2DArray texture = new Texture2DArray(64, 64, 256, TextureFormat.RGFloat, false);
        texture.filterMode = FilterMode.Point;
        texture.wrapMode = TextureWrapMode.Repeat;
        texture.anisoLevel = 1;

        //Texture3D texture = new Texture3D(64, 64, 256, TextureFormat.RGFloat, false);
        //texture.filterMode = FilterMode.Point;
        //texture.wrapMode = TextureWrapMode.Repeat;
        //texture.anisoLevel = 1;
        //Texture3D texture = new Texture3D(64, 64, 128, TextureFormat.RGFloat, false);

        for (int i = 0; i < 172; i++) {
            string index = "" + (i + 229);
            TextAsset ta = Resources.Load("xy/00" + index) as TextAsset;
            byte[] bytes = ta.bytes;

            tex.LoadRawTextureData(bytes);
            tex.Apply();

            Graphics.CopyTexture(tex, 0, 0, texture, i, 0);
        }

        //Renderer pointRenderer = GetComponent<Renderer>();
        //pointRenderer.material.mainTexture = texture;
        m_computeShader.SetTexture(m_kernelRunField, "_VelTexture", texture);
        m_computeShader.SetTexture(m_kernelRecordField, "_VelTexture", texture);

        //return particles;
    }

    void initializeParticles() {
        //loadPreGenData();

        m_dimensionWidth = 16;
        m_dimensionHeight = 16;
        int threadGroupSize = 16;
        m_threadGroupsX = m_dimensionWidth / threadGroupSize;
        m_threadGroupsY = m_dimensionHeight / threadGroupSize;
        m_dimensionDepth = 1;

        m_particles = new Particle[m_dimensionWidth * m_dimensionHeight * m_dimensionDepth];
        //m_particlesRecorded = new Vector3[m_particles.Length * m_maxIterations / m_validationStepSize];

        Vector3 startingPosition = new Vector3(20.0f, 0.0f, 0.0f);
        
        float deltaPos = 1.0f;

        for (int i = 0; i < m_dimensionWidth; i++) {
            for (int j = 0; j < m_dimensionHeight; j++) {
                for (int k = 0; k < m_dimensionDepth; k++) {
                    int index = i + m_dimensionWidth * (j + m_dimensionDepth * k);
                    m_particles[index].position = new Vector3();
                    float randomVal = 0.0f;//UnityEngine.Random.Range(0.0f, 1.0f);
                    m_particles[index].position.x = i * deltaPos - m_dimensionWidth*deltaPos + m_dimensionWidth*deltaPos*0.5f + randomVal*1.0f + startingPosition.x; // + randomVal - 0.5f + startingPosition.x;
                    randomVal = 0.0f;//UnityEngine.Random.Range(0.0f, 1.0f);
                    m_particles[index].position.y = j * deltaPos - m_dimensionHeight*deltaPos + m_dimensionHeight*deltaPos*0.5f + randomVal*1.0f + startingPosition.y; //- dimensionHeight*deltaPos + dimensionWidth*deltaPos*0.5f + randomVal - 0.5f + startingPosition.y;
                    //points[index].z = k * deltaPos - m_dimensionDepth*deltaPos + m_dimensionDepth*deltaPos*0.5f/* + randomVal*10.0f*/; //- dimensionHeight*deltaPos + dimensionWidth*deltaPos*0.5f + randomVal - 0.5f + startingPosition.y;
                    m_particles[index].velocity = new Vector3();
                }
            }
        }

        m_pointsCount = m_particles.Length;

        m_computeShader.SetInt("_dimensionWidth", m_dimensionWidth);
        m_computeShader.SetInt("_maxIterations", m_maxIterations);
        m_computeShader.SetInt("_validationStepSize", m_validationStepSize);

        particleInOutBuffers = new ComputeBuffer[2];

        particleInOutBuffers[0] = new ComputeBuffer(m_pointsCount, Marshal.SizeOf(typeof(Particle)), ComputeBufferType.Default);
        particleInOutBuffers[0].SetData(m_particles);
        m_computeShader.SetBuffer(m_kernelValidate, "_ParticleDataIn", particleInOutBuffers[0]);
        m_computeShader.SetBuffer(m_kernelRecord, "_ParticleDataIn", particleInOutBuffers[0]);
        m_computeShader.SetBuffer(m_kernelRun, "_ParticleDataIn", particleInOutBuffers[0]);
        m_computeShader.SetBuffer(m_kernelRunField, "_ParticleDataIn", particleInOutBuffers[0]);
        m_computeShader.SetBuffer(m_kernelRecordField, "_ParticleDataIn", particleInOutBuffers[0]);
        //m_computeShader.SetBuffer(m_kernelNodeError, "_ParticleDataIn", particleInOutBuffers[0]);

        particleInOutBuffers[1] = new ComputeBuffer(m_pointsCount, Marshal.SizeOf(typeof(Particle)), ComputeBufferType.Default);
        particleInOutBuffers[1].SetData(m_particles);
        m_computeShader.SetBuffer(m_kernelRun, "_ParticleDataOut", particleInOutBuffers[1]);
        m_pointRenderer.material.SetBuffer("_ParticleData", particleInOutBuffers[1]);
        m_computeShader.SetBuffer(m_kernelRunField, "_ParticleDataOut", particleInOutBuffers[1]);

        recordedDataComputebuffer = new ComputeBuffer(m_pointsCount * m_maxIterations / m_validationStepSize, Marshal.SizeOf(typeof(Vector3)), ComputeBufferType.Default);
        //recordedDataComputebuffer.SetData(m_particlesRecorded);
        m_pointRenderer.material.SetBuffer("_RecordedData", recordedDataComputebuffer);
        m_computeShader.SetBuffer(m_kernelValidate, "_RecordedData", recordedDataComputebuffer);
        m_computeShader.SetBuffer(m_kernelRecord, "_RecordedData", recordedDataComputebuffer);
        m_computeShader.SetBuffer(m_kernelRecordField, "_RecordedData", recordedDataComputebuffer);
        //m_computeShader.SetBuffer(m_kernelNodeError, "_RecordedData", recordedDataComputebuffer);

        sumDataComputeBuffer = new ComputeBuffer(1, Marshal.SizeOf(typeof(float)), ComputeBufferType.Default);
        float[] initialError = { float.PositiveInfinity };
        sumDataComputeBuffer.SetData(initialError);
        m_computeShader.SetBuffer(m_kernelValidate, "_SumData", sumDataComputeBuffer);
        //m_computeShader.SetBuffer(m_kernelNodeError, "_SumData", sumDataComputeBuffer);
    }

    void resetParticles() {
        particleInOutBuffers[0].SetData(m_particles);
        particleInOutBuffers[1].SetData(m_particles);
    }

    Node[] getRandomDeltas(int numberOfNodes) {
        Node[] randomDeltas = new Node[numberOfNodes];

        for (int i = 0; i < randomDeltas.Length; i++) {
            if (UnityEngine.Random.Range(0.0f, 1.0f) > 0.5f) {
                randomDeltas[i].force = 1.0f;
	        } else {
                randomDeltas[i].force = -1.0f;
            }

            if (UnityEngine.Random.Range(0.0f, 1.0f) > 0.5f)
            {
                randomDeltas[i].pos.x = 1.0f;
            } else {
                randomDeltas[i].pos.x = -1.0f;
            }

            if (UnityEngine.Random.Range(0.0f, 1.0f) > 0.5f)
            {
                randomDeltas[i].pos.y = 1.0f;
            } else {
                randomDeltas[i].pos.y = -1.0f;
            }

            if (UnityEngine.Random.Range(0.0f, 1.0f) > 0.5f)
            {
                randomDeltas[i].pos.z = 1.0f;
            } else {
                randomDeltas[i].pos.z = -1.0f;
            }
        }
        return randomDeltas;
    }

    //https://github.com/yanatan16/golang-spsa/blob/master/spsa.go
    //https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
    void gpuEstimateGradient(Node[] nodes) {
        Node[] deltas = getRandomDeltas(nodes.Length);

        deltaNodesComputeBuffer.SetData(deltas);
        m_computeShader.Dispatch(m_kernelValidate, m_threadGroupsX, m_threadGroupsY, 1);
    }

    void gpuRecordSimulation(Node[] trainingNodes) {
        m_computeShader.Dispatch(m_kernelRecord, m_threadGroupsX, m_threadGroupsY, 1);
        //m_computeShader.Dispatch(m_kernelRecordField, m_threadGroupsX, m_threadGroupsY, 1);
        //No need to restore anything or switch buffers because the record kernel doesn't modify the particles buffer.
    }

    void gpuTrainValidateModel(Node[] testNodes) {
        float gradientScale = 0.000001f;
        float deltaScale = 0.0001f;
        float numberOfErrorVals = m_maxIterations * m_pointsCount / m_validationStepSize;

        //float bestError = float.PositiveInfinity;
        //int bestIteration = 0;
        //Node[] bestNodes = (Node[])testNodes.Clone();

        m_computeShader.SetFloat("_deltaScale", deltaScale);
        m_computeShader.SetFloat("_gradientScale", gradientScale);

        nodesComputeBuffer.SetData(testNodes);

        Node[] nodes = new Node[testNodes.Length];

        for (int j = 0; j < 10000; j++)
        {
            gpuEstimateGradient(testNodes);
            //sumDataComputeBuffer.GetData(m_errorSingleData);
            //float error = m_errorSingleData[0];
            //Debug.Log("Error: " + error);

            if (j % 1000 == 0)
            {
                //m_computeShader.Dispatch(m_kernelNodeError, m_threadGroupsX, m_threadGroupsY, 1);

                //Node[] bNodes = new Node[testNodes.Length];
                //bestNodesComputeBuffer.GetData(bNodes);
                sumDataComputeBuffer.GetData(m_errorSingleData);
                float bError = m_errorSingleData[0] / numberOfErrorVals;
                Debug.Log("Best error at: " + j + " " + bError);
            }

        }

        //m_computeShader.Dispatch(m_kernelNodeError, m_threadGroupsX, m_threadGroupsY, 1);

        Node[] bestNodes = new Node[testNodes.Length];
        bestNodesComputeBuffer.GetData(bestNodes);
        sumDataComputeBuffer.GetData(m_errorSingleData);


        float bestError = m_errorSingleData[0] / numberOfErrorVals;
        Debug.Log("Error: " + bestError);

        //Debug.Log("Best Error: " + bestIteration + " " + bestError);

        for (int i = 0; i < bestNodes.Length; i++)
        {
            Debug.Log(i + " " + bestNodes[i].pos + ", force: " + bestNodes[i].force);
        }

        nodesComputeBuffer.SetData(bestNodes);
    }

    void Start () {
        Screen.SetResolution(720, 1280, true);
        m_pointRenderer = GetComponent<Renderer>();
        m_kernelValidate = m_computeShader.FindKernel("validate");  
        m_kernelRecord = m_computeShader.FindKernel("record");
        m_kernelRun = m_computeShader.FindKernel("run");
        //m_kernelNodeError = m_computeShader.FindKernel("nodeError");        

        m_kernelRunField = m_computeShader.FindKernel("runField");
        m_kernelRecordField = m_computeShader.FindKernel("recordField");

        //nodes for training:`
        Node[] trainingNodes = {
            new Node(new Vector3(0.0f, 0.0f, 0.0f), -0.005f),
            new Node(new Vector3(2.0f, 3.0f, 0.0f), -0.005f),
            new Node(new Vector3(-5.0f, 2.0f, 0.0f), 0.02f)
        };

        Node[] testNodes = {
            new Node(new Vector3(-5.0f, -5.0f, 0.0f), 0.0f),
            new Node(new Vector3(5.0f, 5.0f, 0.0f), 0.0f),
            new Node(new Vector3(5.0f, -5.0f, 0.0f), 0.0f),

            new Node(new Vector3(-5.0f, 5.0f, 0.0f), 0.0f),
            new Node(new Vector3(0.0f, 0.0f, 0.0f), 0.0f),

            new Node(new Vector3(-2.0f, -2.0f, 0.0f), 0.0f),
            new Node(new Vector3(2.0f, 2.0f, 0.0f), 0.0f)
        };

        Node[] checkNodes = {
            new Node(new Vector3(-3.3f, 2.2f, 1.7f), 0.02121057f),
            new Node(new Vector3(-1.0f, 2.6f, 0.2f), -0.03288997f),
            new Node(new Vector3(-3.3f, 3.6f, -1.2f), 0.02232057f)
        };

        initializeParticles();

        nodesComputeBuffer = new ComputeBuffer(testNodes.Length, Marshal.SizeOf(typeof(Node)), ComputeBufferType.Default);
        nodesComputeBuffer.SetData(trainingNodes);
        m_computeShader.SetBuffer(m_kernelValidate, "_Nodes", nodesComputeBuffer);
        m_computeShader.SetBuffer(m_kernelRecord, "_Nodes", nodesComputeBuffer);
        m_computeShader.SetBuffer(m_kernelRun, "_Nodes", nodesComputeBuffer);
        //m_computeShader.SetBuffer(m_kernelNodeError, "_Nodes", nodesComputeBuffer);

        deltaNodesComputeBuffer = new ComputeBuffer(testNodes.Length, Marshal.SizeOf(typeof(Node)), ComputeBufferType.Default);
        m_computeShader.SetBuffer(m_kernelValidate, "_DeltaNodes", deltaNodesComputeBuffer);

        bestNodesComputeBuffer = new ComputeBuffer(testNodes.Length, Marshal.SizeOf(typeof(Node)), ComputeBufferType.Default);
        m_computeShader.SetBuffer(m_kernelValidate, "_BestNodes", bestNodesComputeBuffer);
        //m_computeShader.SetBuffer(m_kernelNodeError, "_BestNodes", bestNodesComputeBuffer);


        m_computeShader.SetInt("_particleCount", m_pointsCount);
        m_pointRenderer.material.SetInt("_PointsCount", m_pointsCount);
        float aspect = Camera.main.GetComponent<Camera>().aspect;
        m_pointRenderer.material.SetFloat("aspect", aspect);
        Vector4 trans = transform.position;
        m_pointRenderer.material.SetVector("trans", trans);
        m_pointRenderer.material.SetInt("_TextureSwitchFrameNumber", m_textureSwitchFrameNumber);   

        Debug.Log("Number of points: " + m_pointsCount);

        //CPUParticles.cpuRecordSimulation(trainingNodes, ref m_particles, ref m_particlesRecorded, m_pointsCount, m_maxIterations);
        //CPUParticles.cpuTrainValidateModel(testNodes, ref m_particles, ref m_particlesError, ref m_particlesRecorded, m_pointsCount, m_maxIterations);

        m_computeShader.SetInt("_numberOfNodes", trainingNodes.Length);
        m_computeShader.SetFloat("_numberOfNodesReciproc", 1.0f / trainingNodes.Length);
        gpuRecordSimulation(trainingNodes);

        m_computeShader.SetInt("_numberOfNodes", testNodes.Length);
        m_computeShader.SetFloat("_numberOfNodesReciproc", 1.0f / testNodes.Length);
        gpuTrainValidateModel(testNodes);

        //m_pointRenderer.material.SetBuffer("_ParticleData", particleInOutBuffers[bufferSwitch]);

        //nodesComputeBuffer.SetData(trainingNodes);

        //resetParticles();

        /*Vector3[] particlesRecorded = new Vector3[m_particles.Length * m_maxIterations / m_validationStepSize];
        recordedDataComputebuffer.GetData(particlesRecorded);

        Vector3 sumVec = new Vector3();
        for (int i = 0; i < particlesRecorded.Length; i++)
        {
            if (i % 528 == 0)
            {
                Debug.Log(sumVec);
            }

            sumVec += particlesRecorded[i];
        }*/

        //m_computeShader.SetInt("_iteration", m_iteration);
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
    }

    private void OnRenderObject()
    {
        if (m_iter < 2000)
        //if (m_iter < 175 * 5)
        {
            m_computeShader.SetInt("_iteration", m_iteration);
            m_computeShader.Dispatch(m_kernelRun, m_threadGroupsX, m_threadGroupsY, 1);
            //m_computeShader.Dispatch(m_kernelRunField, m_threadGroupsX, m_threadGroupsY, 1);

            incrementBufferSwitch();
            m_computeShader.SetBuffer(m_kernelRun, "_ParticleDataIn", particleInOutBuffers[bufferSwitch]);
            m_computeShader.SetBuffer(m_kernelRun, "_ParticleDataOut", particleInOutBuffers[1 - bufferSwitch]);

            //m_computeShader.SetBuffer(m_kernelRunField, "_ParticleDataIn", particleInOutBuffers[bufferSwitch]);
            //m_computeShader.SetBuffer(m_kernelRunField, "_ParticleDataOut", particleInOutBuffers[1 - bufferSwitch]);

        }

        m_pointRenderer.material.SetBuffer("_ParticleData", particleInOutBuffers[bufferSwitch]);


        m_currentTime += Time.deltaTime;
        //if (m_currentTime >= m_updateFrequency) {
            m_iteration = (m_iteration + 1);
            m_currentTime = 0.0f;
            //Debug.Log(m_iteration);
        //}
        //Debug.Log(m_iter);
        m_iter++;



        m_pointRenderer.material.SetPass(0);
        m_pointRenderer.material.SetMatrix("model", transform.localToWorldMatrix);
        //m_pointRenderer.material.SetInt("_iteration", m_iteration);
        //Graphics.DrawProcedural(MeshTopology.Points, 1, m_pointsCount);
        Graphics.DrawProcedural(MeshTopology.Points, m_pointsCount);  // index buffer.
            
    }

    /*void OnPostRender ()
    {
        Dispatch();
    }*/

    void OnDestroy() {
        particleInOutBuffers[0].Release();
        particleInOutBuffers[1].Release();
        //errorDataComputebuffer.Release();
        recordedDataComputebuffer.Release();
        nodesComputeBuffer.Release();
        sumDataComputeBuffer.Release();
    }
    /*private void Dispatch()
    {
        float deltaTime = Time.fixedTime - m_lastFrameTime;

        //Debug.Log("Delta time: " + deltaTime);

        m_computeShader.SetFloat("deltaTime", deltaTime*1.0f);

        m_computeShader.Dispatch(m_kernel, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
    }*/
}
