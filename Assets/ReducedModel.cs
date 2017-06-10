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

    private ComputeBuffer particleComputebuffer, nodesComputeBuffer, errorDataComputebuffer;
    private int m_dimensionWidth, m_dimensionHeight, m_dimensionDepth;

    private float m_updateFrequency = 0.0333f;
    private float m_currentTime;
    private float m_lastFrameTime = 0.0f;
    private int m_iteration = 0;
    private int m_maxIterations = 1024;

    public ComputeShader m_computeShader;
    public ComputeShader m_computeShaderSum;
    private int m_kernel, m_kernelSum, m_kernelRecord;

    private Particle[] m_particles;
    private ErrorData[] m_particlesError;

    struct Node {
        public Vector3 pos;
        public float force;
        public Node(Vector3 pos, float force) {
            this.pos = pos;
            this.force = force;
        }
    };

    struct Particle {
        public Vector3 position;
        public Vector3 velocity;
    };

    struct ErrorData {
        public Vector3 recordedPosition;
        public float error;
    };


    void cpuRun (uint id, uint iteration, Node[] nodes, Vector3 g) {
        Vector3 recordedPosition = m_particlesError[id + iteration * m_pointsCount].recordedPosition;

        Vector3 position = m_particles[id].position;
        Vector3 velocity = m_particles[id].velocity;

        Vector3 sum_vel = new Vector3(0, 0, 0);
        for (int i = 0; i < 3; i++)
        {
            Vector3 node_pos = nodes[i].pos;
            float distanceToNode = Mathf.Max(2.0f, Vector3.Distance(node_pos, position));
            Vector3 deltaVelocity = Vector3.Normalize(node_pos - position) * nodes[i].force / Mathf.Pow(distanceToNode, 1.7f);
            sum_vel = sum_vel + deltaVelocity;
        }

        velocity += (sum_vel / 6.0f) * 0.5f;
        velocity += g;
        float damping = 0.995f;
        velocity = velocity * damping;

        position += velocity;
        m_particles[id].position = position;
        m_particles[id].velocity = velocity;

        float errorDistance = Vector3.Distance(position, recordedPosition);

        m_particlesError[id + iteration * m_pointsCount].error = errorDistance;
    }

    void cpuRecord (uint id, uint iteration, Node[] nodes, Vector3 g) {
        Vector3 position = m_particles[id].position;
        Vector3 velocity = m_particles[id].velocity;

        Vector3 sum_vel = new Vector3(0, 0, 0);
        for (int i = 0; i < 3; i++) {
            Vector3 node_pos = nodes[i].pos;
            float distanceToNode = Mathf.Max(2.0f, Vector3.Distance(node_pos, position));
            Vector3 deltaVelocity = Vector3.Normalize(node_pos - position) * nodes[i].force / Mathf.Pow(distanceToNode, 1.7f);
            sum_vel = sum_vel + deltaVelocity;
        }

        velocity += (sum_vel / 6.0f) * 0.5f;
        velocity += g;
        float damping = 0.995f;
        velocity = velocity * damping;

        m_particles[id].position = position + velocity;
        m_particles[id].velocity = velocity;

        m_particlesError[id + iteration * m_pointsCount].recordedPosition = m_particles[id].position;
    }

    void cpuRunAll(Node[] nodes, uint iteration)
    {
        Vector3 g = new Vector3(-0.0000918f, 0.0f, 0.0f);
        for (uint i = 0; i < m_particles.Length; i++)
        {
            cpuRun(i, iteration, nodes, g);
        }
    }

    void cpuRecordAll(Node[] nodes, uint iteration) {
        Vector3 g = new Vector3(-0.0000918f, 0.0f, 0.0f);
        for (uint i = 0; i < m_particles.Length; i++)
        {
            cpuRecord(i, iteration, nodes, g);
        }
    }

    float cpuErrorDirection(Node[] nodes, float[] deltas, float deltaScale, float direction) {
        Node[] dirNodes = (Node[])nodes.Clone();

        for (int i = 0; i < dirNodes.Length; i++)
        {
            dirNodes[i].force += direction * deltas[i/**4*/] * deltaScale;
            //Vector3 deltaPos = new Vector3(deltas[i*4+1], deltas[i*4+2], deltas[i*4+3]) * deltaScale*1000.0f;
            //posNodes[i].pos += deltaPos;
        }

        //Run test:
        Particle[] particlesOld = (Particle[])m_particles.Clone();

        for (uint i = 0; i < m_maxIterations; i++)
        {
            cpuRunAll(dirNodes, i);
        }

        float sum = 0f;
        for (int i = 0; i < m_particlesError.Length; i++)
        {
            float error = m_particlesError[i].error;
            sum += error;
        }

        m_particles = particlesOld;

        return sum;
    }

    //https://github.com/yanatan16/golang-spsa/blob/master/spsa.go
    //https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
    float[] cpuEstimateGradient(Node[] nodes, int numberOfErrorVals, float deltaScale)
    {
        float[] deltas = getRandomDeltas(nodes.Length);

        float errorPos = cpuErrorDirection(nodes, deltas, deltaScale, 1.0f);
        float errorNeg = cpuErrorDirection(nodes, deltas, deltaScale, -1.0f);

        // Calculate estimated gradient
        float[] gradient = new float[deltas.Length];
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = (errorPos - errorNeg) / (2.0f * deltas[i]);
            Debug.Log(errorPos + " " + errorNeg + " " + gradient[i]);
        }

        return gradient;
    }

    void cpuRecordSimulation(Node[] trainingNodes)
    {
        Particle[] particlesOld = (Particle[])m_particles.Clone();

        for (uint i = 0; i < m_maxIterations; i++)
        {
            cpuRecordAll(trainingNodes, i);
        }
        m_particles = particlesOld;
    }

    void cpuTrainValidateModel(Node[] testNodes)
    {
        float gradientScale = 0.000001f;
        int numberOfErrorVals = m_maxIterations * m_pointsCount;

        float bestError = float.PositiveInfinity;
        int bestIteration = 0;
        Node[] bestNodes = (Node[])testNodes.Clone();

        for (int j = 0; j < 50; j++)
        {
            //m_iteration = j;
            float[] gradient = cpuEstimateGradient(testNodes, numberOfErrorVals, 0.0001f);

            Debug.Log("New Gradient: " + gradient[0] + " " + gradient[1] + " " + gradient[2]);

            for (int i = 0; i < testNodes.Length; i++)
            {
                testNodes[i].force -= gradient[i] * gradientScale;
                //Vector3 deltaPos = new Vector3(gradient[i*4+1], gradient[i*4+2], gradient[i*4+3]) * gradientScale * 100.0f;
                //testNodes[i].pos -= deltaPos;
            }

            Debug.Log("Test nodes: " + testNodes[0].force);
            Debug.Log("Test nodes: " + testNodes[1].force);
            Debug.Log("Test nodes: " + testNodes[2].force);

            float sum = 0f;
            for (int i = 0; i < m_particlesError.Length; i++)
            {
                float error = m_particlesError[i].error;
                sum += error;
            }

            float avgError = sum / numberOfErrorVals;
            if (avgError < bestError)
            {
                bestError = avgError;
                bestIteration = j;
                bestNodes = (Node[])testNodes.Clone();
            }

            Debug.Log(j + " Error: " + sum / numberOfErrorVals);
            if (avgError < 0.005f)
            {
                //break;
            }
        }

        Debug.Log("Best Error: " + bestIteration + " " + bestError);

        for (int i = 0; i < bestNodes.Length; i++)
        {
            Debug.Log(i + " " + bestNodes[i].pos + " " + bestNodes[i].force);
        }
    }

    void initializeParticles() {
        m_dimensionWidth = 256;
        m_dimensionHeight = 1;
        m_dimensionDepth = 1;

        m_particles = new Particle[m_dimensionWidth * m_dimensionHeight * m_dimensionDepth];
        m_particlesError = new ErrorData[m_dimensionWidth * m_dimensionHeight * m_dimensionDepth * m_maxIterations];

        for (int i = 0; i < m_particlesError.Length; i++) {
            m_particlesError[i].recordedPosition = new Vector3();
            m_particlesError[i].error = 0.0f;
        }

        //m_points = new Vector3[m_dimensionWidth * m_dimensionHeight * m_dimensionDepth];
        //m_velocities = new Vector3[m_points.Length];
        Vector3 startingPosition = new Vector3(20.0f, 0.0f, 0.0f);
        
        float deltaPos = 0.05f;

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

        particleComputebuffer = new ComputeBuffer (m_pointsCount, Marshal.SizeOf(typeof(Particle)), ComputeBufferType.Default);
        particleComputebuffer.SetData(m_particles);
        m_pointRenderer.material.SetBuffer ("_ParticleData", particleComputebuffer);
        m_computeShader.SetBuffer(m_kernel, "_ParticleData", particleComputebuffer);
        m_computeShader.SetBuffer(m_kernelRecord, "_ParticleData", particleComputebuffer);

        errorDataComputebuffer = new ComputeBuffer (m_particlesError.Length, Marshal.SizeOf(typeof(ErrorData)), ComputeBufferType.Default);
        errorDataComputebuffer.SetData(m_particlesError);
        m_pointRenderer.material.SetBuffer ("_ErrorData", errorDataComputebuffer);
        m_computeShader.SetBuffer(m_kernel, "_ErrorData", errorDataComputebuffer);
        m_computeShader.SetBuffer(m_kernelRecord, "_ErrorData", errorDataComputebuffer);
        m_computeShaderSum.SetBuffer(m_kernelSum, "g_data", errorDataComputebuffer);
    }

    void resetParticles() {
        particleComputebuffer.SetData(m_particles);
    }

    float[] getRandomDeltas(int numberOfNodes) {
        float[] randomDeltas = new float[numberOfNodes];

        for (int i = 0; i < randomDeltas.Length; i++) {
            if (UnityEngine.Random.Range(0.0f, 1.0f) > 0.5f) {
                randomDeltas[i] = 1.0f;
	        } else {
                randomDeltas[i] = -1.0f;
            }
        }
        return randomDeltas;
    }

    float gpuErrorDirection(Node[] nodes, float[] deltas, float deltaScale, float direction)
    {
        Node[] dirNodes = (Node[])nodes.Clone();

        for (int i = 0; i < dirNodes.Length; i++)
        {
            dirNodes[i].force += direction * deltas[i/**4*/] * deltaScale;
            //Vector3 deltaPos = new Vector3(deltas[i*4+1], deltas[i*4+2], deltas[i*4+3]) * deltaScale*1000.0f;
            //posNodes[i].pos += deltaPos;
        }

        //Store the current state of the particles somehow
        particleComputebuffer.GetData(m_particles);

        //Run test:
        nodesComputeBuffer.SetData(dirNodes);
        for (int i = 0; i < m_maxIterations; i++)
        {
            m_computeShader.SetInt("_iteration", i);
            m_computeShader.Dispatch(m_kernel, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
        }

        errorDataComputebuffer.GetData(m_particlesError);

        //Restore the previous state of the particles somehow
        particleComputebuffer.SetData(m_particles);

        float sum = 0f;
        for (int i = 0; i < m_particlesError.Length; i++)
        {
            float error = m_particlesError[i].error;
            sum += error;
        }

        return sum;
    }

    //https://github.com/yanatan16/golang-spsa/blob/master/spsa.go
    //https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
    float[] gpuEstimateGradient(Node[] nodes, int numberOfErrorVals, float deltaScale) {
        float[] deltas = getRandomDeltas(nodes.Length);

        float errorPos = gpuErrorDirection(nodes, deltas, deltaScale, 1.0f) /*/ numberOfErrorVals*/;
        float errorNeg = gpuErrorDirection(nodes, deltas, deltaScale, -1.0f) /*/ numberOfErrorVals*/;

        // Calculate estimated gradient
        float[] gradient = new float[deltas.Length];
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = (errorPos - errorNeg) / (2.0f * deltas[i]);
            Debug.Log(errorPos + " " + errorNeg + " " + gradient[i]);
        }

        return gradient;
    }

    void gpuRecordSimulation(Node[] trainingNodes) {
        for (int i = 0; i < m_maxIterations; i++)
        {
            m_computeShader.SetInt("_iteration", i);
            m_computeShader.Dispatch(m_kernelRecord, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
        }
        //Reset particles but not the recorded particles:
        resetParticles();
    }

    void gpuTrainValidateModel(Node[] testNodes) {
        float gradientScale = 0.000001f;
        int numberOfErrorVals = m_maxIterations * m_pointsCount;

        float bestError = float.PositiveInfinity;
        int bestIteration = 0;
        Node[] bestNodes = (Node[])testNodes.Clone();

        for (int j = 0; j < 500; j++)
        {
            //m_iteration = j;
            float[] gradient = gpuEstimateGradient(testNodes, numberOfErrorVals, 0.0001f);

            Debug.Log("New Gradient: " + gradient[0] + " " + gradient[1] + " " + gradient[2]);

            for (int i = 0; i < testNodes.Length; i++)
            {
                testNodes[i].force -= gradient[i] * gradientScale;
                //Vector3 deltaPos = new Vector3(gradient[i*4+1], gradient[i*4+2], gradient[i*4+3]) * gradientScale * 100.0f;
                //testNodes[i].pos -= deltaPos;
            }

            Debug.Log("Test nodes: " + testNodes[0].force);
            Debug.Log("Test nodes: " + testNodes[1].force);
            Debug.Log("Test nodes: " + testNodes[2].force);

            float sum = 0f;
            for (int i = 0; i < m_particlesError.Length; i++)
            {
                float error = m_particlesError[i].error;
                sum += error;
            }

            float avgError = sum / numberOfErrorVals;
            if (avgError < bestError)
            {
                bestError = avgError;
                bestIteration = j;
                bestNodes = (Node[])testNodes.Clone();
            }

            Debug.Log(j + " Error: " + sum / numberOfErrorVals);
            if (avgError < 0.005f)
            {
                //break;
            }
        }

        Debug.Log("Best Error: " + bestIteration + " " + bestError);

        for (int i = 0; i < bestNodes.Length; i++)
        {
            Debug.Log(i + " " + bestNodes[i].pos + " " + bestNodes[i].force);
        }
    }

    void Start () {
        Screen.SetResolution(720, 1280, true);
        m_pointRenderer = GetComponent<Renderer>();
        m_kernel = m_computeShader.FindKernel("CSMain");
        m_kernelSum = m_computeShaderSum.FindKernel("reduce1");
        m_kernelRecord = m_computeShader.FindKernel("record");

        //nodes for training:
        Node[] trainingNodes = {
            new Node(new Vector3(0.0f, 0.0f, 0.0f), -0.005f),
            new Node(new Vector3(2.0f, 3.0f, 0.0f), -0.005f),
            new Node(new Vector3(-5.0f, 2.0f, 0.0f), 0.02f)
        };

        Node[] testNodes = {
            new Node(new Vector3(0.0f, 0.0f, 0.0f), 0.0f),
            new Node(new Vector3(2.0f, 3.0f, 0.0f), 0.0f),
            new Node(new Vector3(-5.0f, 2.0f, 0.0f), 0.0f)
        };

        initializeParticles();

        nodesComputeBuffer = new ComputeBuffer(trainingNodes.Length, Marshal.SizeOf(typeof(Node)), ComputeBufferType.Default);
        nodesComputeBuffer.SetData(trainingNodes);
        m_computeShader.SetBuffer(m_kernel, "_Nodes", nodesComputeBuffer);
        m_computeShader.SetBuffer(m_kernelRecord, "_Nodes", nodesComputeBuffer);

        m_pointRenderer.material.SetInt("_PointsCount", m_pointsCount);
        float aspect = Camera.main.GetComponent<Camera>().aspect;
        m_pointRenderer.material.SetFloat("aspect", aspect);
        Vector4 trans = transform.position;
        m_pointRenderer.material.SetVector("trans", trans);
        m_pointRenderer.material.SetInt("_TextureSwitchFrameNumber", m_textureSwitchFrameNumber);   

        Debug.Log("Number of points: " + m_pointsCount);

        m_computeShader.SetInt("_particleCount", m_pointsCount);

        //cpuRecordSimulation(trainingNodes);

        //cpuTrainValidateModel(testNodes);

        gpuRecordSimulation(trainingNodes);

        gpuTrainValidateModel(testNodes);

        m_iteration = 0;
        //resetParticles();

        /*for (int j = 0; j < 5; j++) {
            m_iteration = 0;
            resetParticles();

            // update test node forces using gradient
            float delta = -0.001f;
            testNodes[0].force += delta;
            testNodes[1].force += delta;
            testNodes[2].force -= delta*4.0f;

            nodesComputeBuffer.SetData(testNodes);

            //Debug.Log("Running validation:");
    

            for (int i = 0; i < m_maxIterations; i++) {
                m_computeShader.SetInt("_iteration", i);
                m_computeShader.Dispatch(m_kernel, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
            }

            errorDataComputebuffer.GetData(m_particlesError);

            //Debug.Log("Error: " + m_particlesError[0].error);

            float sum = 0f;
            for (int i = 0; i < m_particlesError.Length; i++) {
                float error = m_particlesError[i].error;
                //if (error > 0.001f) {
                //    Debug.Log("Error: " + error);
                //}
                sum += error;
            }
            //Debug.Log("Error Sum: " + sum);

            m_computeShaderSum.Dispatch(m_kernelSum, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
            m_computeShaderSum.Dispatch(m_kernelSum, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
            //m_computeShaderSum.Dispatch(m_kernelSum, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
            //m_computeShaderSum.Dispatch(m_kernelSum, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
            //m_computeShaderSum.Dispatch(m_kernelSum, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
            //m_computeShaderSum.Dispatch(m_kernelSum, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);


            //Run another compute dispatch where we use the sum to determine the gradient. And then do everything again until the error is very small.
            //It shouldn't be necessary to actually do the division since that just gives you a correct error. We're only interested in optimizing it, so division I assume is unnecessary.

            errorDataComputebuffer.GetData(m_particlesError);

            Debug.Log("Error: " + m_particlesError[0].error);

        }*/

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
        m_computeShader.Dispatch(m_kernel, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);

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
        particleComputebuffer.Release();
        errorDataComputebuffer.Release();
        nodesComputeBuffer.Release();
    }
    /*private void Dispatch()
    {
        float deltaTime = Time.fixedTime - m_lastFrameTime;

        //Debug.Log("Delta time: " + deltaTime);

        m_computeShader.SetFloat("deltaTime", deltaTime*1.0f);

        m_computeShader.Dispatch(m_kernel, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
    }*/
}
