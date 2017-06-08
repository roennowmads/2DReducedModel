using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System.Text;
using System.IO;
using UnityEngine.Rendering;
using System.Runtime.InteropServices;
using System;

public class ReducedModel : MonoBehaviour {
    public string m_valueDataPath = "OilRigData";
    public int m_lastFrameIndex = 25;
    
    public float m_frameSpeed = 5.0f;
    public int m_textureSideSizePower = 14;

    private int m_pointsCount = 61440;
    private Renderer pointRenderer;

    private int m_textureSideSize;
    private int m_textureSize;
    private int m_lookupTextureSize = 256; //Since the values in the value texture are only 0-255, it doesn't make sense to have more values here.

    private int m_textureSwitchFrameNumber = -1;

    private ComputeBuffer positionComputebuffer, velocityComputebuffer, nodesComputeBuffer, positionRecordComputebuffer;
    private int m_dimensionWidth, m_dimensionHeight, m_dimensionDepth;

    private float m_updateFrequency = 0.0333f;
    private float m_currentTime;
    private float m_lastFrameTime = 0.0f;
    private int m_iteration = 0;
    private int m_maxIterations = 5000;

    public ComputeShader computeShader;
    private int m_kernel;

    struct Node {
        Vector3 pos;
        float force;
        public Node(Vector3 pos, float force) {
            this.pos = pos;
            this.force = force;
        }
    };

    public int getPointCount () {
        return m_pointsCount;
    }

    public void setPointCount (int pointsCount) {
        m_pointsCount = pointsCount;
    }

    public void changePointsCount(int increment) {
        m_pointsCount += increment;
    }

    Texture2D createColorLookupTexture() {
        int numberOfValues = m_lookupTextureSize;

        Texture2D lookupTexture = new Texture2D(m_lookupTextureSize, 1, TextureFormat.RGB24, false, false);
        lookupTexture.filterMode = FilterMode.Point;
        lookupTexture.anisoLevel = 1;

        for (int i = 0; i < numberOfValues; i++) {
            float textureIndex = i;

            //0 - 255 --> 0.0 - 1.0
            float value = textureIndex / numberOfValues;

            var a = (1.0f - value) / 0.25f; //invert and group
            float X = Mathf.Floor(a);   //this is the integer part
            float Y = a - X; //fractional part from 0 to 255

            Color color;

            switch ((int)X) {
                case 0:
                    color = new Color(1.0f, Y, 0);
                    break;
                case 1:
                    color = new Color((1.0f - Y), 1.0f, 0);
                    break;
                case 2:
                    color = new Color(0, 1.0f, Y);
                    break;
                case 3:
                    color = new Color(0, (1.0f - Y), 1.0f);
                    break;
                case 4:
                    color = new Color(0, 0, 1.0f);
                    break;
                default:
                    color = new Color(1.0f, 0, 0);
                    break;
            }            
            lookupTexture.SetPixel(i, 0, color); 

            //alternatives: (necessary if I want to store only one component per pixel)
            //tex.LoadRawTextureData()
            //tex.SetPixels(x, y, width, height, colors.ToArray()); 
            //pixels are stored in rectangle blocks... maybe it would actually be better for caching anyway? problem is a frame's colors would need to fit in a rectangle.
        }

        lookupTexture.Apply();

        return lookupTexture;
    }



    void Start () {
        //m_textureSideSize = 1 << m_textureSideSizePower;
        //m_textureSize = m_textureSideSize * m_textureSideSize;


        //nodes for training:
         Node[] trainingNodes = {
            new Node(new Vector3(0.0f, 0.0f, 0.0f), -0.005f),
            new Node(new Vector3(2.0f, 3.0f, 0.0f), -0.005f),
            new Node(new Vector3(-5.0f, 2.0f, 0.0f), 0.02f)
        };


        Screen.SetResolution(720, 1280, true);

        m_dimensionWidth = 256;
        m_dimensionHeight = 256;
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

        //Set up mesh:
        m_pointsCount = points.Length;

         //Set up textures:
        /*Texture2D colorTexture = createColorLookupTexture();
        
        //We don't need more precision than the resolution of the colorTexture. 10 bits is sufficient for 1024 different color values.
        //That means we can pack 3 10bit integer values into a pixel 
        //Texture2D texture = new Texture2D(m_textureSize, m_textureSize, TextureFormat.RFloat, false, false);
        Texture2D texture = new Texture2D(m_textureSideSize, m_textureSideSize, TextureFormat.Alpha8, false, false);
        texture.filterMode = FilterMode.Point;
        texture.wrapMode = TextureWrapMode.Repeat;

        Texture2D texture2 = new Texture2D(m_textureSideSize, m_textureSideSize, TextureFormat.Alpha8, false, false);
        texture2.filterMode = FilterMode.Point;
        texture2.wrapMode = TextureWrapMode.Repeat;*/

        /*bool supportsTextureFormat = SystemInfo.SupportsTextureFormat(TextureFormat.R16); 
        if (supportsTextureFormat) {
            Debug.Log("");
        }*/

        /*texture.anisoLevel = 1;
        readPointsFile1Value(texture, texture2);
        pointRenderer = GetComponent<Renderer>();
        pointRenderer.material.mainTexture = texture;
        //pointRenderer.material.SetTexture("_MainTex2", texture2);
        pointRenderer.material.SetTexture("_ColorTex", colorTexture);*/

        m_kernel = computeShader.FindKernel("CSMain");

        pointRenderer = GetComponent<Renderer>();

        positionComputebuffer = new ComputeBuffer (m_pointsCount, Marshal.SizeOf(typeof(Vector3)), ComputeBufferType.GPUMemory);
        positionComputebuffer.SetData(points);
        pointRenderer.material.SetBuffer ("_Positions", positionComputebuffer);

        velocityComputebuffer = new ComputeBuffer (m_pointsCount, Marshal.SizeOf(typeof(Vector3)), ComputeBufferType.GPUMemory);
        velocityComputebuffer.SetData(velocities);
        pointRenderer.material.SetBuffer ("_Velocities", velocityComputebuffer);

        nodesComputeBuffer = new ComputeBuffer(trainingNodes.Length, Marshal.SizeOf(typeof(Node)), ComputeBufferType.GPUMemory);
        nodesComputeBuffer.SetData(trainingNodes);

        positionRecordComputebuffer = new ComputeBuffer (m_pointsCount*m_maxIterations, Marshal.SizeOf(typeof(Vector3)), ComputeBufferType.GPUMemory);

        computeShader.SetBuffer(m_kernel, "_Positions", positionComputebuffer);
        computeShader.SetBuffer(m_kernel, "_Velocities", velocityComputebuffer);
        computeShader.SetBuffer(m_kernel, "_Nodes", nodesComputeBuffer);
        computeShader.SetBuffer(m_kernel, "_PositionsRecord", positionRecordComputebuffer);

        pointRenderer.material.SetInt("_PointsCount", m_pointsCount);
        float aspect = Camera.main.GetComponent<Camera>().aspect;
        pointRenderer.material.SetFloat("aspect", aspect);
        Vector4 trans = transform.position;
        pointRenderer.material.SetVector("trans", trans);
        pointRenderer.material.SetInt("_Magnitude", m_textureSideSizePower);
        pointRenderer.material.SetInt("_TextureSwitchFrameNumber", m_textureSwitchFrameNumber);   

        Debug.Log("Number of points: " + m_pointsCount);

        computeShader.SetInt("_particleCount", m_pointsCount);

        computeShader.SetBool("_recording", true);

        //Record simulation:
        for (int i = 0; i < m_maxIterations; i++) {
            computeShader.SetInt("_iteration", i);
            Dispatch();
        }

        computeShader.SetBool("_recording", false);

        m_iteration = 0;

    }
	
	// Update is called once per frame
	void Update () {
        //Debug.Log(Time.fixedTime);

        int t = ((int)(Time.fixedTime * m_frameSpeed)) % m_lastFrameIndex;

        //t = 29;

        //Debug.Log(count);

        //Debug.Log(t);
        pointRenderer.material.SetInt("_FrameTime", t);
        float aspect = Camera.main.GetComponent<Camera>().aspect;
        pointRenderer.material.SetFloat("aspect", aspect);

        //Debug.Log("Support instancing: " + SystemInfo.supportsInstancing);

        //int a = pointRenderer.material.GetInt("_FrameTime");

        //Debug.Log(a);
        //m_currentTime += Time.deltaTime;
        //if (m_currentTime >= m_updateFrequency)
        {
            //m_currentTime = 0.0f;
            computeShader.SetInt("_iteration", m_iteration);
            Dispatch();
            //m_lastFrameTime = Time.fixedTime;
        }

        m_iteration++;
        m_iteration = m_iteration % m_maxIterations;
    }

    private void OnRenderObject()
    {
        //m_currentTime += Time.deltaTime;
        //if (m_currentTime >= m_updateFrequency)
        {
            m_currentTime = 0.0f;
            pointRenderer.material.SetPass(0);
            pointRenderer.material.SetMatrix("model", transform.localToWorldMatrix);
            //Graphics.DrawProcedural(MeshTopology.Points, 1, m_pointsCount);
            Graphics.DrawProcedural(MeshTopology.Points, m_pointsCount);  // index buffer.
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
    }
    private void Dispatch()
    {
        float deltaTime = Time.fixedTime - m_lastFrameTime;

        Debug.Log("Delta time: " + deltaTime);

        computeShader.SetFloat("deltaTime", deltaTime*1.0f);

        computeShader.Dispatch(m_kernel, m_dimensionWidth, m_dimensionHeight, m_dimensionDepth);
    }
}
