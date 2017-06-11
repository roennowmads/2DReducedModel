using UnityEngine;

public class CPUParticles {
    static float[] getRandomDeltas(int numberOfNodes)
    {
        float[] randomDeltas = new float[numberOfNodes];

        for (int i = 0; i < randomDeltas.Length; i++)
        {
            if (UnityEngine.Random.Range(0.0f, 1.0f) > 0.5f)
            {
                randomDeltas[i] = 1.0f;
            }
            else
            {
                randomDeltas[i] = -1.0f;
            }
        }
        return randomDeltas;
    }

    static void cpuRun(uint id, uint iteration, ReducedModel.Node[] nodes, Vector3 g, ref ReducedModel.Particle[] particles, ref float[] particlesError, ref Vector3[] particlesRecorded, int pointsCount)
    {
        Vector3 recordedPosition = particlesRecorded[id + iteration * pointsCount];

        Vector3 position = particles[id].position;
        Vector3 velocity = particles[id].velocity;

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
        particles[id].position = position;
        particles[id].velocity = velocity;

        float errorDistance = Vector3.Distance(position, recordedPosition);

        particlesError[id + iteration * pointsCount] = errorDistance;
    }

    static void cpuRecord(uint id, uint iteration, ReducedModel.Node[] nodes, Vector3 g, ref ReducedModel.Particle[] particles, ref Vector3[] particlesRecorded, int pointsCount)
    {
        Vector3 position = particles[id].position;
        Vector3 velocity = particles[id].velocity;

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

        particles[id].position = position + velocity;
        particles[id].velocity = velocity;

        particlesRecorded[id + iteration * pointsCount] = particles[id].position;
    }

    static void cpuRunAll(ReducedModel.Node[] nodes, uint iteration, ref ReducedModel.Particle[] particles, ref float[] particlesError, ref Vector3[] particlesRecorded, int pointsCount)
    {
        Vector3 g = new Vector3(-0.0000918f, 0.0f, 0.0f);
        for (uint i = 0; i < particles.Length; i++)
        {
            cpuRun(i, iteration, nodes, g, ref particles, ref particlesError, ref particlesRecorded, pointsCount);
        }
    }

    static void cpuRecordAll(ReducedModel.Node[] nodes, uint iteration, ref ReducedModel.Particle[] particles, ref Vector3[] particlesRecorded, int pointsCount)
    {
        Vector3 g = new Vector3(-0.0000918f, 0.0f, 0.0f);
        for (uint i = 0; i < particles.Length; i++)
        {
            cpuRecord(i, iteration, nodes, g, ref particles, ref particlesRecorded, pointsCount);
        }
    }

    static float cpuErrorDirection(ReducedModel.Node[] nodes, float[] deltas, float deltaScale, float direction, ref ReducedModel.Particle[] particles, ref float[] particlesError, ref Vector3[] particlesRecorded, int pointsCount, int maxIterations)
    {
        ReducedModel.Node[] dirNodes = (ReducedModel.Node[])nodes.Clone();

        for (int i = 0; i < dirNodes.Length; i++)
        {
            dirNodes[i].force += direction * deltas[i/**4*/] * deltaScale;
            //Vector3 deltaPos = new Vector3(deltas[i*4+1], deltas[i*4+2], deltas[i*4+3]) * deltaScale*1000.0f;
            //posNodes[i].pos += deltaPos;
        }

        //Run test:
        ReducedModel.Particle[] particlesOld = (ReducedModel.Particle[])particles.Clone();

        for (uint i = 0; i < maxIterations; i++)
        {
            cpuRunAll(dirNodes, i, ref particles, ref particlesError, ref particlesRecorded, pointsCount);
        }

        float sum = 0f;
        for (int i = 0; i < particlesError.Length; i++)
        {
            float error = particlesError[i];
            sum += error;
        }

        particles = particlesOld;

        return sum;
    }

    //https://github.com/yanatan16/golang-spsa/blob/master/spsa.go
    //https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
    static float[] cpuEstimateGradient(ReducedModel.Node[] nodes, int numberOfErrorVals, float deltaScale, ref ReducedModel.Particle[] particles, ref float[] particlesError, ref Vector3[] particlesRecorded, int pointsCount, int maxIterations)
    {
        float[] deltas = getRandomDeltas(nodes.Length);

        float errorPos = cpuErrorDirection(nodes, deltas, deltaScale, 1.0f, ref particles, ref particlesError, ref particlesRecorded, pointsCount, maxIterations);
        float errorNeg = cpuErrorDirection(nodes, deltas, deltaScale, -1.0f, ref particles, ref particlesError, ref particlesRecorded, pointsCount, maxIterations);

        // Calculate estimated gradient
        float[] gradient = new float[deltas.Length];
        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = (errorPos - errorNeg) / (2.0f * deltas[i]);
            Debug.Log(errorPos + " " + errorNeg + " " + gradient[i]);
        }

        return gradient;
    }

    public static void cpuRecordSimulation(ReducedModel.Node[] trainingNodes, ref ReducedModel.Particle[] particles, ref Vector3[] particlesRecorded, int pointsCount, int maxIterations)
    {
        ReducedModel.Particle[] particlesOld = (ReducedModel.Particle[])particles.Clone();

        for (uint i = 0; i < maxIterations; i++)
        {
            cpuRecordAll(trainingNodes, i, ref particles, ref particlesRecorded, pointsCount);
        }
        particles = particlesOld;
    }

    public static void cpuTrainValidateModel(ReducedModel.Node[] testNodes, ref ReducedModel.Particle[] particles, ref float[] particlesError, ref Vector3[] particlesRecorded, int pointsCount, int maxIterations)
    {
        float gradientScale = 0.000001f;
        int numberOfErrorVals = maxIterations * pointsCount;

        float bestError = float.PositiveInfinity;
        int bestIteration = 0;
        ReducedModel.Node[] bestNodes = (ReducedModel.Node[])testNodes.Clone();

        for (int j = 0; j < 50; j++)
        {
            //m_iteration = j;
            float[] gradient = cpuEstimateGradient(testNodes, numberOfErrorVals, 0.0001f, ref particles, ref particlesError, ref particlesRecorded, pointsCount, maxIterations);

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
            for (int i = 0; i < particlesError.Length; i++)
            {
                float error = particlesError[i];
                sum += error;
            }

            float avgError = sum / numberOfErrorVals;
            if (avgError < bestError)
            {
                bestError = avgError;
                bestIteration = j;
                bestNodes = (ReducedModel.Node[])testNodes.Clone();
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
}
