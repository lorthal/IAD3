using System;
namespace RadialNetworkTrain
{
  class RadialTrainProgram
  {
    static void Main(string[] args)
    {
      Console.WriteLine("\nBegin radial basis function (RBF) network training demo\n");
      Console.WriteLine("Goal is to train an RBF network on iris flower data to predict");
      Console.WriteLine("species from sepal length and width, and petal length and width.");

      // (0,0,1) = Iris setosa; (0,1,0) = Iris versicolor; (1,0,0) = Iris virginica
      // program assumes x-data precedes y-data

      double[][] allData = new double[30][];
      allData[0] = new double[] { -0.784, 1.255, -1.332, -1.306, 0, 0, 1 };
      allData[1] = new double[] { -0.995, -0.109, -1.332, -1.306, 0, 0, 1 };
      allData[2] = new double[] { -1.206, 0.436, -1.386, -1.306, 0, 0, 1 };
      allData[3] = new double[] { -1.312, 0.164, -1.278, -1.306, 0, 0, 1 };
      allData[4] = new double[] { -0.890, 1.528, -1.332, -1.306, 0, 0, 1 };
      allData[5] = new double[] { -0.468, 2.346, -1.170, -1.048, 0, 0, 1 };
      allData[6] = new double[] { -1.312, 0.982, -1.332, -1.177, 0, 0, 1 };
      allData[7] = new double[] { -0.890, 0.982, -1.278, -1.306, 0, 0, 1 };
      allData[8] = new double[] { -1.523, -0.382, -1.332, -1.306, 0, 0, 1 };
      allData[9] = new double[] { -0.995, 0.164, -1.278, -1.435, 0, 0, 1 };

      allData[10] = new double[] { 1.220, 0.436, 0.452, 0.241, 0, 1, 0 };
      allData[11] = new double[] { 0.587, 0.436, 0.344, 0.370, 0, 1, 0 };
      allData[12] = new double[] { 1.115, 0.164, 0.560, 0.370, 0, 1, 0 };
      allData[13] = new double[] { -0.362, -2.019, 0.074, 0.112, 0, 1, 0 };
      allData[14] = new double[] { 0.693, -0.655, 0.398, 0.370, 0, 1, 0 };
      allData[15] = new double[] { -0.151, -0.655, 0.344, 0.112, 0, 1, 0 };
      allData[16] = new double[] { 0.482, 0.709, 0.452, 0.498, 0, 1, 0 };
      allData[17] = new double[] { -0.995, -1.746, -0.305, -0.275, 0, 1, 0 };
      allData[18] = new double[] { 0.798, -0.382, 0.398, 0.112, 0, 1, 0 };
      allData[19] = new double[] { -0.679, -0.927, 0.020, 0.241, 0, 1, 0 };

      allData[20] = new double[] { 0.482, 0.709, 1.155, 1.659, 1, 0, 0 };
      allData[21] = new double[] { -0.046, -0.927, 0.669, 0.885, 1, 0, 0 };
      allData[22] = new double[] { 1.326, -0.109, 1.101, 1.143, 1, 0, 0 };
      allData[23] = new double[] { 0.482, -0.382, 0.939, 0.756, 1, 0, 0 };
      allData[24] = new double[] { 0.693, -0.109, 1.047, 1.272, 1, 0, 0 };
      allData[25] = new double[] { 1.853, -0.109, 1.479, 1.143, 1, 0, 0 };
      allData[26] = new double[] { -0.995, -1.473, 0.344, 0.627, 1, 0, 0 };
      allData[27] = new double[] { 1.537, -0.382, 1.317, 0.756, 1, 0, 0 };
      allData[28] = new double[] { 0.904, -1.473, 1.047, 0.756, 1, 0, 0 };
      allData[29] = new double[] { 1.431, 1.528, 1.209, 1.659, 1, 0, 0 };

      Console.WriteLine("\nFirst four and last line of normalized, encoded input data:\n");
      Helpers.ShowMatrix(allData, 4, 3, true, true);

      Console.WriteLine("\nSplitting data into 80%-20% train and test sets");
      double[][] trainData = null;
      double[][] testData = null;
      int seed = 8; // gives a good demo
      GetTrainTest(allData, seed, out trainData, out testData); // 80-20 hold-out 

      Console.WriteLine("\nTraining data: \n");
      Helpers.ShowMatrix(trainData, trainData.Length, 3, true, false);
      Console.WriteLine("\nTest data:\n");
      Helpers.ShowMatrix(testData, testData.Length, 3, true, false);

      Console.WriteLine("\nCreating a 4-5-3 radial basis function network");
      int numInput = 4;
      int numHidden = 5;
      int numOutput = 3;
      RadialNetwork rn = new RadialNetwork(numInput, numHidden, numOutput);

      Console.WriteLine("\nBeginning RBF training\n");
      int maxIterations = 100; // max for PSO 
      double[] bestWeights = rn.Train(trainData, maxIterations);

      Console.WriteLine("\nEvaluating result RBF classification accuracy on the test data");
      rn.SetWeights(bestWeights);

      double acc = rn.Accuracy(testData);
      Console.WriteLine("Classification accuracy = " + acc.ToString("F4"));

      Console.WriteLine("\nEnd RBF network training demo\n");
      Console.ReadLine();

    } // Main()

    static void GetTrainTest(double[][] allData, int seed, out double[][] trainData, out double[][] testData)
    {
      // 80-20 hold-out validation
      int[] allIndices = new int[allData.Length];
      for (int i = 0; i < allIndices.Length; ++i)
        allIndices[i] = i;

      Random rnd = new Random(seed);
      for (int i = 0; i < allIndices.Length; ++i) // shuffle indices
      {
        int r = rnd.Next(i, allIndices.Length);
        int tmp = allIndices[r];
        allIndices[r] = allIndices[i];
        allIndices[i] = tmp;
      }

      int numTrain = (int)(0.80 * allData.Length);
      int numTest = allData.Length - numTrain;

      trainData = new double[numTrain][];
      testData = new double[numTest][];

      int j = 0; 
      for (int i = 0; i < numTrain; ++i)
        trainData[i] = allData[allIndices[j++]];
      for (int i = 0; i < numTest; ++i)
        testData[i] = allData[allIndices[j++]];

    } // GetTrainTest



  } // class Program


  public class RadialNetwork
  {
    private static Random rnd = null;
    private int numInput;
    private int numHidden;
    private int numOutput;
    private double[] inputs;
    private double[][] centroids;
    private double[] widths;
    private double[][] hoWeights;
    private double[] oBiases;
    private double[] outputs;

    public RadialNetwork(int numInput, int numHidden, int numOutput)
    {
      rnd = new Random(0);
      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;
      this.inputs = new double[numInput];
      this.centroids = MakeMatrix(numHidden, numInput);
      this.widths = new double[numHidden];
      this.hoWeights = MakeMatrix(numHidden, numOutput);
      this.oBiases = new double[numOutput];
      this.outputs = new double[numOutput];
    } // ctor

    private static double[][] MakeMatrix(int rows, int cols) // helper for ctor
    {
      double[][] result = new double[rows][];
      for (int r = 0; r < rows; ++r)
        result[r] = new double[cols];
      return result;
    }

    // -- methods related to getting and setting centroids, widths, weights, bias values ----------

    public void SetWeights(double[] weights)
    {
      // this.hoWeights has numHidden row and numOutput cols
      // this.oBiases has numOutput values
      if (weights.Length != (numHidden * numOutput) + numOutput)
        throw new Exception("Bad weights length in SetWeights");
      int k = 0; // ptr into weights
      for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
          this.hoWeights[i][j] = weights[k++];
      for (int i = 0; i < numOutput; ++i)
        this.oBiases[i] = weights[k++];
    }

    public double[] GetWeights()
    {
      double[] result = new double[numHidden * numOutput + numOutput];
      int k = 0;
      for (int i = 0; i < numHidden; ++i)
        for (int j = 0; j < numOutput; ++j)
          result[k++] = this.hoWeights[i][j];
      for (int i = 0; i < numOutput; ++i)
        result[k++] = this.oBiases[i];
      return result;
    }

    // put GetAllWeights() and SetAllWeights() here to fetch and set
    // centroid, width, weight, and bias values as a group

    // -- methods related to training error and test classification accuracy ----------------------

    private double MeanSquaredError(double[][] trainData, double[] weights)
    {
      // assumes that centroids and widths have been set!
      this.SetWeights(weights); // copy the weights to valuate in
      
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // targets
      double sumSquaredError = 0.0;
      for (int i = 0; i < trainData.Length; ++i) // walk through each trainingb data item
      {
        // following assumes data has all x-values first, followed by y-values!
        Array.Copy(trainData[i], xValues, numInput); // extract inputs
        Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // extract targets
        double[] yValues = this.ComputeOutputs(xValues); // compute the outputs using centroids, widths, weights, bias values
        for (int j = 0; j < yValues.Length; ++j)
          sumSquaredError += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
      }
      return sumSquaredError / trainData.Length;
    }

    // consider MeanCrossEntropy() here as an alternative to MSE

    public double Accuracy(double[][] testData)
    {
      // percentage correct using winner-takes all
      int numCorrect = 0;
      int numWrong = 0;
      double[] xValues = new double[numInput]; // inputs
      double[] tValues = new double[numOutput]; // targets
      double[] yValues; // computed Y

      for (int i = 0; i < testData.Length; ++i)
      {
        Array.Copy(testData[i], xValues, numInput); // parse test data into x-values and t-values
        Array.Copy(testData[i], numInput, tValues, 0, numOutput);
        yValues = this.ComputeOutputs(xValues);
        int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?
        if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
    }

    private static int MaxIndex(double[] vector) // helper for Accuracy()
    {
      // index of largest value
      int bigIndex = 0;
      double biggestVal = vector[0];
      for (int i = 0; i < vector.Length; ++i)
      {
        if (vector[i] > biggestVal)
        {
          biggestVal = vector[i]; bigIndex = i;
        }
      }
      return bigIndex;
    }

    // -- methods related to RBF network input-output mechanism -----------------------------------

    public double[] ComputeOutputs(double[] xValues)
    {
      // use centroids, widths, weights and input xValues to compute, store, and return numOutputs output values
      Array.Copy(xValues, this.inputs, xValues.Length); // place data inputs into RBF net inputs

      double[] hOutputs = new double[numHidden]; // hidden node outputs
      for (int j = 0; j < numHidden; ++j) // each hidden node
      {
        double d = EuclideanDist(inputs, centroids[j], inputs.Length); // could use a 'distSquared' approach
        //Console.WriteLine("\nHidden[" + j + "] distance = " + d.ToString("F4"));
        double r = -1.0 * (d * d) / (2 * widths[j] * widths[j]);
        double g = Math.Exp(r);
        //Console.WriteLine("Hidden[" + j + "] output = " + g.ToString("F4"));
        hOutputs[j] = g;
      }

      double[] tempResults = new double[numOutput];

      for (int k = 0; k < numOutput; ++k)
        for (int j = 0; j < numHidden; ++j)
          tempResults[k] += (hOutputs[j] * hoWeights[j][k]); // accumulate

      for (int k = 0; k < numOutput; ++k)
        tempResults[k] += oBiases[k]; // add biases

      double[] finalOutputs = Softmax(tempResults); // scale the raw output so values sum to 1.0

      //Console.WriteLine("outputs:");
      //Helpers.ShowVector(finalOutputs, 3, finalOutputs.Length, true);
      //Console.ReadLine();

      Array.Copy(finalOutputs, this.outputs, finalOutputs.Length); // transfer computed outputs to RBF net outputs

      double[] returnResult = new double[numOutput]; // also return computed outputs for convenience
      Array.Copy(finalOutputs, returnResult, outputs.Length);
      return returnResult;
    } // ComputeOutputs

    

    private static double[] Softmax(double[] rawOutputs)
    {
      // helper for ComputeOutputs
      // does all output nodes at once so scale doesn't have to be re-computed each time
      // determine max output sum
      double max = rawOutputs[0];
      for (int i = 0; i < rawOutputs.Length; ++i)
        if (rawOutputs[i] > max) max = rawOutputs[i];

      // determine scaling factor -- sum of exp(each val - max)
      double scale = 0.0;
      for (int i = 0; i < rawOutputs.Length; ++i)
        scale += Math.Exp(rawOutputs[i] - max);

      double[] result = new double[rawOutputs.Length];
      for (int i = 0; i < rawOutputs.Length; ++i)
        result[i] = Math.Exp(rawOutputs[i] - max) / scale;

      return result; // now scaled so that all values sum to 1.0
    }

    // -- methods related to training: DoCentroids, DistinctIndices, AvgAbsDist, DoWidths,
    //    DoWeights, Train, Shuffle ----------------------------------------------------------

    private void DoCentroids(double[][] trainData)
    {
      // centroids are representative inputs that are relatively different
      // compute centroids using the x-vaue of training data
      // store into this.centroids
      int numAttempts = trainData.Length;
      int[] goodIndices = new int[numHidden];  // need one centroid for each hidden node
      double maxAvgDistance = double.MinValue; // largest average distance for a set of candidate indices
      for (int i = 0; i < numAttempts; ++i)
      {
        int[] randomIndices = DistinctIndices(numHidden, trainData.Length); // candidate indices
        double sumDists = 0.0; // sum of distances between adjacent candidates (not all candiates)
        for (int j = 0; j < randomIndices.Length - 1; ++j) // adjacent pairs only
        {
          int firstIndex = randomIndices[j];
          int secondIndex = randomIndices[j + 1];
          sumDists += AvgAbsDist(trainData[firstIndex], trainData[secondIndex], numInput); // just the input terms
        }

        double estAvgDist = sumDists / numInput; // estimated average distance for curr candidates
        if (estAvgDist > maxAvgDistance) // curr candidates are far apart
        {
          maxAvgDistance = estAvgDist;
          Array.Copy(randomIndices, goodIndices, randomIndices.Length); // save curr candidates
        }
      } // now try a new set of candidates

      Console.WriteLine("The indices (into training data) of the centroids are:");
      Helpers.ShowVector(goodIndices, goodIndices.Length, true);

      // store copies of x-vales of data pointed to by good indices into this.centroids
      for (int i = 0; i < numHidden; ++i)
      {
        int idx = goodIndices[i]; // idx points to trainData
        for (int j = 0; j < numInput; ++j)
        {
          this.centroids[i][j] = trainData[idx][j]; // make a copy of values
        }
      }
    } // DoCentroids

    private static double AvgAbsDist(double[] v1, double[] v2, int numTerms)
    {
      // average absolute difference distance between two vectors, first numTerms only
      // helper for computing centroids
      if (v1.Length != v2.Length)
        throw new Exception("Vector lengths not equal in AvgAbsDist()");
      double sum = 0.0;
      for (int i = 0; i < numTerms; ++i)
      {
        double delta = Math.Abs(v1[i] - v2[i]);
        sum += delta;
      }
      return sum / numTerms;
    }

    private int[] DistinctIndices(int n, int range)
    {
      // helper for ComputeCentroids()
      // generate n distinct numbers in [0, range-1] using reservoir sampling
      // assumes rnd exists
      int[] result = new int[n];
      for (int i = 0; i < n; ++i)
        result[i] = i;

      for (int t = n; t < range; ++t)
      {
        int m = rnd.Next(0, t + 1);
        if (m < n) result[m] = t;
      }
      return result;
    }

    private void DoWidths(double[][] centroids)
    {
      // compute widths based on centroids, store into this.widths
      // note the centroids parameter could be omitted - the intent is to make relationship clear
      // this version uses a common width which is the average dist between all centroids
      double sumOfDists = 0.0;
      int ct = 0; // could calculate number pairs instead
      for (int i = 0; i < centroids.Length - 1; ++i)
      {
        for (int j = i + 1; j < centroids.Length; ++j)
        {
          double dist = EuclideanDist(centroids[i], centroids[j], centroids[i].Length);
          sumOfDists += dist;
          ++ct;
        }
      }
      double avgDist = sumOfDists / ct;
      double width = avgDist;

      Console.WriteLine("The common width is: " + width.ToString("F4"));

      for (int i = 0; i < this.widths.Length; ++i) // all widths the same
        this.widths[i] = width;
    }

    //private void DoWidths(double[][] centroids)
    //{
    //  // this version uses a common width which is dmax / sqrt(2*numHidden)
    //  double maxDist = double.MinValue;
    //  for (int i = 0; i < centroids.Length - 1; ++i)
    //  {
    //    for (int j = i + 1; j < centroids.Length; ++j)
    //    {
    //      double dist = EuclideanDist(centroids[i], centroids[j], centroids[i].Length);
    //      if (dist > maxDist)
    //        maxDist = dist;
    //    }
    //  }

    //  double width = maxDist / Math.Sqrt(2.0 * numHidden);
    //  Console.WriteLine("The common width is: " + width.ToString("F4"));
    //  for (int i = 0; i < this.widths.Length; ++i) // all widths the same
    //    this.widths[i] = width;
    //}

    private double[] DoWeights(double[][] trainData, int maxIterations)
    {
      // use PSO to find weights and bias values that produce a RBF network
      // that best matches training data
      int numberParticles = trainData.Length / 3;
   
      int Dim = (numHidden * numOutput) + numOutput; // dimensions is num weights + num biases
      double minX = -10.0; // implicitly assumes data has been normalizzed
      double maxX = 10.0;
      double minV = minX;
      double maxV = maxX;
      Particle[] swarm = new Particle[numberParticles];
      double[] bestGlobalPosition = new double[Dim]; // best solution found by any particle in the swarm. implicit initialization to all 0.0
      double smallesttGlobalError = double.MaxValue; // smaller values better

      // initialize swarm
      for (int i = 0; i < swarm.Length; ++i) // initialize each Particle in the swarm
      {
        double[] randomPosition = new double[Dim];
        for (int j = 0; j < randomPosition.Length; ++j)
        {
          double lo = minX;
          double hi = maxX;
          randomPosition[j] = (hi - lo) * rnd.NextDouble() + lo; // 
        }
 
        double err = MeanSquaredError(trainData, randomPosition); // error associated with the random position/solution
        double[] randomVelocity = new double[Dim];

        for (int j = 0; j < randomVelocity.Length; ++j)
        {
          double lo = -1.0 * Math.Abs(maxV - minV);
          double hi = Math.Abs(maxV - minV);
          randomVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
        }
        swarm[i] = new Particle(randomPosition, err, randomVelocity, randomPosition, err);

        // does current Particle have global best position/solution?
        if (swarm[i].error < smallesttGlobalError)
        {
          smallesttGlobalError = swarm[i].error;
          swarm[i].position.CopyTo(bestGlobalPosition, 0);
        }
      } // initialization

      // main PSO algorithm
      // compute new velocity -> compute new position -> check if new best

      double w = 0.729; // inertia weight
      double c1 = 1.49445; // cognitive/local weight
      double c2 = 1.49445; // social/global weight
      double r1, r2; // cognitive and social randomizations

      int[] sequence = new int[numberParticles]; // process particles in random order
      for (int i = 0; i < sequence.Length; ++i)
        sequence[i] = i;

      int iteration = 0;
      while (iteration < maxIterations)
      {
        if (smallesttGlobalError < 0.060) break; // early exit (MSE)

        double[] newVelocity = new double[Dim]; // step 1
        double[] newPosition = new double[Dim]; // step 2
        double newError; // step 3

        Shuffle(sequence); // move particles in random sequence

        for (int pi = 0; pi < swarm.Length; ++pi) // each Particle (index)
        {
          int i = sequence[pi];
          Particle currP = swarm[i]; // for coding convenience

          // 1. compute new velocity
          for (int j = 0; j < currP.velocity.Length; ++j) // each x value of the velocity
          {
            r1 = rnd.NextDouble();
            r2 = rnd.NextDouble();

            // velocity depends on old velocity, best position of parrticle, and 
            // best position of any particle
            newVelocity[j] = (w * currP.velocity[j]) +
              (c1 * r1 * (currP.bestPosition[j] - currP.position[j])) +
              (c2 * r2 * (bestGlobalPosition[j] - currP.position[j]));

            if (newVelocity[j] < minV)
              newVelocity[j] = minV;
            else if (newVelocity[j] > maxV)
              newVelocity[j] = maxV;     // crude way to keep velocity in range
          }

          newVelocity.CopyTo(currP.velocity, 0);

          // 2. use new velocity to compute new position
          for (int j = 0; j < currP.position.Length; ++j)
          {
            newPosition[j] = currP.position[j] + newVelocity[j];  // compute new position
            if (newPosition[j] < minX)
              newPosition[j] = minX;
            else if (newPosition[j] > maxX)
              newPosition[j] = maxX;
          }

          newPosition.CopyTo(currP.position, 0);

          // 3. use new position to compute new error
          // consider cross-entropy error instead of MSE
          newError = MeanSquaredError(trainData, newPosition); // makes next check a bit cleaner
          currP.error = newError;

          if (newError < currP.smallestError) // new particle best?
          {
            newPosition.CopyTo(currP.bestPosition, 0);
            currP.smallestError = newError;
          }

          if (newError < smallesttGlobalError) // new global best?
          {
            newPosition.CopyTo(bestGlobalPosition, 0);
            smallesttGlobalError = newError;
          }

          // consider using weight decay, particle death here

        } // each Particle

        ++iteration;

      } // while (main PSO processing loop)

      //Console.WriteLine("\n\nFinal training MSE = " + smallesttGlobalError.ToString("F4") + "\n\n");

      // copy best weights found into RBF network, and also return them
      this.SetWeights(bestGlobalPosition);
      double[] returnResult = new double[(numHidden * numOutput) + numOutput];
      Array.Copy(bestGlobalPosition, returnResult, bestGlobalPosition.Length);

      Console.WriteLine("The best weights and bias values found are:\n");
      Helpers.ShowVector(bestGlobalPosition, 3, 10, true);
      return returnResult;
    } // DoWeights

    private static void Shuffle(int[] sequence)
    {
      // helper for DoWeights to process particles in random order
      for (int i = 0; i < sequence.Length; ++i)
      {
        int r = rnd.Next(i, sequence.Length);
        int tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    }

    public double[] Train(double[][] trainData, int maxIterations)
    {
      Console.WriteLine("\n1. Computing " + numHidden + " centroids");
      DoCentroids(trainData); // find representative data, store their x-values into this.centroids

      Console.WriteLine("\n2. Computing a common width for each hidden node");
      DoWidths(this.centroids); // measure of how far apart centroids are

      int numWts = (numHidden * numOutput) + numOutput;
      Console.WriteLine("\n3. Determining " + numWts + " weights and bias values using PSO algorithm");
      double[] bestWeights =
        DoWeights(trainData, maxIterations); // use PSO to find weights that best (lowest MSE) weights and biases

      return bestWeights;
    } // Train

    // -- The Euclidean Distance function is used by RBF ComputeOutputs and also DoWidths ----

    private static double EuclideanDist(double[] v1, double[] v2, int numTerms)
    {
      // Euclidean distance between two vectors, first numTerms only
      // helper for computing RBF outputs and computing hidden node widths
      if (v1.Length != v2.Length)
        throw new Exception("Vector lengths not equal in EuclideanDist()");
      double sum = 0.0;
      for (int i = 0; i < numTerms; ++i)
      {
        double delta = (v1[i] - v2[i]) * (v1[i] - v2[i]);
        sum += delta;
      }
      return Math.Sqrt(sum);
    }


  } // class RadialNetwork

  // ===========================================================================

  public class Particle
  {
    public double[] position; // equivalent to x-Values and/or solution
    public double error; // error so smaller is better
    public double[] velocity;

    public double[] bestPosition; // best position found so far by this Particle
    public double smallestError;

    public Particle(double[] position, double error, double[] velocity, double[] bestPosition, double smallestError)
    {
      this.position = new double[position.Length];
      position.CopyTo(this.position, 0);
      this.error = error;
      this.velocity = new double[velocity.Length];
      velocity.CopyTo(this.velocity, 0);
      this.bestPosition = new double[bestPosition.Length];
      bestPosition.CopyTo(this.bestPosition, 0);
      this.smallestError = smallestError;
    }

    //public override string ToString()
    //{
    //  string s = "";
    //  s += "==========================\n";
    //  s += "Position: ";
    //  for (int i = 0; i < this.position.Length; ++i)
    //    s += this.position[i].ToString("F2") + " ";
    //  s += "\n";
    //  s += "Error = " + this.error.ToString("F4") + "\n";
    //  s += "Velocity: ";
    //  for (int i = 0; i < this.velocity.Length; ++i)
    //    s += this.velocity[i].ToString("F2") + " ";
    //  s += "\n";
    //  s += "Best Position: ";
    //  for (int i = 0; i < this.bestPosition.Length; ++i)
    //    s += this.bestPosition[i].ToString("F2") + " ";
    //  s += "\n";
    //  s += "Smallest Error = " + this.smallestError.ToString("F4") + "\n";
    //  s += "==========================\n";
    //  return s;
    //}

  } // class Particle

  // ===========================================================================

  public class Helpers
  {
    public static double[][] MakeMatrix(int rows, int cols)
    {
      double[][] result = new double[rows][];
      for (int i = 0; i < rows; ++i)
        result[i] = new double[cols];
      return result;
    }

    public static void ShowVector(double[] vector, int decimals, int valsPerLine, bool blankLine)
    {
      for (int i = 0; i < vector.Length; ++i)
      {
        if (i > 0 && i % valsPerLine == 0) // max of 12 values per row 
          Console.WriteLine("");
        if (vector[i] >= 0.0) Console.Write(" ");
        Console.Write(vector[i].ToString("F" + decimals) + " "); // n decimals
      }
      if (blankLine) Console.WriteLine("\n");
    }

    public static void ShowVector(int[] vector, int valsPerLine, bool blankLine)
    {
      for (int i = 0; i < vector.Length; ++i)
      {
        if (i > 0 && i % valsPerLine == 0) // max of 12 values per row 
          Console.WriteLine("");
        if (vector[i] >= 0.0) Console.Write(" ");
        Console.Write(vector[i] + " ");
      }
      if (blankLine) Console.WriteLine("\n");
    }

    public static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool lineNumbering, bool showLastLine)
    {
      int ct = 0;
      if (numRows == -1) numRows = int.MaxValue; // if numRows == -1, show all rows
      for (int i = 0; i < matrix.Length && ct < numRows; ++i)
      {
        if (lineNumbering == true)
          Console.Write(i.ToString().PadLeft(3) + ": ");
        for (int j = 0; j < matrix[0].Length; ++j)
        {
          if (matrix[i][j] >= 0.0) Console.Write(" "); // blank space instead of '+' sign
          Console.Write(matrix[i][j].ToString("F" + decimals) + " ");
        }
        Console.WriteLine("");
        ++ct;
      }
      if (showLastLine == true && numRows < matrix.Length)
      {
        Console.WriteLine("      ........\n ");
        int i = matrix.Length - 1;
        Console.Write(i.ToString().PadLeft(3) + ": ");
        for (int j = 0; j < matrix[0].Length; ++j)
        {
          if (matrix[i][j] >= 0.0) Console.Write(" "); // blank space instead of '+' sign
          Console.Write(matrix[i][j].ToString("F" + decimals) + " ");
        }
      }
      Console.WriteLine("");
    }

  } // class Helpers

  // the raw data used in the demo is a 3-item subset of Fisher's Iris data set.
  //

      //double[][] rawData = new double[30][]; // 30-item subset of Iris data set (10 each class)
      //rawData[0] = new double[] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 }; // sepal length, sepal width, petal length, petal width -> 
      //rawData[1] = new double[] { 4.9, 3.0, 1.4, 0.2, 0, 0, 1 }; // Iris setosa = 0 0 1, Iris versicolor = 0 1 0, Iris virginica = 1 0 0
      //rawData[2] = new double[] { 4.7, 3.2, 1.3, 0.2, 0, 0, 1 };
      //rawData[3] = new double[] { 4.6, 3.1, 1.5, 0.2, 0, 0, 1 };
      //rawData[4] = new double[] { 5.0, 3.6, 1.4, 0.2, 0, 0, 1 };
      //rawData[5] = new double[] { 5.4, 3.9, 1.7, 0.4, 0, 0, 1 };
      //rawData[6] = new double[] { 4.6, 3.4, 1.4, 0.3, 0, 0, 1 };
      //rawData[7] = new double[] { 5.0, 3.4, 1.5, 0.2, 0, 0, 1 };
      //rawData[8] = new double[] { 4.4, 2.9, 1.4, 0.2, 0, 0, 1 };
      //rawData[9] = new double[] { 4.9, 3.1, 1.5, 0.1, 0, 0, 1 };

      //rawData[10] = new double[] { 7.0, 3.2, 4.7, 1.4, 0, 1, 0 };
      //rawData[11] = new double[] { 6.4, 3.2, 4.5, 1.5, 0, 1, 0 };
      //rawData[12] = new double[] { 6.9, 3.1, 4.9, 1.5, 0, 1, 0 };
      //rawData[13] = new double[] { 5.5, 2.3, 4.0, 1.3, 0, 1, 0 };
      //rawData[14] = new double[] { 6.5, 2.8, 4.6, 1.5, 0, 1, 0 };
      //rawData[15] = new double[] { 5.7, 2.8, 4.5, 1.3, 0, 1, 0 };
      //rawData[16] = new double[] { 6.3, 3.3, 4.7, 1.6, 0, 1, 0 };
      //rawData[17] = new double[] { 4.9, 2.4, 3.3, 1.0, 0, 1, 0 };
      //rawData[18] = new double[] { 6.6, 2.9, 4.6, 1.3, 0, 1, 0 };
      //rawData[19] = new double[] { 5.2, 2.7, 3.9, 1.4, 0, 1, 0 };

      //rawData[20] = new double[] { 6.3, 3.3, 6.0, 2.5, 1, 0, 0 };
      //rawData[21] = new double[] { 5.8, 2.7, 5.1, 1.9, 1, 0, 0 };
      //rawData[22] = new double[] { 7.1, 3.0, 5.9, 2.1, 1, 0, 0 };
      //rawData[23] = new double[] { 6.3, 2.9, 5.6, 1.8, 1, 0, 0 };
      //rawData[24] = new double[] { 6.5, 3.0, 5.8, 2.2, 1, 0, 0 };
      //rawData[25] = new double[] { 7.6, 3.0, 6.6, 2.1, 1, 0, 0 };
      //rawData[26] = new double[] { 4.9, 2.5, 4.5, 1.7, 1, 0, 0 };
      //rawData[27] = new double[] { 7.3, 2.9, 6.3, 1.8, 1, 0, 0 };
      //rawData[28] = new double[] { 6.7, 2.5, 5.8, 1.8, 1, 0, 0 };
      //rawData[29] = new double[] { 7.2, 3.6, 6.1, 2.5, 1, 0, 0 };

      //double[] originalMeans = new double[] { 5.843, 3.040, 3.863, 1.213 }; // used to normalize original data
      //double[] originalStdDevs = new double[] { 0.948, 0.367, 1.850, 0.776 };

  //

} // ns
