using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;

namespace Zadanie3
{
    public class Neuron
    {
        public double[] Weights { get; private set; }
        private double[] lastWeightChanges;
        public double[] Inputs { get; set; }
        private Neuron[] neuronsAbove;

        public double Error { get; private set; }
        private double bias;
        private double lastBiasChange = 0;
        public double Output { get; private set; }
        public double ExpectedOutput { get; set; }
        private double learningRate;
        private int numOfLayer;
        private int numInLayer;
        private double center;
        private double sigma;

        private bool radial;
        private bool useBias;

        private double momentum = 0.7;

        public Neuron(double learningRate, int numInputs, int numOfLayer, int numInLayer, bool useBias)
        {
            this.useBias = useBias;
            this.learningRate = learningRate;
            this.numOfLayer = numOfLayer;
            this.numInLayer = numInLayer;
            this.Inputs = new double[numInputs];
            this.Weights = new double[numInputs];
            this.lastWeightChanges = new double[numInputs];

            for (int i = 0; i < lastWeightChanges.Length; i++)
            {
                lastWeightChanges[i] = 0;
            }

            double range01 = Math.Sqrt(2.0 / NeuralNet.InputlayerLength);
            double range12 = Math.Sqrt(2.0 / NeuralNet.HiddenLayerLength);

            for (int i = 0; i < numInputs; i++)
            {
                if (numOfLayer == 1)
                {
                    Weights[i] = Helper.random.NextDouble(-range01, range01);
                }
                else if (numOfLayer == 2)
                {
                    Weights[i] = Helper.random.NextDouble(-range12, range12);
                }
            }

            if (useBias)
            {
                this.bias = Helper.random.NextDouble(-1.0, 1.0);
            }
        }

        public Neuron(double learningRate, int numberOfInputs, int numberOfLayer, int numberInLayer, bool useBias, double center, double sigma)
            : this(learningRate, numberOfInputs, numberOfLayer, numberInLayer, useBias)
        {
            this.radial = true;
            this.center = center;
            this.sigma = sigma;
        }

        public void SetNeuronsAbove(Neuron[] neurons)
        {
            this.neuronsAbove = neurons;
        }

        public void CalculateOutput(int numOfLayer)
        {
            Output = 0;
            switch (numOfLayer)
            {
                case 0:
                    Output = Inputs[0];
                    break;
                case 1:
                    for (int i = 0; i < Inputs.Length; i++)
                    {
                        Output += Inputs[i] * Weights[i];
                    }

                    if (useBias)
                    {
                        Output += bias;
                    }

                    Output = radial
                        ? Math.Exp((-((Output - center) * (Output - 0.0))) / (2.0 * sigma * sigma))
                        : 1.0 / (1.0 + Math.Exp(-Output));
                    break;
                case 2:
                    for (int i = 0; i < Inputs.Length; i++)
                    {
                        Output += Inputs[i] * Weights[i];
                    }

                    if (useBias)
                    {
                        Output += bias;
                    }
                    break;
                default:
                    break;
            }
        }

        public void CalculateError(int numOfLayer)
        {
            Error = 0.0;

            switch (numOfLayer)
            {
                case 0:
                    break;
                case 1:
                    double errorSum = 0.0;
                    foreach (var neuron in neuronsAbove)
                    {
                        errorSum += neuron.Weights[this.numOfLayer] * neuron.Error;
                    }
                    Error = Derivative(Output) * errorSum;
                    break;
                case 2:
                    Error = -Output + ExpectedOutput;
                    break;
                default:
                    break;

            }
        }

        public double Derivative(double data)
        {
            return data * (1 - data);
        }

        public void UpdateWeights()
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                double change = learningRate * Error * Inputs[i] + lastWeightChanges[i] * momentum;

                Weights[i] = Weights[i] + change;
                lastWeightChanges[i] = change;
            }

            if (useBias)
            {
                double change = learningRate * Error + lastBiasChange * momentum;
                bias = bias + change;
                lastBiasChange = change;
            }
        }
    }
}
