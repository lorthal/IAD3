using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zadanie3
{
    public class NeuralNet
    {
        public static int InputlayerLength;
        public static int HiddenLayerLength;
        public static int OutputLayerLength;
        private bool useBias;

        private Neuron[][] neurons;
        private double[][] hiddenWeights;
        private double[][] outputWeights;
        private double[] errors;
        private double[] inputX;
        private double[] inputY;
        private double[] outputY;
        private double learningRate;
        private int epochs;
        private string filename;
        private static List<List<double>> valuesFromFile;
        private static List<List<double>> values;

        public NeuralNet(string filename, int epochs, int inputLayerLength, int hiddenLayerLength,
            int outputLayerLength, double learningRate, bool useBias)
        {
            this.useBias = useBias;
            this.filename = filename;
            this.epochs = epochs;
            this.learningRate = learningRate;
            InputlayerLength = inputLayerLength;
            HiddenLayerLength = hiddenLayerLength;
            OutputLayerLength = outputLayerLength;
            neurons = new Neuron[3][];
            neurons[0] = new Neuron[InputlayerLength];
            neurons[1] = new Neuron[HiddenLayerLength];
            neurons[2] = new Neuron[OutputLayerLength];

            errors = new double[epochs];

            valuesFromFile = Helper.ReadData(Program.resPath);

            List<double> centers = valuesFromFile.Select(l => l[0]).ToList();

            KMeans kMeans = new KMeans(HiddenLayerLength, valuesFromFile.Count);
            kMeans.SetCentroids(valuesFromFile);
            //kMeans.Run(valuesFromFile);

            double max = centers.Max();
            double min = centers.Min();
            double sigma = (max - min) / Math.Sqrt(2 * kMeans.centroids.Count);

            for (int i = 0; i < InputlayerLength; i++)
            {
                neurons[0][i] = new Neuron(learningRate, 1, 0, i, useBias);
            }

            for (int i = 0; i < HiddenLayerLength; i++)
            {
                neurons[1][i] = new Neuron(learningRate, InputlayerLength, 1, i, useBias, centers[i], sigma);
            }

            for (int i = 0; i < OutputLayerLength; i++)
            {
                neurons[2][i] = new Neuron(learningRate, HiddenLayerLength, 2, i, useBias);
            }

            for (int i = 0; i < HiddenLayerLength; i++)
            {
                neurons[1][i].SetNeuronsAbove(neurons[2]);
            }
        }

        public void Learn()
        {
            values = Helper.ReadData(Program.resPath);
            inputX = new double[values.Count];
            inputY = new double[values.Count];
            outputY = new double[values.Count];

            double[] errorSinlgePass = new double[values.Count];

            for (int i = 0; i < epochs; i++)
            {
                double errorSum = 0;
                values.Shuffle();
                for (int j = 0; j < values.Count; j++)
                {
                    ForwardPropagation(j);

                    errorSinlgePass[j] = MeanSquaredError(BackPropagation(i));

                    //Console.WriteLine("Error: {0}", errorSinlgePass[j]);

                    if (i == epochs - 1)
                    {
                        inputX[j] = values[j][0];
                        inputY[j] = values[j][1];
                        outputY[j] = neurons[2][0].Output;
                    }
                }
                Console.WriteLine("Epoch: {0}, Error: {1}", i + 1, MeanSquaredError(errorSinlgePass));
            }
        }

        private void ForwardPropagation(int number)
        {
            double[] tmp = new double[InputlayerLength];
            double[] tmp2 = new double[HiddenLayerLength];

            for (int i = 0; i < InputlayerLength; i++)
            {
                neurons[0][i].Inputs[0] = values[number][0];
                neurons[0][i].ExpectedOutput = values[number][1];
                neurons[0][i].CalculateOutput(0);
                tmp[i] = neurons[0][i].Output;
            }

            for (int i = 0; i < HiddenLayerLength; i++)
            {
                neurons[1][i].Inputs = tmp;
                neurons[1][i].ExpectedOutput = values[number][1];
                neurons[1][i].CalculateOutput(1);
                tmp2[i] = neurons[1][i].Output;
            }

            for (int i = 0; i < OutputLayerLength; i++)
            {
                neurons[2][i].Inputs = tmp2;
                neurons[2][i].ExpectedOutput = values[number][1];
                neurons[2][i].CalculateOutput(2);
            }
        }

        private double[] BackPropagation(int epoch)
        {
            double[] errors = new double[OutputLayerLength];

            for (int i = 0; i < OutputLayerLength; i++)
            {
                neurons[2][i].CalculateError(2);
                errors[i] = neurons[2][i].Error;
            }

            for (int i = 0; i < OutputLayerLength; i++)
            {
                neurons[2][i].UpdateWeights();
            }

            return errors;
        }

        private double MeanSquaredError(double[] tab)
        {
            double sum = 0;
            for (int i = 0; i < tab.Length; i++)
            {
                sum += tab[i] * tab[i] * 0.5;
            }
            sum = sum / tab.Length;
            return sum;
        }
    }
}
