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
        private ILayer layer;
        private Random random;
        private IdentityNeuron outputNeuron;
        private double[] inputs;
        private double learningRate;

        public NeuralNet(int numNeuronsRadialLayer, double[] inputs, double learningRate = 0.6)
        {
            layer = new ILayer(numNeuronsRadialLayer, new RadialActivation());
            outputNeuron = new IdentityNeuron(new IdentityActivation());
            random = new Random();
            this.inputs = inputs;
            this.learningRate = learningRate;
        }

        public void Train(double input, double expectedOuput)
        {
            InitNeurons(input);
            Predict();
            ComputeOutputError(expectedOuput);

            Display(input, outputNeuron.output, expectedOuput, outputNeuron.neuronError);

            BackPropagation(expectedOuput);
        }

        public void Predict()
        {
            foreach (var neuron in layer.neurons)
            {
                neuron.ComputeOutput();
            }
            outputNeuron.ComputeOutput();
        }

        public void InitNeurons(double input)
        {
            foreach (var neuron1 in layer.neurons)
            {
                var neuron = neuron1 as RadialNeuron;
                neuron.inputs.Clear();
                neuron.inputs.Add(new Neuron(null) { output = input });

                int i = random.Next(0, inputs.Length);
                neuron.centroid = inputs[i];
                neuron.weight = random.NextDouble(-1, 1);

                if (!outputNeuron.inputs.Contains(neuron))
                {
                    outputNeuron.inputs.Add(neuron);
                }
            }

            foreach (var neuron1 in layer.neurons)
            {
                var neuron = neuron1 as RadialNeuron;

                neuron.sigma = Helper.EuclideanDistance(new[] { neuron.centroid },
                                   new[]
                                   {
                                       Helper.FindClosest(neuron.centroid,
                                           layer.neurons.ToList().FindAll(n => n != neuron)
                                               .Select(n => ((RadialNeuron) n).centroid).ToArray())
                                   });
            }
        }

        private void ComputeOutputError(double expectedOutput)
        {
            outputNeuron.ComputeNeuronError(expectedOutput);
        }

        private void BackPropagation(double expectedOutput)
        {
            layer.ComputeNeuronErrors(expectedOutput);

            foreach (var neuron in layer.neurons)
            {
                neuron.UpdateWeights(learningRate);
            }
        }

        private void Display(double input, double output, double expected, double error)
        {
            Console.WriteLine("Input: {0}\nOutput: {1}\nExpectedOutput: {2}\nError: {3}", input, output, expected, error);
        }
    }
}
