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
        private IdentityNeuron outputNeuron;
        private double[] inputs;
        private double learningRate;

        private double totalError = 0;

        public double TotalError
        {
            get => totalError;
            set => totalError = value;
        }

        public NeuralNet(int numNeuronsRadialLayer, double[] inputs, double learningRate = 0.6)
        {
            layer = new ILayer(numNeuronsRadialLayer, new RadialActivation());
            outputNeuron = new IdentityNeuron(new IdentityActivation());
            this.inputs = inputs;
            this.learningRate = learningRate;
        }

        public void Train(double[] expectedOuput)
        {
            InitNeurons(inputs);
            Predict();
            ComputeOutputError(expectedOuput);

            //Display(outputNeuron.neuronError);

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

        public void InitNeurons(double[] inputs)
        {
            foreach (var neuron1 in layer.neurons)
            {
                var neuron = neuron1 as RadialNeuron;
                if (neuron.inputs.Count == 0)
                    neuron.inputs.Add(new Neuron(null) { output = inputs.ToList() });

                //int i = Helper.random.Next(0, inputs.Length);
                neuron.centroid = Helper.random.NextDouble(inputs.Min(), inputs.Max());

                if (outputNeuron.inputs.Count < layer.neurons.Count)
                {
                    outputNeuron.inputs.Add(neuron);
                }
            }

            for (int i = 0; i < layer.neurons.Count; i++)
            {
                var neuron = layer.neurons[i] as RadialNeuron;
                //double[] list = inputs.ToList().FindAll(n => Math.Abs(n - neuron.centroid) > 0.00001).ToArray();

                double sum = 0;

                for (int j = 0; j < inputs.Length; j++)
                {
                    sum += Helper.EuclideanDistance(new[] { neuron.centroid }, new[] { inputs[j] });
                }

                neuron.sigma = sum;
            }
        }

        private void ComputeOutputError(double[] expectedOutputs)
        {
            outputNeuron.ComputeNeuronError(expectedOutputs);

            totalError = outputNeuron.neuronError;
        }

        private void BackPropagation(double[] expectedOutput)
        {
            layer.ComputeNeuronErrors(expectedOutput);

            outputNeuron.UpdateWeights(learningRate);
        }

        private void Display(double error)
        {
            Console.WriteLine("Error {0}", error);
        }
    }
}
