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
        public double weight;
        public double previousWeight;
        public List<double> output;
        public List<Neuron> inputs;

        public double neuronError;
        private double neighbourErrorSum;
        protected IActivation activation;
        public double prevWeightDelta;

        public Neuron(IActivation activation)
        {
            this.activation = activation;
            inputs = new List<Neuron>();
            output = new List<double>();
            weight = Helper.random.NextDouble(-1, 1);
        }

        public virtual void ComputeOutput() { }

        public virtual void ComputeNeuronError(double[] expectedOutput)
        {
            double sum = 0;

            for (int i = 0; i < expectedOutput.Length; i++)
            {
                sum += expectedOutput[i] * expectedOutput[i] - 2 * expectedOutput[i] * output[i] +
                       output[i] * output[i];
            }
            
            neuronError = sum / (2 * expectedOutput.Length);
        }
        public virtual void UpdateWeights(double learningRate) { }
        public virtual double[] GetInputSum()
        {
            List<double> sums = new List<double>();

            for (int k = 0; k < inputs[0].output.Count; k++)
            {
                sums.Add(0);
            }

            for (int i = 0; i < sums.Count; i++)
            {
                sums[i] = 0;

                foreach (var neuron in inputs)
                {
                    sums[i] += neuron.output[i] * neuron.weight;
                }
            }
            
            return sums.ToArray();
        }
    }
}
