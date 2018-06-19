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
        public double output;
        public List<Neuron> inputs;

        public double neuronError;
        private double neighbourErrorSum;
        protected IActivation activation;
        protected double prevWeightDelta;

        public Neuron(IActivation activation)
        {
            this.activation = activation;
            inputs = new List<Neuron>();
        }

        public virtual void ComputeOutput() { }

        public virtual void ComputeNeuronError(double expectedOutput)
        {
            double sum = expectedOutput * expectedOutput - 2 * expectedOutput * output +
                         output * output;
            neuronError = sum / 2;
        }
        public virtual void UpdateWeights(double learningRate) { }
        public virtual double GetInputSum()
        {
            double sum = 0;

            foreach (var neuron in inputs)
            {
                sum += neuron.output * neuron.weight;
            }
            return sum;
        }
    }
}
