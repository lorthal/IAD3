using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zadanie3
{
    public class IdentityNeuron : Neuron
    {

        public IdentityNeuron(IActivation activation) : base(activation)
        {
        }

        public override void ComputeOutput()
        {
            double[] list = GetInputSum();
            for (int i = 0; i < list.Length; i++)
            {
                if (output.Count < inputs[0].output.Count)
                {
                    output.Add(activation.GetResult(list[i]));
                }
                else
                {
                    output[i] = activation.GetResult(list[i]);
                }
            }
        }

        public override void UpdateWeights(double learningRate)
        {
            foreach (var neuron1 in inputs)
            {
                var neuron = (RadialNeuron)neuron1;
                neuron.previousWeight = neuron.weight;
                double delta = 0;
                for (int i = 0; i < neuron.output.Count; i++)
                {
                    //double d = Math.Exp(Helper.SquaredEuclideanDistance(new[] { output[i] }, new[] { neuron.centroid }) /
                    //                    (2 * neuron.sigma * neuron.sigma)) / (Math.Sqrt(2 * Math.PI) * neuron.sigma);
                    double e = Math.Exp(((neuron.inputs[0].output[i] - neuron.centroid) * (neuron.inputs[0].output[i] - neuron.centroid)) /
                                        (2 * neuron.sigma * neuron.sigma));
                    double d = (Math.Sqrt(2 / Math.PI) * e * ((neuron.previousWeight * e) / (Math.Sqrt(2 * Math.PI) * neuron.sigma) - neuron.output[i])) / neuron.sigma;

                    delta += neuronError * d;
                }

                delta = learningRate * delta + prevWeightDelta;
                neuron.prevWeightDelta = delta;
                neuron.weight = neuron.previousWeight - delta;
            }
        }
    }
}
