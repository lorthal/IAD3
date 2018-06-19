using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zadanie3
{
    public class RadialNeuron : Neuron
    {
        public double centroid;
        public double sigma;

        public RadialNeuron(IActivation activation) : base(activation)
        {

        }

        public override void ComputeOutput()
        {
            for (int i = 0; i < inputs[0].output.Count; i++)
            {
                double d = Helper.SquaredEuclideanDistance(new[] { inputs[0].output[i] }, new[] { centroid });
                if (output.Count < inputs[0].output.Count)
                {
                    output.Add(activation.GetResult(d, sigma));                
                }
                else
                {
                    output[i] = activation.GetResult(d, sigma);
                }
            }
        }

        //public override void UpdateWeights(double learningRate, Neuron nextNeuron)
        //{
        //    previousWeight = weight;
        //    double delta = 0;

        //        delta += learningRate * nextNeuron.neuronError * output.Average() + prevWeightDelta;
        //    delta /= output.Count;
        //    prevWeightDelta = delta;
        //    weight = previousWeight - delta;
        //}
    }
}
