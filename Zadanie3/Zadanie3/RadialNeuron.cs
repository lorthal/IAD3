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
            output = activation.GetResult(Helper.SquaredEuclideanDistance(new [] {inputs[0].output} ,new [] {centroid}), sigma);
        }

        public override void UpdateWeights(double learningRate)
        {
            previousWeight = weight;
            double delta = learningRate * neuronError * output * prevWeightDelta;
            prevWeightDelta = delta;
            weight = previousWeight - delta;
        }
    }
}
