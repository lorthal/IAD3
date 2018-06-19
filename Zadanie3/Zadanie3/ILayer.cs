using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zadanie3
{
    public class ILayer
    {
        private double numNeurons;
        private IActivation activation;
        public List<Neuron> neurons { get; private set; }

        public ILayer(int numNeurons, IActivation activation)
        {
            this.activation = activation;
            this.numNeurons = numNeurons;

            neurons = new List<Neuron>();

            InitNeurons();
        }

        public void ComputeNeuronErrors(double expectedOutput)
        {
            foreach (var neuron in neurons)
            {
                neuron.ComputeNeuronError(expectedOutput);
            }
        }

        private void InitNeurons()
        {
            for (int i = 0; i < numNeurons; i++)
            {
                if (activation is RadialActivation)
                {
                    neurons.Add(new RadialNeuron(activation));
                }
                else
                {
                    neurons.Add(new IdentityNeuron(activation));
                }
            }
        }
    }
}
