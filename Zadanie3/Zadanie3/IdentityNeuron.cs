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
            output = activation.GetResult(GetInputSum());
        }
    }
}
