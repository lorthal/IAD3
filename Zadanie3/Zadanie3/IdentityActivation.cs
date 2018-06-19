using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zadanie3
{
    public class IdentityActivation : IActivation
    {
        public double GetResult(params double[] x)
        {
            return x[0];
        }

        public double GetDerivative(params double[] x)
        {
            return 1;
        }
    }
}
