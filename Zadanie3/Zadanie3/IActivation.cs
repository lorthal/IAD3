using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zadanie3
{
    public interface IActivation
    {
        double GetResult(params double[] x);
        double GetDerivative(params double[] x);
    }
}
