using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zadanie3
{
    public class RadialActivation : IActivation
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="p">
        /// 0 - dSrq
        /// 1 - sigma
        /// </param>
        /// <returns></returns>
        public double GetResult(params double[] p)
        {
            double dSrq = p[0];
            double sigma = p[1];

            return Math.Exp(-(dSrq / (2 * sigma * sigma))) / (Math.Sqrt(2 * Math.PI) * sigma);
        }

        public double GetDerivative(params double[] x)
        {
            throw new NotImplementedException();
        }
    }
}
