using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zadanie3
{
    class Program
    {
        private const string resPath = "D:\\Semestr 6\\IAD\\Zad3Res\\";

        static void Main(string[] args)
        {
            int n = 11;
            int epochs = 2000;
            double[] inputs, expectedOutputs;

            Helper.ReadTrainingSet(resPath + "approximation_train_2.txt", out inputs, out expectedOutputs);
            NeuralNet net = new NeuralNet(n, inputs);

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                Console.WriteLine("Epoch: {0}", epoch);

                net.Train(expectedOutputs);
                Console.WriteLine("\nTotal Error: {0}\n", net.TotalError);
            }
            Console.ReadKey();
        }
    }
}
