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
            int n = 10;
            int epochs = 100;
            double[] inputs, expectedOutputs;

            Helper.ReadTrainingSet(resPath + "approximation_train_1.txt", out inputs, out expectedOutputs);
            NeuralNet net = new NeuralNet(n, inputs);

            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    Console.WriteLine("Epoch: {0}", epoch);
                    net.Train(inputs[i], expectedOutputs[i]);
                    Console.WriteLine();
                }
            }
            Console.ReadKey();
        }
    }
}
