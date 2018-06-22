using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zadanie3
{
    class Program
    {
        public const string resPath = "D:\\Semestr 6\\IAD\\Zad3Res\\approximation_train_1.txt";

        static void Main(string[] args)
        {
            int n = 11;
            int epochs = 2000;

            NeuralNet net = new NeuralNet(resPath, epochs, 1, 10, 1, 0.003, true);

            net.Learn();

            Console.ReadKey();
        }
    }
}
