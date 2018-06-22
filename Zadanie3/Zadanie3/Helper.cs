using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Zadanie3
{
    public static class Helper
    {
        public static Random random = new Random();

        public static void ReadTrainingSet(string path, out double[] inputs, out double[] expectedOutputs)
        {
            List<double> inputsList = new List<double>();
            List<double> eOutputsList = new List<double>();
            using (StreamReader sr = File.OpenText(path))
            {
                string line;
                while (!sr.EndOfStream)
                {
                    line = sr.ReadLine();
                    line = line.Replace('.', ',');
                    string[] split = line.Split(' ');
                    inputsList.Add(double.Parse(split[0]));
                    eOutputsList.Add(double.Parse(split[1]));
                }
            }
            inputs = inputsList.ToArray();
            expectedOutputs = eOutputsList.ToArray();
        }

        public static List<List<double>> ReadInputs(string path)
        {
            List<List<double>> inputs = new List<List<double>>();
            using (StreamReader sr = File.OpenText(path))
            {
                string line;
                while (!sr.EndOfStream)
                {
                    line = sr.ReadLine();
                    line = line.Replace('.', ',');
                    string[] split = line.Split(' ');
                    inputs.Add(new List<double>() { double.Parse(split[0]) });
                }
            }
            return inputs;
        }

        public static List<List<double>> ReadData(string path)
        {
            List<List<double>> data = new List<List<double>>();
            using (StreamReader sr = File.OpenText(path))
            {
                string line;
                while (!sr.EndOfStream)
                {
                    line = sr.ReadLine();
                    line = line.Replace('.', ',');
                    string[] split = line.Split(' ');
                    data.Add(new List<double>() { double.Parse(split[0]), double.Parse(split[1]) });
                }
            }
            return data;
        }

        public static double EuclideanDistance(double[] a, double[] b)
        {
            var length = a.Length;
            if (length != b.Length)
            {
                throw new ArgumentException("Lenghts of weights arrays are differ.");
            }

            return Math.Sqrt(SquaredEuclideanDistance(a, b));
        }

        public static double SquaredEuclideanDistance(double[] a, double[] b)
        {
            var length = a.Length;
            if (length != b.Length)
            {
                throw new ArgumentException("Lenghts of weights arrays are differ.");
            }

            double distance = 0;
            for (int i = 0; i < length; i++)
            {
                var diff = a[i] - b[i];
                distance += diff * diff;
            }
            return distance;
        }

        public static double FindClosest(double point, double[] others)
        {
            double closest = others[0];

            for (int i = 1; i < others.Length; i++)
            {
                closest = SquaredEuclideanDistance(new[] { others[i] }, new[] { point }) >
                          SquaredEuclideanDistance(new[] { closest }, new[] { point })
                    ? others[i]
                    : closest;
            }

            return closest;
        }
    }

    public static class Extentions
    {
        public static double NextDouble(this Random r, double minimum, double maximum)
        {
            return r.NextDouble() * (maximum - minimum) + minimum;
        }

        private static Random rng = Helper.random;

        public static void Shuffle<T>(this IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }

        public static string ToString<T>(this IList<T> list)
        {
            int n = list.Count;
            StringBuilder sb= new StringBuilder();
            while (n > 1)
            {
                n--;
                sb.AppendLine(list[n].ToString());
            }
            return sb.ToString();
        }
    }
}
