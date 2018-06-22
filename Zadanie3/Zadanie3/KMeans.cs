using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace Zadanie3
{
    public class KMeans
    {
        public List<List<double>> centroids;
        public List<double> errors;

        private double[][] groups;
        private int numNeurons;
        private List<double> distances;

        private double error = 0;

        public KMeans(int numNeurons, int numInputs)
        {
            this.numNeurons = numNeurons;
            this.centroids = new List<List<double>>();
            this.groups = new double[numInputs][];

            for (int i = 0; i < groups.Length; i++)
            {
                groups[i] = new double[3];
            }

            this.distances = new List<double>();
            errors = new List<double>();
        }

        public void SetCentroids(List<List<double>> data)
        {
            int random;
            for (int i = 0; i < numNeurons; i++)
            {
                random = Helper.random.Next(0, data.Count);
                centroids.Add(data[random]);
            }
        }

        public void SetDefaultCentroids(int numNeurons)
        {
            List<double> list = new List<double>();
            for (int i = 0; i < numNeurons; i++)
            {
                list.Clear();
                list.Add(0);
                list.Add(0);
                centroids.Add(list);
            }
        }

        public void SetGroups(List<List<double>> data)
        {
            double x2 = 0, x1 = 0, y2 = 0, y1 = 0;
            for (int i = 0; i < data.Count; i++)
            {
                for (int j = 0; j < numNeurons; j++)
                {
                    x2 = data[i][0];
                    x1 = centroids[j][0];
                    y2 = data[i][1];
                    y1 = centroids[j][1];
                    distances.Add(Helper.EuclideanDistance(new[] {x2, x1},
                        new[] {y2, y1}));
                }
                groups[i][0] = distances.Min();
                groups[i][1] = x2;
                groups[i][2] = y2;

                error = error / distances.Count;
                distances.Clear();
            }
        }

        public double[][] Run(List<List<double>> data)
        {
            while (ifMoving)
            {
                SetGroups(data);
                UpdateWeights();

                for (int i = 0; i < groups.Length; i++)
                {
                    double dist = Helper.EuclideanDistance(new[] {groups[i][1], centroids[(int) groups[i][0]][0]},
                        new[] {groups[i][2], centroids[(int) groups[i][0]][1]});
                    error += dist * dist;
                }
                error = error / groups.Length;
                errors.Add(error);
                error = 0;
            }
            return groups;
        }

        private bool ifMoving = true;

        public void UpdateWeights()
        {
            double newX = 0, newY = 0;
            int counter = 0;
            for (int i = 0; i < centroids.Count; i++)
            {
                for (int j = 0; j < groups.Length; j++)
                {
                    if (Math.Abs(groups[j][0] - centroids[i][0]) < 0.0001)
                    {
                        newX += groups[j][1];
                        newY += groups[j][2];
                        counter++;
                    }
                }

                newX = newX / counter;
                newY = newY / counter;

                if (Math.Abs(centroids[i][0] - newX) < 0.0001 && Math.Abs(centroids[i][1] - newY) < 0.0001)
                {
                    ifMoving = false;
                }
                if (!newX.Equals(double.NaN) && !newY.Equals(double.NaN))
                {
                    centroids[i].Clear();
                    centroids[i].Add(newX);
                    centroids[i].Add(newY);
                }
                counter = 0;
                newX = 0.0;
                newY = 0.0;
            }
        }

        public void CalculateError()
        {
            int lenght;
            for (int i = 0; i < centroids.Count; i++)
            {
                lenght = 0;
                for (int j = 0; j < groups.Length; j++)
                {
                    if (Math.Abs(groups[j][0] - centroids[i][0]) < 0.0001)
                    {
                        lenght++;
                    }
                }
                for (int k = 0; k < lenght; k++)
                {
                    double dist = Helper.EuclideanDistance(new[] {groups[k][1], centroids[i][0]},
                        new[] {groups[k][2], centroids[i][1]});
                    error += dist * dist;
                }
                error = error / lenght;
                errors.Add(error);
                error = 0;
            }
        }
    }
}
