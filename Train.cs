using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Numpy;
using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;

namespace TextMatchPTS 
{
    class Train 
    {

        // This example requires installation of additional NuGet package for 
        // Microsoft.ML.FastTree at
        // https://www.nuget.org/packages/Microsoft.ML.FastTree/
        public static void LightGbm(List<Tuple<bool, float[]>> data)
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness. Setting the seed to a fixed number
            // in this example to make outputs deterministic.
            var mlContext = new MLContext(seed: 0);

            int split = (data.Count / 5) * 4; // 80%
            // Create a list of training data points.
            var dataPoints = PrepareData(data.GetRange(0, split));

            // Convert the list of data points to an IDataView object, which is
            // consumable by ML.NET API.
            Console.WriteLine(split);
            var trainingData = mlContext.Data.LoadFromEnumerable(dataPoints);
            Console.WriteLine(trainingData.ToString());
            // Define trainer options.
            /*
            var options = new FastForestBinaryTrainer.Options
            {
                // Only use 80% of features to reduce over-fitting.
                FeatureFraction = 0.8,
                // Create a simpler model by penalizing usage of new features.
                FeatureFirstUsePenalty = 0.01,
                // Reduce the number of trees to 50.
                NumberOfTrees = 100
            };*/

            // Define the trainer.
            var pipeline = mlContext.BinaryClassification.Trainers
                .LightGbm();

            // Train the model.
            var model = pipeline.Fit(trainingData);

            // Create testing data. Use different random seed to make it different
            // from training data.
            var testData = mlContext.Data
                .LoadFromEnumerable(PrepareData(data.GetRange(split + 1, data.Count - (split + 1) )));

            // Run the model on test data set.
            var transformedTestData = model.Transform(testData);

            // Convert IDataView object to a list.
            var predictions = mlContext.Data
                .CreateEnumerable<Prediction>(transformedTestData,
                reuseRowObject: false).ToList();

            // Print 5 predictions.
            foreach (var p in predictions.Take(5))
                Console.WriteLine($"Label: {p.Label}, " 
                    + $"Prediction: {p.PredictedLabel}");

            // Expected output:
            //   Label: True, Prediction: True
            //   Label: False, Prediction: False
            //   Label: True, Prediction: True
            //   Label: True, Prediction: True
            //   Label: False, Prediction: False
            
            // Evaluate the overall metrics.
            var metrics = mlContext.BinaryClassification
                .Evaluate(transformedTestData); // evaluate non-calibrated

            PrintMetrics(metrics);
            
            // Expected output:
            //   Accuracy: 0.78
            //   AUC: 0.88
            //   F1 Score: 0.79
            //   Negative Precision: 0.83
            //   Negative Recall: 0.74
            //   Positive Precision: 0.74
            //   Positive Recall: 0.84
            //   Log Loss: 0.62
            //   Log Loss Reduction: 37.77
            //   Entropy: 1.00
            //
            //  TEST POSITIVE RATIO:    0.4760 (238.0/(238.0+262.0))
            //  Confusion table
            //            ||======================
            //  PREDICTED || positive | negative | Recall
            //  TRUTH     ||======================
            //   positive ||      185 |       53 | 0.7773
            //   negative ||       83 |      179 | 0.6832
            //            ||======================
            //  Precision ||   0.6903 |   0.7716 |
        }

        private static IEnumerable<DataPoint> PrepareData(List<Tuple<bool, float[]>> data)
        {
            foreach(var relation in data)
            {
                yield return new DataPoint
                {
                    Label = relation.Item1, 
                    // Create random features that are correlated with the label.
                    // For data points with false label, the feature values are
                    // slightly increased by adding a constant.
                    Features = relation.Item2
                };
            }
        }

        // Example with label and 50 feature values. A data set is a collection of
        // such examples.
        private class DataPoint
        {
            public bool Label { get; set; }
            [VectorType(600)]
            public float[] Features { get; set; }
        }

        // Class used to capture predictions.
        private class Prediction
        {
            // Original label.
            public bool Label { get; set; }
            // Predicted label from the trainer.
            public bool PredictedLabel { get; set; }
        }

        // Pretty-print BinaryClassificationMetrics objects.
        private static void PrintMetrics(BinaryClassificationMetrics metrics)
        {
            Console.WriteLine($"Accuracy: {metrics.Accuracy:F2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:F2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:F2}");
            Console.WriteLine($"Negative Precision: " + 
                $"{metrics.NegativePrecision:F2}");

            Console.WriteLine($"Negative Recall: {metrics.NegativeRecall:F2}");
            Console.WriteLine($"Positive Precision: " +
                $"{metrics.PositivePrecision:F2}");

            Console.WriteLine($"Positive Recall: {metrics.PositiveRecall:F2}\n");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
        }
        private static Tuple<NDarray, NDarray> ListToNDarrays(List<Tuple<bool, float[]>> data, int vectorSize){
            var y = new bool[data.Count];
            var x = new float[data.Count, vectorSize];
            for(int i = 0; i < data.Count; i++){
                for(int j = 0; j < data[i].Item2.Length; j++){
                    x[i, j] = data[i].Item2[j];
                }
                y[i] = data[i].Item1;
            }
            NDarray f = np.array(x);
            NDarray l = np.array(y);
            return Tuple.Create(l, f);
        }
        public static void MultilayerPerceptron1DInputLg(List<Tuple<bool, float[]>> train, List<Tuple<bool, float[]>> test)
        {
            int vectorSize = train[0].Item2.Length;
            //Load train data
            var nTrain = ListToNDarrays(train, vectorSize);
            var nTest = ListToNDarrays(test, vectorSize);

            //Build sequential model
            var model = new Sequential();
            model.Add(new Dense(32, activation: "relu", input_shape: new Shape(vectorSize)));
            model.Add(new Dense(64, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(128, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(128, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(256, activation: "relu"));
            model.Add(new Dense(1, activation: "sigmoid"));

            //Compile and train
            model.Compile(optimizer:"adam", loss:"binary_crossentropy", metrics: new string[] { "accuracy" });
            model.Fit(
                nTrain.Item2, 
                nTrain.Item1, 
                batch_size: 32, 
                epochs: 20, 
                verbose: 1,
                validation_data: new NDarray[] { nTest.Item2, nTest.Item1 } );
            //Save model and weights
            string json = model.ToJson();
            File.WriteAllText("./models/lg_model.json", json);
            model.SaveWeight("./models/lg_model.h5");

        }
        public static void MultilayerPerceptron1DInputSm(List<Tuple<bool, float[]>> train, List<Tuple<bool, float[]>> test)
        {
            int vectorSize = train[0].Item2.Length;
            //Load train data
            var nTrain = ListToNDarrays(train, vectorSize);
            var nTest = ListToNDarrays(test, vectorSize);

            //Build sequential model
            var model = new Sequential();
            model.Add(new Dense(8, activation: "relu", input_shape: new Shape(vectorSize)));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(8, activation: "relu"));
            model.Add(new Dropout(0.5));
            model.Add(new Dense(1, activation: "sigmoid"));

            //Compile and train
            //model.Compile(optimizer:"adam", loss:"sparse_categorical_crossentropy", metrics: new string[] { "accuracy" });
            model.Compile(optimizer:"adam", loss:"binary_crossentropy", metrics: new string[] { "accuracy" });
            model.Fit(
                nTrain.Item2,
                nTrain.Item1,
                batch_size: 8, 
                epochs: 50, 
                verbose: 1,
                validation_data: new NDarray[] { nTest.Item2, nTest.Item1 } );

            //Save model and weights
            string json = model.ToJson();
            File.WriteAllText("./models/sm_model.json", json);
            model.SaveWeight("./models/sm_model.h5");

        }
        public static void MultilayerPerceptron2DInput(List<Tuple<bool, float[,], float[,]>> train, List<Tuple<bool, float[,], float[,]>> test){
            int vectorSize = train[0].Item2.Length;
            //Load train data
            //var nTrain = listToNDarrays(train, vectorSize);
            //var nTest = listToNDarrays(test, vectorSize);


        }
    }
}