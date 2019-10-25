using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;
using Proxem.Word2Vec;
using NumSharp;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace text_match_pts
{

     
    class Program
    {
        const int MODEL_VECTOR_SIZE = 300;

        private class QuoraQuestion
        {
            public string Question1 { get; set; }
            public string Question2 { get; set; }
            public bool Related { get; set; }
        }
       

        static List<QuoraQuestion> readQuestions(int limit){
            string[] lines = File.ReadAllLines("data/quora_duplicate_questions.tsv");
            var questions = new List<QuoraQuestion>();
            int i = 0;
            int nRelated = 0;
            int nUnrelated = 0;
            while(nRelated + nUnrelated < limit){
                var question = new QuoraQuestion();
                try{
                    var l = lines[i].Split('\t');
                    question.Question1 = l[3];
                    question.Question2 = l[4];
                    question.Related = l[5].Equals("1");

                    if(nRelated < (limit / 2) && question.Related){
                        questions.Add(question);
                        Console.WriteLine("adding related");
                        nRelated++;
                    }
                    if(nUnrelated < (limit / 2) && !question.Related){
                        questions.Add(question);
                        Console.WriteLine("adding unrelated");
                        nUnrelated++;
                    }
                }catch(System.Exception){
                    Console.WriteLine("problem at line: " + i);

                }
                i++;
            }
            Console.WriteLine("Questions length: " + questions.Count);
            return questions.OrderBy(a => Guid.NewGuid()).ToList();
        }
       
        static void Main(string[] args)
        {
            //var word2VecModel = Word2Vec.LoadText("data/sv.txt", true, false, "");
            var word2VecModel = Word2Vec.LoadBinary("data/GoogleNews-vectors-negative300.bin", true, false, "", Encoding.UTF8 /* Encoding.GetEncoding("ISO-8859-1") */);
            var questions = readQuestions(50000);
            var docRelations = new List<Tuple<bool, float[]>>();
            foreach(QuoraQuestion question in questions){
                try{
                    NDArray v1 = Vectorize(question.Question1, word2VecModel);
                    //Console.WriteLine(question.Question2.Length);
                    NDArray v2 = Vectorize(question.Question2, word2VecModel);
                    NDArray diff = np.subtract(v1, v2);
                    
                    float[] similarity = Array.ConvertAll(
                        diff.ToArray<double>(),
                        s => (float) s);

                    var docRelation = Tuple.Create(question.Related, similarity);
                    docRelations.Add(docRelation);
                }catch(System.Exception){

                }
            }
            Console.WriteLine("Done reading questions");
            
            TrainTree(docRelations);
            
            //Console.WriteLine("Running");
            //var model = Word2Vec.LoadBinary("models/sv.bin", true, fale, "", Encoding.UTF8 /* Encoding.GetEncoding("ISO-8859-1") */);
            /*
            var word2VecModel = Word2Vec.LoadText("data/sv.txt", true, false, "");
            
            string[] docs = File.ReadAllLines("data/documents.txt");
            var docRelations = new List<Tuple<bool, float[]>>();
            int i = 0;
            while(i + 2 < docs.Length){
                string first = docs[i];
                string second = docs[i + 1];
                Console.WriteLine(first);
                Console.WriteLine(second);

                bool label = bool.Parse(docs[i + 2]);
                
                NDArray v1 = Vectorize(first, word2VecModel);
                NDArray v2 = Vectorize(second, word2VecModel);

                NDArray diff = np.subtract(v1, v2);
                
                float[] similarity = Array.ConvertAll(
                    diff.ToArray<double>(), 
                    s => (float) s);

                var docRelation = Tuple.Create(label, similarity);
                docRelations.Add(docRelation);
                
                i += 3;
            }

            Console.WriteLine("Done");
            
            TrainTree(docRelations);
            */
        }

        static bool IsStopWord(string word){
            string[] stopwords = File.ReadAllLines("data/english-stopwords.txt");

            return Array.Exists(stopwords, stopword => stopword == word);
        }
        static NDArray Vectorize(string doc, Word2Vec model){
            doc = doc.ToLower();
            Regex rgx = new Regex("[^a-öA-Ö0-9 -]");
            doc = rgx.Replace(doc, "");
            
            var words = Array.FindAll(doc.Split(' '), 
                        word => !IsStopWord(word));
            
            var word_vectors = np.zeros((words.Length, MODEL_VECTOR_SIZE));
            for(int i = 0; i < words.Length; i++){
                try
                {
                    var vector = model[words[i]];
                    for(int j = 0; j < vector.Size; j++){
                        word_vectors[i, j] = (float) vector[j];
                    }
                }
                catch (System.Exception)
                {
                    //Console.WriteLine("Could not find word, " + words[i]);
                }
            }
            
            //Console.WriteLine(word_vectors.ToString());
            //Console.WriteLine(np.mean(word_vectors, 0).ToString());
            return np.mean(word_vectors, 0);
        }
        static float Magnitude(NDArray v){
            return np.sqrt(np.dot(v, v));
        }
        static float CosineSimilarity(NDArray v1, NDArray v2){
            return np.dot(v1, v2) / (Magnitude(v1) * Magnitude(v2));
        }

        // This example requires installation of additional NuGet package for 
        // Microsoft.ML.FastTree at
        // https://www.nuget.org/packages/Microsoft.ML.FastTree/
        public static void TrainTree(List<Tuple<bool, float[]>> data)
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
            [VectorType(300)]
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
    }
}
