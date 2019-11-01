using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Proxem.Word2Vec;
using NumSharp;
using System.Text.RegularExpressions;

namespace TextMatchPTS
{

    class Program
    {

        private class QuoraQuestionPair
        {
            public string Question1 { get; set; }
            public string Question2 { get; set; }
            public bool Related { get; set; }
        }
        
        static List<QuoraQuestionPair> readQuestionPairs(int from, int to, string filePath){
            string[] lines = File.ReadAllLines(filePath);
            var questionPairs = new List<QuoraQuestionPair>();    
            for(int i = from; i < to; i++){
                var questionPair = new QuoraQuestionPair();
                try{
                    var l = lines[i].Split(',');
                    
                    questionPair.Question1 = l[4];
                    questionPair.Question2 = l[5];
                    questionPair.Related = l[6].Equals("1");
                    
                    Console.WriteLine(questionPair.Question1);
                    Console.WriteLine(questionPair.Question2);
                    questionPairs.Add(questionPair);

                }catch(System.Exception){
                    Console.WriteLine("problem at line: " + i);
                }
            }
            Console.WriteLine("Questions length: " + questionPairs.Count);
            return questionPairs.OrderBy(a => Guid.NewGuid()).ToList();
        }

        static List<Tuple<bool, float[]>> getQuestionsAsVector(int from, int to, string filePath, Word2Vec word2VecModel){
            //var word2VecModel = Word2Vec.LoadText("models/newsv/model.txt", true, false, "");
            
            var docRelations = new List<Tuple<bool, float[]>>();
            var qPs = readQuestionPairs(from, to, filePath);
            foreach(QuoraQuestionPair questionPair in qPs){
                try{
                    NDArray v1 = TextProcessing.Vectorize(questionPair.Question1, word2VecModel);
                    NDArray v2 = TextProcessing.Vectorize(questionPair.Question2, word2VecModel);
                    //NDArray diff = np.subtract(v1, v2);
                    double[] conc = np.concatenate((v1, v2)).ToArray<double>();
                    // NDArray -> double[] -> float[]  
                    float[] merged = Array.ConvertAll(conc, s => (float) s);
                    var docRelation = Tuple.Create(questionPair.Related, merged);
                    docRelations.Add(docRelation);
                }catch(System.Exception){
                    Console.WriteLine("could not vectorize.");
                }
            }
            return docRelations;
        }
        static void Main(string[] args)
        {
            
            //var word2VecModel = Word2Vec.LoadBinary("models/GoogleNews-vectors-negative300.bin", true, false, "", Encoding.GetEncoding("ISO-8859-1"));
            var word2VecModel = Word2Vec.LoadText("models/bloggmix2013w2v.txt", true, false, "");
            
            var train = getQuestionsAsVector(0, 3000, "data/train_sv.csv", word2VecModel);
            var test = getQuestionsAsVector(0, 100, "data/test_sv.csv", word2VecModel);
            Learning.LargeNetwork(train, test);
            Learning.LightGbm(train);
        }
    }
}
