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
        
        static List<QuoraQuestionPair> readQuestions(int from, int to, string filePath){
            string[] lines = File.ReadAllLines(filePath);
            var questions = new List<QuoraQuestionPair>();
    
            for(int i = from; i < to; i++){
                var question = new QuoraQuestionPair();
                try{
                    var l = lines[i].Split(',');
                    
                    question.Question1 = l[4];
                    question.Question2 = l[5];
                    question.Related = l[6].Equals("1");
                    
                    Console.WriteLine(question.Question1);
                    Console.WriteLine(question.Question2);
                    questions.Add(question);

                }catch(System.Exception){
                    Console.WriteLine("problem at line: " + i);
                }
            }
            Console.WriteLine("Questions length: " + questions.Count);
            return questions.OrderBy(a => Guid.NewGuid()).ToList();
        }

        static List<Tuple<bool, float[]>> getQuestionsAsVectors(int from, int to, string filePath, Word2Vec word2VecModel){
            //var word2VecModel = Word2Vec.LoadText("models/newsv/model.txt", true, false, "");
            
            var questions = readQuestions(from, to, filePath);
            var docRelations = new List<Tuple<bool, float[]>>();
            foreach(QuoraQuestionPair question in questions){
                try{
                    NDArray v1 = TextProcessing.VectorizeMean(question.Question1, word2VecModel);
                    //Console.WriteLine(question.Question2.Length);
                    NDArray v2 = TextProcessing.VectorizeMean(question.Question2, word2VecModel);
                    //NDArray diff = np.subtract(v1, v2);
                    double[] conc = np.concatenate((v1, v2)).ToArray<double>();
                    
                    float[] merged = Array.ConvertAll(conc, s => (float) s);
                    
                    var docRelation = Tuple.Create(question.Related, merged);
                    docRelations.Add(docRelation);
                }catch(System.Exception){

                }
            }
            return docRelations;
        }
        static List<Tuple<bool, float[,], float[,]>> getQuestionsAsMatrices(int from, int to, string filePath, Word2Vec word2VecModel){
            //var word2VecModel = Word2Vec.LoadText("models/newsv/model.txt", true, false, "");
            
            var questions = readQuestions(from, to, filePath);
            var docRelations = new List<Tuple<bool, float[,], float[,]>>();
            foreach(QuoraQuestionPair question in questions){
                try{
                    float[,] v1 = TextProcessing.VectorizeFull(question.Question1, word2VecModel);
                    //Console.WriteLine(question.Question2.Length);
                    float[,] v2 = TextProcessing.VectorizeFull(question.Question2, word2VecModel);
                    //double[] conc = np.concatenate((v1, v2)).ToArray<double>();
                    //Console.WriteLine(v1.ToString());
                    //Console.WriteLine(v2.ToString());
                    //float[] merged = Array.ConvertAll(conc, s => (float) s);
                     
                    var docRelation = Tuple.Create(question.Related, v1, v2);
                    docRelations.Add(docRelation);
                }catch(System.Exception){

                }
            }
            return docRelations;
        }
        static void Main(string[] args)
        {
            
            //var word2VecModel = Word2Vec.LoadBinary("models/GoogleNews-vectors-negative300.bin", true, false, "", Encoding.GetEncoding("ISO-8859-1"));
            var word2VecModel = Word2Vec.LoadText("models/bloggmix2008w2v.txt", true, false, "");
            
            var train = getQuestionsAsMatrices(0, 1000, "data/train_sv.csv", word2VecModel);
            var test = getQuestionsAsMatrices(0, 100, "data/test_sv.csv", word2VecModel);

            Train.MultilayerPerceptron2DInput(train, test);
            //Console.WriteLine("Training small network");
            //Train.MultilayerPerceptron1DInputSm(train, test);
            //Console.WriteLine("Training large network");
            //Train.MultilayerPerceptron1DInputLg(train, test);
            //Train.LightGbm(train);
        }
    }
}
