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

    class TextProcessing
    {

        const int MODEL_VECTOR_SIZE = 300;
        static bool IsStopWord(string word){
            string[] stopwords = File.ReadAllLines("data/swedish-stopwords.txt");

            return Array.Exists(stopwords, stopword => stopword == word);
        }
        public static NDArray VectorizeMean(string doc, Word2Vec model){
            doc = doc.ToLower();
            Regex rgx = new Regex("[^a-zåäöA-ZÅÄÖ0-9 -]");
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
        public static float[,] VectorizeFull(string doc, Word2Vec model){
            doc = doc.ToLower();
            Regex rgx = new Regex("[^a-zåäöA-ZÅÄÖ0-9 -]");
            doc = rgx.Replace(doc, "");
            
            var words = Array.FindAll(doc.Split(' '), 
                        word => !IsStopWord(word));
            
            var word_vectors = np.zeros((words.Length, MODEL_VECTOR_SIZE));
            float[,] array = new float[words.Length, MODEL_VECTOR_SIZE];
            for(int i = 0; i < words.Length; i++){
                try
                {
                    var vector = model[words[i]];
                    for(int j = 0; j < vector.Size; j++){
                        word_vectors[i, j] = (float) vector[j];
                        array[i, j] = (float) vector[j];
                    }
                }
                catch (System.Exception)
                {
                    //Console.WriteLine("Could not find word, " + words[i]);
                }
            }
            
            return array;
        }
        static float Magnitude(NDArray v){
            return np.sqrt(np.dot(v, v));
        }
        static float CosineSimilarity(NDArray v1, NDArray v2){
            return np.dot(v1, v2) / (Magnitude(v1) * Magnitude(v2));
        }
    }
}
 