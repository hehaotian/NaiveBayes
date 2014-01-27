/*
LING 572 Homework 3
 
@author: Haotian He
@time: 23:28, 01/23/2014
 
This is a collection for Naive Bayes model
 
*/



import java.io.*;
import java.util.*;

public class NaiveBayes {
   
   private String train_path;
   private String test_path;
   private int prior_delta;
   private double cond_prob_delta;
   private String model_file;
   private String sys_output;
   
   private Map<String, Map<String, Integer>> train_data = new HashMap<String, Map<String, Integer>>();
   private Map<String, Map<String, Integer>> test_data = new HashMap<String, Map<String, Integer>>();
   private Map<String, Double> train_model = new HashMap<String, Double>();
   
   private Set<String> classLabs = new TreeSet<String>();
   private Map<String, Double> classProbs = new HashMap<String, Double>();
   
   private Map<String, Integer> documents_count = new HashMap<String, Integer>();
   private Map<String, Integer> words_count = new HashMap<String, Integer>();
   private Set<String> unique_features_count = new TreeSet<String>();
   
   private Map<String, Map<String, Integer>> train_matrix = new HashMap<String, Map<String, Integer>>();
   private Map<String, Map<String, Integer>> test_matrix = new HashMap<String, Map<String, Integer>>();
   
   public NaiveBayes(String train_path, String test_path, int prior_delta, double cond_prob_delta) throws IOException {
      this.train_path = train_path;
      this.test_path = test_path;
      this.prior_delta = prior_delta;
      this.cond_prob_delta = cond_prob_delta;
   }
   
   public void build_model(PrintStream ps, boolean bernoulli) throws IOException {
      
      BufferedReader br = new BufferedReader(new FileReader(train_path));
      String line = "";
      String classLabel = "";
      
      int all_documents = 0;
      
      while ((line = br.readLine()) != null) {
         all_documents ++;
         String[] tokens = line.split(" ");
         classLabel = tokens[0];
         classLabs.add(classLabel);
         
         if (!train_data.containsKey(classLabel)) {
            train_data.put(classLabel, new HashMap<String, Integer>());
            documents_count.put(classLabel, 1);
         } else {
            documents_count.put(classLabel, documents_count.get(classLabel) + 1);
         }
         
         for (int i = 1; i < tokens.length; i++) {
            String token = tokens[i];
            String word = token.replaceAll(":[\\d]+", "");
            
            unique_features_count.add(word);
            
            int count = 0;
            if (!bernoulli) count = Integer.parseInt(token.replaceAll("[\\w]+:", ""));
            else count = 1;
               
            if (!words_count.containsKey(classLabel)) {
               words_count.put(classLabel, count);
            } else {
               words_count.put(classLabel, words_count.get(classLabel) + count);
            }
            
            if (train_data.get(classLabel).containsKey(word)) {
               train_data.get(classLabel).put(word, train_data.get(classLabel).get(word) + count);
            } else {
               train_data.get(classLabel).put(word, count);
            }
         }
      }
      
      ps.println("%%%%% prior prob P(c) %%%%%");
      for (String doc_cl : documents_count.keySet()) {
         double prior_prob = (prior_delta + documents_count.get(doc_cl) * 1.0) / (all_documents + prior_delta * documents_count.keySet().size());
         double lg_prior_prob = Math.log10(prior_prob);
         ps.println(doc_cl + "\t" + prior_prob + "\t" + lg_prior_prob);
         classProbs.put(doc_cl, lg_prior_prob);
      }
      
      ps.println("%%%%% conditional prob P(f|c) %%%%%");
      for (String className : train_data.keySet()) {
         ps.println("%%%%% conditional prob P(f|c) c=" + className + " %%%%%");
         for (String w : unique_features_count) {
            int value = 0;
            if (train_data.get(className).get(w) != null) {
               value = train_data.get(className).get(w);
            }
            double cond_prob = 0.0;
            if (!bernoulli) cond_prob = (cond_prob_delta + value * 1.0) / (cond_prob_delta * unique_features_count.size() + words_count.get(className));
            else cond_prob = (cond_prob_delta + value * 1.0) / (cond_prob_delta * 2 + words_count.get(className));
               
            double lg_cond_prob = Math.log10(cond_prob);
            ps.println(w + "\t" + className + "\t" + cond_prob + "\t" + lg_cond_prob);
            String feat_class = w + "_" + className;
            train_model.put(feat_class, lg_cond_prob);
         }
      }
   }
   
   public void prediction(String file_path, PrintStream ps) throws IOException {
      
      BufferedReader br = new BufferedReader(new FileReader(file_path));
      
      boolean train = false;
      if (file_path.contains("train")) {
         ps.println("%%%%% training data:");
         train = true;
      } else {
         ps.println();
         ps.println();
         ps.println();
         ps.println("%%%%% test data:");
      }
      
      String line = "";
      String correct_classLabel = "";
      int instanceName = -1;
      while ((line = br.readLine()) != null) {
         
         instanceName ++;
         String[] tokens = line.split(" ");
         correct_classLabel = tokens[0];
         Map<String, Double> pred_probs = new HashMap<String, Double>();
         
         Iterator itr = classLabs.iterator();
         while (itr.hasNext()) {
            String label = "" + itr.next();
            double pred_prob = 0.0;
            double sum_cond_probs = 0.0;
            
            for (int i = 1; i < tokens.length; i++) {
               String word = tokens[i].replaceAll(":[\\d]+", "");
               String feature_key = word + "_" + label;
               if (train_model.containsKey(feature_key)) {
                  sum_cond_probs += train_model.get(feature_key);
               }
            }
            
            pred_prob = Math.pow(1.077, sum_cond_probs + classProbs.get(label));
            pred_probs.put(label, pred_prob);
         }
         
         Map<String, Double> final_pred_probs = new HashMap<String, Double>();
         
         double sum_classes = 0.0;
         for (String class_1 : pred_probs.keySet()) {
            sum_classes += pred_probs.get(class_1);
         }
         for (String class_2 : pred_probs.keySet()) {
            final_pred_probs.put(class_2, pred_probs.get(class_2) / sum_classes);
         }
         
         Map<String, String> descend_probs = sortByComparator(final_pred_probs);
         
         ps.println("array:" + instanceName + " " + correct_classLabel);
         
         int counter = 0;
         for (Map.Entry entry : descend_probs.entrySet()) {
            ps.print(" " + entry.getKey() + " " + entry.getValue());
            ps.println();
            String key = "" + entry.getKey(); // predicted class
            counter ++;
            if (counter == 1) {
               if (train) {
                  if (!train_matrix.containsKey(correct_classLabel)) {
                     train_matrix.put(correct_classLabel, new HashMap<String, Integer>());
                  } else if (train_matrix.get(correct_classLabel).containsKey(key)) {
                     train_matrix.get(correct_classLabel).put(key, train_matrix.get(correct_classLabel).get(key) + 1);
                  } else {
                     train_matrix.get(correct_classLabel).put(key, 1);
                  }
               } else {
                  if (!test_matrix.containsKey(correct_classLabel)) {
                     test_matrix.put(correct_classLabel, new HashMap<String, Integer>());
                  } else if (test_matrix.get(correct_classLabel).containsKey(key)) {
                     test_matrix.get(correct_classLabel).put(key, test_matrix.get(correct_classLabel).get(key) + 1);
                  } else {
                     test_matrix.get(correct_classLabel).put(key, 1);
                  }
               }
            }
         }
      }
   }
   
   private Map sortByComparator(Map unsortMap) {
      List list = new LinkedList(unsortMap.entrySet());
      Collections.sort(list, new Comparator() {
         public int compare(Object o1, Object o2) {
            return ((Comparable) ((Map.Entry) (o2)).getValue())
            .compareTo(((Map.Entry) (o1)).getValue());
         }
      });
      Map sortedMap = new LinkedHashMap();
      for (Iterator it = list.iterator(); it.hasNext();) {
         Map.Entry entry = (Map.Entry) it.next();
         sortedMap.put(entry.getKey(), entry.getValue());
      }
      return sortedMap;
   }
   
   public void confusion_matrix() throws IOException {
      
      System.out.println("class_num=" + classLabs.size() + " feat_num=" + unique_features_count.size());
      System.out.println("class_num=" + classLabs.size() + " feat_num=" + unique_features_count.size());
      print_matrix("training");
      print_matrix("test");
   }
   
   private void print_matrix(String train_test) {
      System.out.println("Confusion matrix for the " + train_test + " data:");
      System.out.println("row is the truth, column is the system output\n");
      System.out.print("\t");
      for (String class_3 : classLabs) {
         System.out.print(class_3 + " ");
      }
      System.out.println();
      for (String class_4 : classLabs) {
         System.out.print(class_4 + " ");
         for (String class_5 : classLabs) {
            if (train_test.equals("training")) {
               if (train_matrix.get(class_4).get(class_5) != null) {
                  System.out.print(train_matrix.get(class_4).get(class_5) + " ");
               } else {
                  System.out.print("0 ");
               }
            } else if (test_matrix.get(class_4).get(class_5) != null) {
               System.out.print(test_matrix.get(class_4).get(class_5) + " ");
            } else {
               System.out.print("0 ");
            }
         }
         System.out.println();
      }
      System.out.println();
      if (train_test.equals("training")) {
         System.out.println(" Training accuracy=" + getAccuracy(train_matrix) + "\n\n");
      } else {
         System.out.println(" Test accuracy=" + getAccuracy(test_matrix));
      }
   }
    
   private double getAccuracy(Map<String, Map<String, Integer>> matrix) {
      
      int correct_count = 0;
      int sum = 0;
      for (String str_1 : matrix.keySet()) {
         for (String str_2 : matrix.get(str_1).keySet()) {
            int count = matrix.get(str_1).get(str_2);
            if (str_1.equals(str_2)) {
               correct_count += count;
            }
            sum += count;
         }
      }
      return correct_count * 1.0 / sum;
   }  
}
