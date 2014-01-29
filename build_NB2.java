import java.io.*;
import java.util.*;

public class build_NB2 {
    
    private static String train_path;
    private static String test_path;
    private static int prior_delta;
    private static double cond_prob_delta;
    private static String model_file;
    private static String sys_output;
    
    public static void main(String[] args) throws IOException {

        train_path = args[0];
        test_path = args[1];
        prior_delta = Integer.parseInt(args[2]);
        cond_prob_delta = Double.parseDouble(args[3]);
        model_file = args[4];
        sys_output = args[5];
        
        PrintStream model = new PrintStream(model_file);
        PrintStream sys = new PrintStream(sys_output);
        
        boolean bernoulli = false;
        
        NaiveBayes nb = new NaiveBayes(train_path, test_path, prior_delta, cond_prob_delta);
        nb.build_model(model, bernoulli);
        nb.prediction(train_path, sys, bernoulli);
        nb.prediction(test_path, sys, bernoulli);
        nb.confusion_matrix();

    }

}