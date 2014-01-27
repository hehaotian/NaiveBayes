NaiveBayes
==========

NB model

1. binarize shell script binarizes training vectors;
2. build_NB1 builds the Bernoulli Naive Bayes model;
3. build_NB2 builds the multinomial Naive Bayes model.

Shell script format:

build_NB*.sh training_data test_data class_prior_delta cond_prob_delta model_file sys_output > acc_file
