# Improving Machine Learning Pipeline

## Run the code:
* Download the whole folder and keep its hierarchy.
* Open your terminal and change the directory to where the folder is at.
* Input command as the following:
```
python3 main.py
```

## Notice:
* The default test_length is 6. If you want to change the testing window, please open main.py in text editor, find the __main__, and change the test_length.
* Folder 'eval_results' provide PR curves and evaluation summary (classifier_eval.csv) for testing window = 6 months.
* Folder 'eval_results_1' provide PR curves and evaluation summary (classifier_eval.csv) for testing window = 12 months.
* Folder 'results' provide distribution plots, boxplots and summary stats for each used variable.
* Caveat: Running SVM model is extremely slow (it may take more than 1 day to finish!). The user can decide whether to include SVM model for analysis by editing model_lst variable in main.py.

## Specification:
* utils.py - Read, explore, and split data.
* preprocess.py - Impute, discretize, and pre-process data.
* models.py - Build supervised learning classifiers and evaluate each by multiple metrics.
* main.py - Apply this pipeline-library to the data given.
* eval_analysis - helper functions used for results analysis.
* Results_Analysis_test6.ipynb - Analyze results as the test window = 6 months.
* Results_Analysis_test12.ipynb - Analyze results as the test window = 12 months.
