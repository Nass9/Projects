# Projects

## Master 2:

### Quantitative finance : 
      
      Project on Python/R : SEEKING SIGNALS FROM ESG DATA 

      Purpose:  Show that applying Machine Learning methods on ESG dataset allows us to create
      equity portfolio with higher return and lower volatility than their benchmark. We also investigate the
      interpretability of our models using SHapley Additive exPlanations and compare the results to a traditional
      Logistic Regression-based approach. We next used the Fama/French Five-Factor for a Backtesting of our result.

      Machine Learning : Web scraping of ESG score From 500 company of the S&P 500 and analisys of them as of june 2021
                   Imputation methods : Mean, KNN imputer, MICE (Multivariate Imputation by Chained Equations), VAE (Variational Auto encoders)
                   Supervised Classification : Logistic Regression, Random Forest, Gradient Boosting Trees
                   
      Finance :  Portfolio Creation (Long only, Long-Short) 
                 Improving the Efficient Frontier 
                 Application of Fama-French (5 factors) for the Backtesting



### Data Mining :

    Project on R: Analysis of the determining factors of the matches in a speed-dating
    
    Encoding categorical data : One-hot encoding and Label encoding

    Unsupervised Machine learning: 
      Classification with  K-means
          
    dimensionality-reduction: Principal Component Analysis (PCA)
    
    Supervised Machine learning: 
      Regression:   Logistic Regression
                     Gradient Boosting Trees

    Imbalanced Dataset:
      Over-sampling: SMOTE - ADASYN. 
      Under-sampling: Random under-sampling

    Feature selection: RFE (Recursive Feature Selection) 



### Deep Learning:

    Predicting the number of air passengers (Python)

    After a hudge work on data cleaning and preprocessing we used some Machine Learning and Deep Learning methods 
    to predict the number of air passengers of an airline company.

    Machine Learning:
        Logistic Regression
        ElasticNet
        Random Forest
        Gradient Boosting Trees

    Deep Learning : Sequential Neural Network (Keras)
                    Recurrent Neural Network  (Keras)




### Financial Econometrics:

    Projet on Python :
    Unobserved Components Models in Economics and Finance : The role of the Kalman filter in time series econometrics

    Purpose: The treatment of time series using state space models (SSM), the fundamental objective of this project 
    is to show that systems can be explained by a set of unobservable components (trend, seasonality, cycle, error in particular).
    Major role of the Kalman filter algorithm in the resolution of SSM models and good flexibility of the algorithm according 
    to the characteristics of the system.

    Some Models:

    UNIVARIATE LINEAR MODELS:
      Local level model
      Linear trend model
      Linear trend model with seasonality

    MULTIVARIATE LINEAR MODELS:
      Linear trend cycle decomposition model
      Variable frequency model: Kalman filter offers the flexibility needed for dealing with data irregularities, 
                                such as missing observations and observations at mixed frequencies.



### Machine Learning :

    Project on python : Predict age from brain gray matter

    Modelling Methodology :
    Two step prediction:
    - We used a Kmeans algorithm to binarize our target variable. This cutoff is determined relatively 
    arbitrarily from the results of the interpreted Kmeans.
    We then used the Recursive Feature Elimination Cross-Validation (RFECV) variable selection algorithm, and implemented 
    this algorithm on Random Forest with Accuracy as a performance metric in order to have a balanced distribution of the target. 
    We optimized our models with GridsearchCV.

    -Next, we decided here to apply a Ridge model to each of the two sub-dataframes using the same methodology: 
    Variable selection by RFECV
    Determination of the optimal penalization ("Alpha") of the model thanks to RidgeCV on each of the two sub-models.
    Results obtained by Cross-Validation


    Unsupervised Machine learning : 
          Classification with  K-means

    Supervised Machine learning : 
          Regression :   Ridge Regression
                         Random Forest

    Variable Selection : Recursive Feature Elimination Cross-Validation (RFECV)

    Optimization :  GridSearchCV



### Advanced python :

    Project on Python/PySpark : 

    Resume: For this project we have chosen to work on two IMDb databases found on Kaggle. In this 
    project we will process these databases on PySpark in order to develop an analysis on these data.

    Database processing on PySpark:
    Quasi-Json decoding 
    Data preprocessing (missing values...)

    Statistical analysis:
    Descriptive statistics
    Data visualization
    
    
    Natural Language Processing :

      Hard cleaning, with two funtions made to remove :
      - Punctuation, accents and capitalization
      - Words of less than 2 letters and more than 18 letters
      - Single numbers and numbers embedded in words

      Treat stopwords:
        Tokenization  
        Lemmatization
        Word Embedding ( Modèle FastText de Facebook, Kmeans)

      Text Data Visualization
      Topic Modelling : Optimal topics
      LDA(linear discriminant analysis) for Topic Modelling

    To put our analysis further we have created a movie recommendation system based on the LDA results.




### Advanced Machine Learning :

      Courses:
      -Logistic regression as a neural network
      -Classification with one layer
      -Deep Neural Network for Image Classification

      Project on Python :  Machine Learning avancé : Deep Learning for NLP

      Purpose: We will develop our first Deep Learning models applied to language processing. Rather than 
      recoding everything from scratch, we will rely on the power of the Keras library which will allow us 
      to connect layers on the fly and implement more exotic architectures.


      POS-Tagging and Shallow Parsing are two classic NLP tasks:
      - POS-Tagging: assigns to each word a unique tag that indicates its syntactic role (noun, verb, adverb, ..)
      - Shallow Parsing: assigns to each sentence segment a unique tag indicating the role of the syntactic element 
        to which it belongs (nominal group, verbal group, etc.)

      Here we will redo the work of the NLP almost from scratch article, which consists in creating a 
      neural network to perform each task, then we will make a shared model and finally a hiearchical model.


      POS-Tagging:
            Word embedding
            Creation of the tags
            Creation of the Neural Network (Keras)

      Shallow Parsing:
            Word embedding
            Creation of the tags
            Creation of the Neural Network (Keras)

      Multi-task learning: 
      Full multi tagged (POS tagging + Shallow parsing):
            Word embedding
            Creation of the tags
            Creation of the neural network common to both tasks

      Hierarchical learning:
      Another way to do multi-task is to build a cascade architecture where the tasks do not intervene at the same depth of the neural network.
      Construction of a cascade model of type : 
                                    POS
                                  /
      EMBEDDING - DENSE - DROPOUT 
                                  \
                                    DENSE - DROPOUT - CHUNK






### Série temporelle :

### Gestion de risques: 


### CRM Analytics :
### Systeme repartis :




## Master 1 :

- Thesis on Hedge-Fund Persistence (SAS): Non parametric test (RUNS test), normality test, Goodness-of-fit test, ARMA model.

- Thesis on Crisis prediction and its determinants (Python/SAS): Supervised Marchine Learning (Logit, Random Forest, Cart, Adaboost, Decision Tree, Gradient Boosting Trees), explaination with Shapley-Value.

- Project on Python: Analysis of the determinants of AirBnb’s house prices (Web scraping, application of regression methods -Lasso and Gradient Boosting-, Classification -Logit, Random Forest, Cart, Decision Tree-, Text Mining approach).



# Challenge:

Deloitte : Drim Game

Ekimetrics : 


# Certification :

### 2021 (in progress): 
- Datascientest : Deep-Learning avec le framework Keras 
- Microsoft Certified: Azure Data Scientist Associate

### 2021: 
- Datascientest : Advanced Machine Learning with Scikit-learn 
- Microsoft Azure Fundamentals

### 2020: 
- Datascientest : Text Mining 
- Datascientest : Machine Learning with Scikit-learn
- DataScientest first exam (Python)
