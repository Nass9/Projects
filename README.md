
# Table of contents
1. [Projects](#Projects)
    1. [Master 2](#Master2)
          1. [Quantitative Finance](#Quantitativefinance)
          2. [Data Mining](#DataMining)
          3. [Deep Learning](#DeepLearning)
          4. [Financial Econometrics](#FinancialEconometrics)
          5. [Machine Learning](#MachineLearning)
          6. [Advanced Python](#Advancedpython)
          7. [Advanced Machine Learning](#AdvancedMachineLearning)
    2. [Master 1](#Master1)
2. [Challenge](#Challenge)
    1. [Deloitte](#Deloitte)
    2. [Ekimetrics](#Ekimetrics)
3. [Certification](#Certification)






# Projects <a name="Projects"></a>

## Master 2: <a name="Master2"></a>

### Quantitative finance:  <a name="Quantitativefinance"></a>
      
      Project on Python/R: SEEKING SIGNALS FROM ESG DATA 

      Purpose:  Show that applying Machine Learning methods on ESG dataset allows us to create
      equity portfolio with higher return and lower volatility than their benchmark. We also investigate the
      interpretability of our models using SHapley Additive exPlanations and compare the results to a traditional
      Logistic Regression-based approach. We next used the Fama/French Five-Factor for a Backtesting of our result.

      Machine Learning: Web scraping of ESG score From 500 company of the S&P 500 and analisys of them as of june 2021
                   Imputation methods : Mean, KNN imputer, MICE (Multivariate Imputation by Chained Equations), VAE (Variational Auto encoders)
                   Supervised Classification : Logistic Regression, Random Forest, Gradient Boosting Trees
                   
      Finance:  Portfolio Creation (Long only, Long-Short) 
                Improving the Efficient Frontier 
                Application of Fama-French (5 factors) for the Backtesting



### Data Mining: <a name="DataMining"></a>

    Project on R: Analysis of the determining factors of the matches in a speed-dating
    
    Encoding categorical data: One-hot encoding and Label encoding

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



### Deep Learning:  <a name="DeepLearning"></a>

    Predicting the number of air passengers (Python)

    After a hudge work on data cleaning and preprocessing we used some Machine Learning and Deep Learning methods 
    to predict the number of air passengers of an airline company.

    Machine Learning:
        Logistic Regression
        ElasticNet
        Random Forest
        Gradient Boosting Trees

    Deep Learning: Sequential Neural Network (Keras)
                    Recurrent Neural Network  (Keras)




### Financial Econometrics: <a name="FinancialEconometrics"></a>

    Projet on Python:
    Unobserved Components Models in Economics and Finance: The role of the Kalman filter in time series econometrics

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



### Machine Learning: <a name="MachineLearning"></a>

    Project on python: Predict age from brain gray matter

    Modelling Methodology:
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


    Unsupervised Machine learning: 
          Classification with  K-means

    Supervised Machine learning: 
          Regression:   Ridge Regression
                        Random Forest

    Variable Selection: Recursive Feature Elimination Cross-Validation (RFECV)

    Optimization:  GridSearchCV



### Advanced python: <a name="Advancedpython"></a>

    Project on Python/PySpark: 

    Resume: For this project we have chosen to work on two IMDb databases found on Kaggle. In this 
    project we will process these databases on PySpark in order to develop an analysis on these data.

    Database processing on PySpark:
    Quasi-Json decoding 
    Data preprocessing (missing values...)

    Statistical analysis:
    Descriptive statistics
    Data visualization
    
    
    Natural Language Processing:

      Hard cleaning, with two funtions made to remove:
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




### Advanced Machine Learning: <a name="AdvancedMachineLearning"></a>

      Courses:
      -Logistic regression as a neural network
      -Classification with one layer
      -Deep Neural Network for Image Classification

      Project on Python: Advanced Machine Learning Deep Learning for NLP

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






### Série temporelle:

### Gestion de risques: 


### CRM Analytics:
### Systeme repartis:




## Master 1: <a name="Master1"></a>

- Thesis on Hedge-Fund Persistence (SAS): Non parametric test (RUNS test), normality test, Goodness-of-fit test, ARMA model.

- Thesis on Crisis prediction and its determinants (Python/SAS): Supervised Marchine Learning (Logit, Random Forest, Cart, Adaboost, Decision Tree, Gradient Boosting Trees), explaination with Shapley-Value.

- Project on Python: Analysis of the determinants of AirBnb’s house prices (Web scraping, application of regression methods -Lasso and Gradient Boosting-, Classification -Logit, Random Forest, Cart, Decision Tree-, Text Mining approach).



# Challenge: <a name="Challenge"></a>

### Deloitte:  <a name="Deloitte"></a>

      DRiM Game 2020: Le challenge data science appliqué au risque de crédit (Python - Sas)

      DRiM Game, a competition in which students compete on a banking industry issue, related to the modeling of financial risks, 
       particularly credit risk

      Subject:
      Analysis of the explanatory factors (determinants) of the marginal recovery rates by defaulted maturity 
      band on a credit portfolio.
      The analysis, concerning panel data with a temporal dimension, is to be treated by application of "classical" econometric models and 
      the reasoned use of one or several machine learning methods. 

      Context:
      - Identify the factors influencing the loss given default (LGD) rate.
      - A regulatory constraint to be met in order to be allowed to use these models for the calculation of capital requirements
      - An operational issue to optimize the recovery process by guiding the timing and modalities of recovery actions
      

      Methodology:

      - For defaulted maturities (m=6, 9, 12, 18, 24), a marginal recovery rate can be estimated. For each of these rates, 
      the explanatory factors will be determined through a model. Thus, each maturity will have an explanatory model 
      for the recovery rate of the considered maturity
      - Treatment of the problem via the application of "classic" econometric models
      - Use of machine learning approaches of your choice, to be applied to the modeling steps that seem relevant to you
      - Verification of the assumptions underlying the models used
      - Deliverable in ppt format or other data visualization format
      





### Ekimetrics: <a name="Ekimetrics"></a>

      Hackathon 2021 – Eki x MoSEF (on Python)
      Mars 2021

      Supervised Classification Problem: Purchase prediction for a 12-month horizon 

      Subject:  One of Ekimetrics' clients is a major player in the automotive industry. This client sells 
      different car models worldwide but its sales have been decreasing for the last 3 years. 
      To reverse this trend, this client wants to implement a CRM strategy to better target these customers. 
      This strategy involves building several scores. 
      Today we will focus on one of these scores which represents a prediction of purchase in 12 months time.


      Methodology :

      Supervised Classification algorithm presented : XGBoost (Metrics of performance choosed : F1-Score)

      Presentation with an web apps build thanks to Streamlit


# Certification : <a name="Certification"></a>

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
