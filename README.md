## **Developing predictive models for COVID-19 diagnosis in paediatric patients: A case study about the potentials of Machine Learning in Public Health**
By Anna Mas-Casadesús / July 2020 / [AllWomen Data Science Bootcamp Capstone Project](https://www.allwomen.tech/academy/data-science-immersive-program/)


**Project Description**

A recent study from the Hospital Sant Joan de Déu of Barcelona has found that children appear to have a similar prevalence of COVID-19 antibodies to adults. Research also seems to indicate that most of the affected children present mild or no symptoms. This makes `diagnosis particularly difficult, the number of children affected by COVID-19 being still unknown`. (1)

Amongst many other institutions, Hospital Vall d’Hebron of Barcelona is conducting research on COVID-19 (2). The Hospital made available to the public a dataset with paediatric patients with similar symptoms, some of which were diagnosed with COVID-19 and some others not (3) (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) symptoms_paediatric_original.xlsx). 

I got in touch with the doctor who published the dataset, Dr. Antoni Soriano-Arandes, so I could understand the data and decide on the best strategy to analyse it. With this information, I designed a study using machine learning classification models as an approach to try to identify clinical indicators predictive of COVID-19 diagnosis in children.

**Project Steps**

1) `Data pre-processing`: Cleaning and transformation of the raw data to make it suitable for building and training machine learning models, including:
    - Basic data cleaning and manipulations (e.g. removal of duplicates, encoding)
    - Data standardisation
    - Feature selection (including Principal Component Analyses)
    - Missing data imputation (Multiple Imputation by Chained Equations or MICE method)
    - Oversampling (Synthetic Minority Oversampling TEchnique or SMOTE method)
2) `Exploratory data analyses`: Performing initial analyses on the data to summarise its main characteristics and discover patterns.

3) `Data modelling`: Building and training machine learning algorithms on the data to make predictions on new data
    - 3.1) Pipeline with several classification models (Logistic Regression, k-Nearest Neighbors, Support Vector Machine Classifier, Decision Tree, Random Forest, Extra Gradient Boosting), using stratified train cross-validation and hyperparameter optimisation or tuning with RandomizedSearch strategy
    - 3.2) Pipeline with best performing models (i.e. Random Forests), using stratified train cross-validation and hyperparameter tuning with GridSearch strategy
    - 3.3) Application of the Global Surrogate Method for data interpretability
    - 3.4) Model evaluation through performance metrics
4) `Model interpretation`: Trying to understand and explain the steps and decisions a machine learning model takes when making predictions, including:
    - Decision Tree visualisations
    - Feature importance analyses and visualisations through SHapley Additive exPlanations or SHAP method

Please, go to the ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) Jupyter Notebooks folder for further details on each of the steps and all the Python codes - you will also find a file with the summarised code (![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) Summary Code.ipynb/.py). Refer to the ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) Codebook.md for specific information regarding the dataset variables. 

**Project Outcome**

The analyses determined that `a Random Forest model (i.e. average of multitudes of decision trees) was the best classifier of positive and negative COVID-19 diagnoses for our dataset of paediatric patients`, with an overall `accuracy on the train subset of 88% and of 69% on new data or the test subset` (test accuracy of 83% in a projected model with oversampled data). However, these accuracy metrics were brought about a `good classification of negative COVID-19 cases but a poor classification of positive cases` (F1 scores or harmonic means of precision and recall of 86% and 22%, respectively; projected model: 64% and 51%). 

The dataset used for this project only contained information on 19 COVID-19 positive diagnoses out of a total of 86 cases. Therefore, not only the number of positive cases is very low and imbalanced, but the overall dataset should be bigger to be able to train well the machine learning models and have a stronger predictive power. Given the good results classifying negative cases and the metrics observed for the projected model, a sample increase would likely improve the results. 

On another note, `the analyses with respect to which features were more important to predict COVID-19 in children showed that being younger, having taken antibiotics, not having received oxygen therapy, and/or not having had a chest-x ray taken were all protective factors` - or risk factors if the contrary conditions were true. Although these results must be viewed with caution for the reasons stated above, they hint what might be important factors in paediatric COVID-19 diagnosis and highlight the overall potential of machine learning approaches in the public health and epidemiology fields. 

See the ![#1589F0](https://via.placeholder.com/15/1589F0/000000?text=+) Presentation.pdf for extended discussion and to get an overview of the project process and results.

References:

* (1) https://www.sjdhospitalbarcelona.org/en/kidscorona
* (2) https://www.vallhebron.com/es/que-hacemos-en-el-campus/investigacion/investigacion-contra-el-coronavirus-sars-cov-2
* (3) http://dx.doi.org/10.17632/wvn7fcsgvh.2#file-f7fbc90a-8f10-45e0-ba67-7f6eaa903470 (data downloaded 15/06/2020)

Contact:
* Do not hesitate to contact me at amascasadesus [at] gmail.com if you have any questions or would like any further details. I am also very open to suggestions on how to improve the project and the code itself. I hope you enjoy this project as much as I did and thanks in advance for your interest! 
