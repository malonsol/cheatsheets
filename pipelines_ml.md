# A Simple Guide to Scikit-learn Pipelines
Credit: [Rebecca Vickery](https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf)

Learn how to use pipelines in a scikit-learn machine learning workflow

In most machine learning projects the data that you have to work with is unlikely to be in the ideal format for producing the best performing model. There are quite often a number of transformational steps such as **encoding categorical variables, feature scaling and normalisation** that need to be performed. Scikit-learn has built in functions for most of these commonly used transformations in it’s preprocessing package.

Scikit-learn pipelines are a tool to simplify this process. They have several key benefits:
- They make your workflow much easier to read and understand.
- They enforce the implementation and order of steps in your project.
- These in turn make your work much more reproducible.

Dataset: loan prediction

## Transformers

```python
import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train.drop('Loan_ID', axis=1)
train.dtypes
```
![dtypes](https://us.hidester.com/proxy.php?u=eJwBQAC%2F%2F3M6NTY6IjxTxv8lvmwON%2FEkrH8E%2F7WBbmlIeBquAnhOJtDBRpArEbJfvZ00URuwgRDYUJyEDIHDJeg6%2FNlFIjtb3Bto&b=7)

I can see that I have both categorical and numeric variables so as a minimum I am going to have to apply a one hot encoding transformation and some sort of scaler.

Before building the pipeline I am splitting the training data into a train and test set so that I can validate the performance of the model.


```python
X = train.drop('Loan_Status', axis=1)
y = train['Loan_Status']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

The first step in building the pipeline is to define each transformer type.

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
```

Next we use the ColumnTransformer to apply the transformations to the correct columns in the dataframe. Before building this I have stored lists of the numeric and categorical columns using the pandas dtype method.

```python
numeric_features = train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = train.select_dtypes(include=['object']).drop(['Loan_Status'], axis=1).columns
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
```

## Fitting the classifier

The next step is to **create a pipeline** that combines the preprocessor created above with a classifier. In this case I have used a simple RandomForestClassifier to start with.

```python
from sklearn.ensemble import RandomForestClassifier
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

## Model selection

**A pipeline can also be used during the model selection process**. The following example code loops through a number of scikit-learn classifiers applying the transformations and training the model.

```python
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]
for classifier in classifiers:
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', classifier)])
    pipe.fit(X_train, y_train)   
    print(classifier)
    print("model score: %.3f" % pipe.score(X_test, y_test))
```

![model_scores](https://us.hidester.com/proxy.php?u=eJwBQQC%2B%2F3M6NTc6IjxTxv8lvmwON%2FEkrH8E%2F7WBbmlIeBquAn5JJsffXfwlQqQgnKk%2FYlzqmlvSAImPb536KMtzosdMnCI7mM4ddA%3D%3D&b=7)

**The pipeline can also be used in grid search** to find the best performing parameters.
To do this you first need to create a parameter grid for your chosen model.
*One important thing to note is that you need to append the name that you have given the classifier part of your pipeline to each parameter name.*
*In my code above I have called this ‘classifier’ so I have added classifier__ to each parameter.*
Next I created a grid search object which includes the original pipeline.
When I then call fit, the transformations are applied to the data, before a cross-validated grid-search is performed over the parameter grid.

```python
param_grid = { 
    'classifier__n_estimators': [200, 500],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth' : [4,5,6,7,8],
    'classifier__criterion' :['gini', 'entropy']}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(rf, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)
```

Before I started to use pipelines I would find that when I went back to a project to work on it again even after only a short time I would have trouble following the workflow again.
Pipelines have really helped me to put together projects that are both easily repeatable and extensible.
