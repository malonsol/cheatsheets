# Ideas : Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow

- *Standard correlation coefficient* (also called Pearson’s r):
    - It only measures linear correlations
    - It may completely miss out on nonlinear relationships
    - Use use Pandas’ `scatter_matrix` function, which plots every numerical attribute against every other numerical attribute → **Only for relatively small dataset, otherwise kernel dies!!**
    ```python
    from pandas.plotting import scatter_matrix
    attributes = ["median_house_value", "median_income", "total_rooms",
    "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    ```

## Chapter 2.End-to-End Machine Learning Project
### Prepare the Data for Machine Learning Algorithms
#### Handling Text and Categorical Attributes    
- It is important to fit the transformations (scalers, encoders, etc.) to the training data only, not to the full dataset (including the test set)
#### Transformation Pipelines
- Data transformation steps need to be executed in the right order → Scikit-Learn's `Pipeline` class
    ```python
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()),
        ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    ```
    - All **but the last estimator** must be transformers (i.e., they must have a `fit_transform()` method)
    - The names can be anything you like (as long as they are unique and don’t contain double underscores “__”): they will come in handy later for hyperparameter tuning
    - When you call the pipeline’s `fit()` method:
        - it calls `fit_transform()` sequentially on all transformers, passing the output of each call as the parameter to the next call, until...
        - until it reaches the final estimator, for which it just calls the `fit()` method.
    - The pipeline exposes the **same methods as the final estimator**
    - Single transformer able to handle categorical columns and the numerical columns altogether: `ColumnTransformer`
        ```python
        from sklearn.compose import ColumnTransformer
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs),
            ])
        housing_prepared = full_pipeline.fit_transform(housing)
        ```
### Select and Train a Model
#### Training and Evaluating on the Training Set
- Let’s first train a Linear Regression model:
    ```python
    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    # Let’s measure this regression model’s RMSE on the whole training set:
    from sklearn.metrics import mean_squared_error
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_rmse
    ```
#### Better Evaluation Using **Cross-Validation**:
    - 1st option: use the `train_test_split` function to split the training set into a smaller training set and a validation set
    - 2nd alternative: *K-fold cross-validation* feature:
    ```python
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
    scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    ```
        - **cross-validation** features expect a utility function (greater is better) rather than a cost function (lower is better)
    - *You should save every model you experiment with, so you can come back easily to any model you want. Make sure you save  both the hyperparameters and the trained parameters, as well as the cross-validation scores and perhaps the actual predictions as well. This will allow you to easily compare scores across model types, and compare the types of errors they make.*
    ```python
    from sklearn.externals import joblib
    joblib.dump(my_model, "my_model.pkl")
    # and later...
    my_model_loaded = joblib.load("my_model.pkl")
    ```

### Fine-Tune Your Model
#### Grid Search
```python
from sklearn.model_selection import GridSearchCV
param_grid = [
{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```
- First evaluate all 3 × 4 = 12 combinations; then try all 2 × 3 = 6 combinations; it will train each model five times (since we are using five-fold cross validation) → In other words, all in all, there will be 18 × 5 = **90 rounds of training!**
- When you have no idea what value a hyperparameter should have, a simple approach is to **try out consecutive powers of 10**
#### Randomized Search
- When the hyperparameter search space is large, it is often preferable to use RandomizedSearchCV instead
#### Analyze the Best Models and Their Errors
```python
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
```
#### Evaluate Your System on the Test Set
- Just get the predictors and the labels from your test set, run your `full_pipeline` to transform the data (**call transform(), not fit_transform(), you do not want to fit the test set!**) and evaluate the final model on the test set:
    ```python
    final_model = grid_search.best_estimator_
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = final_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse) # => evaluates to 47,730.2
    ```
- You might want to have an idea of how precise this estimate is. For this, you can compute a 95% confidence interval for the generalization error using `scipy.stats.t.interval()`:
    ```python
    from scipy import stats
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                             loc=squared_errors.mean(),
                             scale=stats.sem(squared_errors)))
    ```
- The performance will usually be slightly worse than what you measured using cross-validation if you did a lot of hyperparameter tuning (because your **system ends up fine-tuned to perform well on the validation data, and will likely not perform as well on unknown datasets**)


## Chapter 3. Classification
### Performance Measures
#### Confusion Matrix
- `cross_val_predict()` function:
    ```python
    from sklearn.model_selection import cross_val_predict
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_train_5, y_train_pred)
    ```
#### Precision/Recall Tradeoff
- How do you decide which threshold to use?
    ```python
    from sklearn.model_selection import cross_val_predict
    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

    from sklearn.metrics import confusion_matrix
    confusion_matrix(y_train_5, y_train_pred)
    
    y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                                 method="decision_function") # Similar to 'predict_proba()'
    
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
    
    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
        # [...]  highlight the threshold, add the legend, axis label and grid
        
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)    
    ```
- Another way to select a good precision/recall tradeoff is to plot precision directly against recall (PR-curve)


- As a rule of thumb, you should **prefer the PR curve to the ROC curve whenever the positive class is rare**
    - Looking at the previous ROC curve (and the ROC AUC score), you may think that the classifier is really good → But this is mostly because there are few positives compared to the negatives
    - In contrast, the **PR curve makes it clear that the classifier has room for improvement (the curve could be closer to the top-right corner)**


## Chapter 6. Decision Trees
### Regularization Hyperparameters
- To avoid overfitting the training data → You need to restrict the Decision Tree’s freedom during training
    - `max_depth` will regularize the model and thus reduce the risk of overfitting
    - `min_samples_split` (minimum number of samples a node must have before it can be split)
    - `min_samples_leaf` (minimum number of samples a leaf node must have
    - `min_weight_fraction_leaf` (same as `min_samples_leaf` but expressed as a fraction of the total number of weighted instances)
    - `max_leaf_nodes` (maximum number of leaf nodes)
    - `max_features` (maximum number of features that are evaluated for splitting at each node)
- **Increasing `min_*` hyperparameters or reducing `max_*` hyperparameters will regularize the model**
### Instability
    - Limitation → Trees follow orthogonal decision boundaries → Sensitive to "data rotation" → Overfitting → Overcome by combination with PCA
    - Main issue → Very sensitive to small variations in the training data
        - Stochastic algorithm → It randomly selects the set of features to evaluate at each node


## Chapter 8. Dimensionality Reduction
- Reducing dimensionality does lose some information:
    - So even though it will speed up training, it may also make your system perform slightly worse.
    - It also makes your pipelines a bit more complex and thus harder to maintain
    - **So you should first try to train your system with the original data before considering using dimensionality reduction if training is too slow**
        - In some cases, however, reducing the dimensionality of the training data may filter out some noise and unnecessary details and thus result in higher performance (but in general it won’t; it will just speed up training)
- Apart from speeding up training, dimensionality reduction is also extremely useful for data visualization:
    - Reducing the number of dimensions down to two (or three) makes it possible to plot a condensed view of a high-dimensional training set on a graph and often gain some important insights by visually detecting patterns, such as **clusters**
    - Moreover, DataViz is essential to **communicate your conclusions to people who are not data scientists**, in particular  decision makers who will use your results
### PCA        
- First it identifies the hyperplane that lies closest to the data, and then it projects the data onto it
#### Principal Components
- PCA identifies the axis that accounts for the largest amount of variance in the training set
    - It also finds a second axis, orthogonal to the first one, that accounts for the largest amount of remaining variance
- How can you find the principal components of a training set?
    - There is a standard matrix factorization technique called **Singular Value Decomposition (SVD)**
- PCA assumes that the dataset is centered around the origin: Scikit-Learn’s PCA classes take care of centering the data for you
#### Using Scikit-Learn
    ```python
    from sklearn.decomposition import PCA

    pca = PCA(n_components = 2)
    X2D = pca.fit_transform(X)

    # You can access the principal components using the 'components_ variable':
    first_PCA = pca.components_.T[:,0]
    ```
#### Explained Variance Ratio
- *Explained variance ratio* of each principal component → It indicates the **proportion of the dataset’s variance that lies along the axis** of each principal component
    ```python
    pca.explained_variance_ratio_
    ```
#### Choosing the Right Number of Dimensions
- It is generally preferable to **choose the number of dimensions** that add up to a sufficiently large portion of the variance (e.g., 95%)
    - Unless, of course, you are reducing dimensionality **for data visualization** —in that case you will generally want to reduce  the dimensionality down to **2 or 3**.
    ```python
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_train)
    ```
- Another option is to plot the explained variance as a function of the number of dimensions
    - There will usually be an elbow in the curve, where the explained variance stops growing fast → **intrinsic dimensionality** of the dataset
- Decompress the reduced dataset: applying the inverse transformation of the PCA projection
    ```python
    pca = PCA(n_components = 154)
    X_reduced = pca.fit_transform(X_train)
    X_recovered = pca.inverse_transform(X_reduced)
    ```
