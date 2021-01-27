### [Classification Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

- Binary classification (only):
    - [precision_recall_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve): Compute precision-recall pairs for different probability thresholds.
        - The precision is the ratio `tp / (tp + fp)` where `tp` is the number of true positives and `fp` the number of false positives. The **precision** is intuitively the ability of the classifier **not to label as positive a sample that is negative**.
        - The recall is the ratio `tp / (tp + fn)` where `tp` is the number of true positives and `fn` the number of false negatives. The **recall** is intuitively the ability of the classifier to **find all the positive samples**.
    - [roc_curve](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve): A receiver operating characteristic (ROC), or simply ROC curve, is a graphical plot which illustrates the performance of a binary classifier system as its discrimination threshold is varied. It is created by plotting the fraction of true positives out of the positives (TPR = true positive rate) vs. the fraction of false positives out of the negatives (FPR = false positive rate), at various threshold settings. TPR is also known as sensitivity, and FPR is one minus the specificity or true negative rate.
        
- Multi-class classification (or binary):
    - [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix): Compute confusion matrix to evaluate the accuracy of a classification.
    ![IMG_Confusion_Matrix](https://scikit-learn.org/stable/_images/sphx_glr_plot_confusion_matrix_0011.png)
    - [roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score): Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

- Multi-label classification (or binary or multi-class):
    - [accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score): Accuracy classification score.
        - It is the ratio of number of correct predictions to the total number of input samples.
        - **It works well only if there are equal number of samples belonging to each class.**
    - [classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report): Build a text report showing the main classification metrics.
    - [f1_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score): Compute the F1 score, also known as balanced F-score or F-measure.
        - The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal: `F1 = 2 * (precision * recall) / (precision + recall)`
    - [precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)
    - [recall_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)
    
### Assessment Metrics
- #### F1-score
    - What does a high F1 score mean? It suggests that both the precision and recall have high values — this is good and is what you would hope to see upon generating a well-functioning classification model on an imbalanced dataset. A low value indicates that either precision or recall is low, and maybe a call for concern
    - Good F1 scores are generally lower than good accuracies (in many situations, an **F1 score of 0.5 would be considered pretty good**)

- #### Receiver Operating Characteristic (ROC) Curve
   - Depending on your application, you may be very averse to false positives as they may be very costly (e.g. launches of nuclear missiles) and thus **would like a classifier that has a very low false-positive rate**.

- #### Area Under Curve (AUC)
   - If a particular classifier has an ROC of 0.6 and another has an ROC of 0.8, the latter is clearly a better classifier. The AUC has the benefit that it is independent of the decision criteria — the classification threshold — and thus makes it easier to compare these classifiers.
   
-------------   
 
### Which metric should be used then?
#### RECALL could be a potentially strong metric for this case; "from all the flights classified as delayed, the actual (true) number of delayed flights is as high as possible."
#### Bear in mind that it's a clear case of imbalanced data

[A Gentle Introduction to Imbalanced Classification](https://machinelearningmastery.com/what-is-imbalanced-classification/)  
**Imbalanced classifications** pose a challenge for predictive modeling as **most of the machine learning algorithms used for classification were designed around the assumption of an equal number of examples for each class**. This results in models that have poor predictive performance, specifically for the minority class. This is a problem because typically, the minority class is more important and therefore **the problem is more sensitive to classification errors for the minority class than the majority class**.

[Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset](https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/)  
1. Try Changing Your Performance Metric:
    - Confusion Matrix
    - Precision: 
    - Recall: 
    - F1 Score
    - Kappa
    - ROC Curves
2. Try Resampling Your Dataset:
    - You can add copies of instances from the under-represented class called **over-sampling** (or more formally **sampling with replacement**) → when you don’t have a lot of data (tens of thousands of records or less)
    - You can delete instances from the over-represented class, called **under-sampling** → when you have an a lot data (tens- or hundreds of thousands of instances or more)
3. Try Different Algorithms:
    - That being said, **decision trees often perform well on imbalanced datasets**. The splitting rules that look at the class variable used in the creation of the trees, can force both classes to be addressed.  
    If in doubt, try a few popular decision tree algorithms like C4.5, C5.0, CART, and Random Forest.
    
[Guide to Classification on Imbalanced Datasets](https://towardsdatascience.com/guide-to-classification-on-imbalanced-datasets-d6653aa5fa23)  
There are two main types of techniques to handle imbalanced datasets:
- ### Sampling methods:
    - #### Oversampling
        - How do we generate these samples? The most common way is to generate points that are close in dataspace proximity to existing samples or are ‘between’ two samples, as illustrated below
        - There are some downsides to adding false data points:
            - **Overfitting** risk
            - In addition, adding these values randomly can also contribute **additional noise to our model**
        - Techniques:
            - **SMOTE** (Synthetic minority oversampling technique) → SMOTE generates new samples in between existing data points based on their local density and their borders with the other class. Algorithm:
                - Find its k-nearest minority neighbours
                – Randomly select j of these neighbours
                – Randomly generate synthetic samples along the lines joining the minority sample and its j selected neighbours (j depends on the amount of oversampling desired)
    - #### Undersampling
        - Is undersampling a good idea? Undersampling is recommended by many statistical researchers but is **only good if enough data points are available on the undersampled class**
        - Is undersampling a good idea? Undersampling is recommended by many statistical researchers but is only good if enough data points are available on the undersampled class
- ### Cost-sensitive methods
    - #### Upweighting
    Upweighting is analogous to over-sampling and works by increasing the weight of one of the classes keeping the weight of the other class at one
    - #### Down-weighting
    Down-weighting is analogous to under-sampling and works by decreasing the weight of one of the classes keeping the weight of the other class at one

```python
# Example of how to implement cost-sensitive learning:
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
model.fit(X_train, y_train, class_weight=class_weights)
```

- Benefits of cost-sensitive learning:
    - It is much simpler to implement
    - Easier to communicate to individuals

-------------

### Case study: Flight Delays

Nomenclature:
- Delayed = Positive
- On-time = Negative 

Considering this:
- False positive (Type I error) → Wrongly classifying an On-time flight as a Delayed flight → Not significantly relevant
- **False negative (Type II error) → Wrongly classifying a Delayed flight as an On-time flight → Highly relevant**

**F-beta** score (\$ F_\beta \$):
![F-beta score](https://wikimedia.org/api/rest_v1/media/math/render/svg/136f45612c08805f4254f63d2f2524bc25075fff)

Two commonly used values for β are:
- **2 : weighs recall higher than precision**
- 0.5 : weighs recall lower than precision.


<em>Probably most people in the industry would accept that an **OTP of 80%* or above is pretty good***. That’s 4 in 5 flights arriving within 15 minutes of their scheduled arrival time. The very best airlines and airports succeed in punctuality closer to 90% - but they remain the exception rather than the rule.</em>  
(Source: [OAG](https://www.oag.com/on-time-performance-airlines-airports))

The actual data from the 7268232 records comprising the OTP dataset accurately confirm this hypothesis:
```
Delays: 5878979 (80.89%)
On-time: 1389253 (19.11%)
```

In some rare cases, the calculation of Precision or Recall can cause a division by 0. Regarding the precision, this can happen if there are no results inside the answer of an annotator and, thus, the true as well as the false positives are 0. For these special cases, we have defined that **if the true positives, false positives and false negatives are all 0, the precision, recall and F1-measure are 1**. This might occur in cases in which the gold standard contains a document without any annotations and the annotator (correctly) returns no annotations. **If true positives are 0 and one of the two other counters is larger than 0, the precision, recall and F1-measure are 0.**


