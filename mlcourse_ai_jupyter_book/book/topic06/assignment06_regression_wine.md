<img src="https://habrastorage.org/webt/ia/m9/zk/iam9zkyzqebnf_okxipihkgjwnw.jpeg" />
    
**<center>[mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course** </center><br>
 
Author: [Yury Kashnitsky](https://www.linkedin.com/in/festline/). All content is distributed under the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

# <center> Assignment #6 (demo)
## <center>  Exploring OLS, Lasso and Random Forest in a regression task
    
**Same assignment as a [Kaggle Kernel](https://www.kaggle.com/kashnitsky/a6-demo-linear-models-and-rf-for-regression) + [solution](https://www.kaggle.com/kashnitsky/a6-demo-regression-solution).**    
    
<img src='../../img/wine_quality.jpg' width=30%>

**Fill in the missing code and choose answers in [this](https://docs.google.com/forms/d/1aHyK58W6oQmNaqEfvpLTpo6Cb0-ntnvJ18rZcvclkvw/edit) web form.**


```{code-cell} ipython3
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler
```

**We are working with UCI Wine quality dataset (no need to download it – it's already there, in course repo and in Kaggle Dataset).**


```{code-cell} ipython3
# for Jupyter-book, we copy data from GitHub, locally, to save Internet traffic,
# you can specify the data/ folder from the root of your cloned 
# https://github.com/Yorko/mlcourse.ai repo, to save Internet traffic
DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/master/data/"
```


```{code-cell} ipython3
data = pd.read_csv(DATA_PATH + "winequality-white.csv", sep=";")
```


```{code-cell} ipython3
data.head()
```


```{code-cell} ipython3
data.info()
```

**Separate the target feature, split data in 7:3 proportion (30% form a holdout set, use random_state=17), and preprocess data with `StandardScaler`.**


```{code-cell} ipython3
# y = None # you code here

# X_train, X_holdout, y_train, y_holdout = train_test_split # you code here
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform # you code here
# X_holdout_scaled = scaler.transform # you code here
```

## Linear regression

**Train a simple linear regression model (Ordinary Least Squares).**


```{code-cell} ipython3
# linreg = # you code here
# linreg.fit # you code here
```

**<font color='red'>Question 1:</font> What are mean squared errors of model predictions on train and holdout sets?**


```{code-cell} ipython3
# print("Mean squared error (train): %.3f" % # you code here
# print("Mean squared error (test): %.3f" % # you code here
```

**Sort features by their influence on the target feature (wine quality). Beware that both large positive and large negative coefficients mean large influence on target. It's handy to use `pandas.DataFrame` here.**

**<font color='red'>Question 2:</font> Which feature this linear regression model treats as the most influential on wine quality?**


```{code-cell} ipython3
# linreg_coef = pd.DataFrame # you code here
# linreg_coef.sort_values # you code here
```

## Lasso regression

**Train a LASSO model with $\alpha = 0.01$ (weak regularization) and scaled data. Again, set random_state=17.**


```{code-cell} ipython3
# lasso1 = Lasso # you code here
# lasso1.fit # you code here
```

**Which feature is the least informative in predicting wine quality, according to this LASSO model?**


```{code-cell} ipython3
# lasso1_coef = pd.DataFrame # you code here
# lasso1_coef.sort_values # you code here
```

**Train LassoCV with random_state=17 to choose the best value of $\alpha$ in 5-fold cross-validation.**


```{code-cell} ipython3
# alphas = np.logspace(-6, 2, 200)
# lasso_cv = LassoCV # you code here
# lasso_cv.fit # you code here
```


```{code-cell} ipython3
# lasso_cv.alpha_
```

**<font color='red'>Question 3:</font> Which feature is the least informative in predicting wine quality, according to the tuned LASSO model?**


```{code-cell} ipython3
# lasso_cv_coef = pd.DataFrame # you code here
# lasso_cv_coef.sort_values # you code here
```

**<font color='red'>Question 4:</font> What are mean squared errors of tuned LASSO predictions on train and holdout sets?**


```{code-cell} ipython3
# print("Mean squared error (train): %.3f" % # you code here
# print("Mean squared error (test): %.3f" % # you code here
```

## Random Forest

**Train a Random Forest with out-of-the-box parameters, setting only random_state to be 17.**


```{code-cell} ipython3
# forest = RandomForestRegressor # you code here
# forest.fit # you code here
```

**<font color='red'>Question 5:</font> What are mean squared errors of RF model on the training set, in cross-validation (cross_val_score with scoring='neg_mean_squared_error' and other arguments left with default values) and on holdout set?**


```{code-cell} ipython3
# print("Mean squared error (train): %.3f" % # you code here
# print("Mean squared error (cv): %.3f" % # you code here
# print("Mean squared error (test): %.3f" % # you code here
```

**Tune the `max_features` and `max_depth` hyperparameters with GridSearchCV and again check mean cross-validation MSE and MSE on holdout set.**


```{code-cell} ipython3
# forest_params = {'max_depth': list(range(10, 25)),
#                  'min_samples_leaf': list(range(1, 8)),
#                  'max_features': list(range(6,12))}

# locally_best_forest = GridSearchCV # you code here
# locally_best_forest.fit # you code here
```


```{code-cell} ipython3
# locally_best_forest.best_params_, locally_best_forest.best_score_
```

**<font color='red'>Question 6:</font> What are mean squared errors of tuned RF model in cross-validation (cross_val_score with scoring='neg_mean_squared_error' and other arguments left with default values) and on holdout set?**


```{code-cell} ipython3
# print("Mean squared error (cv): %.3f" % # you code here
# print("Mean squared error (test): %.3f" % # you code here
```

**Output RF's feature importance. Again, it's nice to present it as a DataFrame.**<br>
**<font color='red'>Question 7:</font> What is the most important feature, according to the Random Forest model?**


```{code-cell} ipython3
rf_importance = pd.DataFrame  # you code here
rf_importance.sort_values  # you code here
```

**Make conclusions about the perdormance of the explored 3 models in this particular prediction task.**
