
#### Contributors:

Ryan Lindberg (lindbergryan04@gmail.com) and Ethan Haus (ethanhaus@gmail.com)

---

## Introduction

Since transferring to UC San Diego, our cooking habits have taken a hit. With busy schedules and limited time, we've started prioritizing convenience over nutrition, which is something we’d like to change. The goal of this project is to find a way to optimize recipe selection, so we can cook meals that are tasty, quick to prepare, and nutritionally balanced without spending hours in the kitchen.

To do this, we analyzed a large dataset of 234,429 recipe reviews from Food.com with 15 features and built a machine learning model to predict how highly a recipe will be rated (out of 5 stars). There’s no shortage of recipes available online, but sorting through them to figure out which are actually worth making can be a pain, especially if a recipe has no reviews. A good prediction model could fill in the gaps, helping users filter through recipes they've saved and prioritize the ones that are most likely to be good.

Our approach started with exploratory data analysis (EDA) to figure out which recipe features were actually useful for predicting ratings. We focused on cooking time, number of steps, and key nutritional values (calories, sugar, sodium, protein, and saturated fat). We also introduced a new metric: the balance score, which takes into account taste (rating), effort (prep time & complexity), and healthiness to give each recipe a holistic score.

To predict rating categories, we trained a Random Forest classifier to categorize recipes as low (≤4.6 stars), medium (4.6-4.99 stars), or high (5 stars). This allows us to generate predicted ratings for new recipes, including our own, so we can calculate balance scores for them and make better cooking decisions.

This project utilizes EDA, feature engineering, statistical hypothesis testing, and machine learning to develop a practical tool for smarter recipe selection. With this, we can finally answer the question: What’s actually worth cooking?

---

## Cleaning and EDA


Our first step in cleaning the data was to replace the ratings of '0' with np.nan. We then created an 'avg_rating' column to represent the average rating for each recipe. We did this by grouping by each recipeID, since most recipes had multiple reviews, then we simply took the mean after grouping. 

We dropped the duplicate reviews, but kept a copy of the dataframe with duplicates for the missingness section. 

Then, we dropped some columns we weren't going to use. These ended up being 'review', 'contributor_id', 'user_id', 'date', 'submitted', 'description', and 'steps'. 

Next, we took the nutrition column which contained a list of nutritional information for the recipe in the form: [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)] (PDV stands for “percentage of daily value”), and we converted each value into its own column for each recipe. 

Since we'd extracted all that we needed from 'nutrition' we dropped the column. Our data ended up looking like this, minus the 'tags', 'ingredients', 'total_fat', and 'carbs' columns, which we couldn't include in the report in order for our dataframe to horizontally fit the page.

| name                                 |   minutes |   n_steps |   avg_rating |... |   calories |   sugar |   sodium |   protein |   sat_fat |
|:-------------------------------------|----------:|----------:|-------------:|---:|-----------:|--------:|---------:|----------:|----------:|
| 1 brownies in the world    best ever |        40 |        10 |            4 |... |      138.4 |      50 |        3 |         3 |        19 |
| 1 in canada chocolate chip cookies   |        45 |        12 |            5 |... |      595.1 |     211 |       22 |        13 |        51 |
| 412 broccoli casserole               |        40 |         6 |            5 |... |      194.8 |       6 |       32 |        22 |        36 |
| 412 broccoli casserole               |        40 |         6 |            5 |... |      194.8 |       6 |       32 |        22 |        36 |
| 412 broccoli casserole               |        40 |         6 |            5 |... |      194.8 |       6 |       32 |        22 |        36 |


For our univariate analysis we wanted more insight on our 'minutes' columns. 
We figured that the amount of effort would heavily factor into people's perceptions of the recipe and therefore their rating, so understanding the distribution of minutes was essential to get a good idea of the recipes we were working with.

<iframe src="assets/eda-minutes.html" width="800" height="420" frameBorder="0"></iframe>

As you can see in the graph above, the distribution has a right skew, with the majority of our values distributed around a rough mean of 30 minutes. There are definitely some outlier recipes that take much longer, but it's clear to see that most of our recipes are around 10 minutes to an hour. These shorter recipes are defniitely preferred in terms of our cooking preferences, so its good to know we have plenty in our dataset.

Moving forward, we wanted to know, do any columns have a relationship with average rating? Knowing this would be a great help for finding columns with the best predictive power for ratings. To answer this question, we created a correlation heatmap of every our column:

<iframe src="assets/eda-heatmap5.html" width="830" height="780" frameBorder="0"></iframe>


As you can see, there not a single column had a strong correlation with 'rating' or 'avg_rating' other than... 'rating' and 'avg_rating'. Our strongest correlation was the calories column which had a whopping correlation coefficient of -0.05. Upon seeing this, we knew we needed to create and explore some aggregate statistics. Maybe the correlation was nonlinear? This prediction task was looking to be pretty difficult thus far. 


We made a pivot table that took the mean and count of the 'avg_rating' column for every value of 'n_steps', which we used as our index. We were hoping to find some sort of pattern between the number of steps and rating. Maybe people tended to rate recipes higher that required less work? We probably would. Or maybe it was the opposite case that the more you put into it, the better the result? Nope. Score one for nihilism. We didn't see any clear patterns between number of steps and average rating.

|   n_steps |   ('mean', 'avg_rating') |   ('count', 'avg_rating') |
|----------:|-------------------------:|--------------------------:|
|         1 |                  4.66336 |                        57 |
|         2 |                  4.72977 |                       157 |
|         3 |                  4.72558 |                       258 |
|         4 |                  4.70076 |                       287 |
|         5 |                  4.71929 |                       367 |
|         6 |                  4.64308 |                       382 |
|         7 |                  4.68536 |                       433 |
|         8 |                  4.69462 |                       402 |
|         9 |                  4.64203 |                       411 |
|        10 |                  4.64816 |                       348 |
|        11 |                  4.65225 |                       292 |
|        12 |                  4.64835 |                       256 |
|        13 |                  4.73724 |                       217 |
|        14 |                  4.72694 |                       183 |
|        15 |                  4.67894 |                       144 |
|        16 |                  4.72795 |                       135 |
|        17 |                  4.75346 |                       121 |
|        18 |                  4.57944 |                        76 |
|        19 |                  4.70668 |                        70 |
|        20 |                  4.72231 |                        66 |



---

## Assessment of Missingness

The rating column has 15036 missing values. This is due to the fact that some users did not leave a 1-5 rating with their review, causing it to default to 0. 
These 0 values were replaced with np.nan during the cleaning process. This might look like NMAR because the value of the missing rating could be dependent on the missing rating itself.
This missingness is common with reviews and ratings, and is known as 'review bias,' which is the phenomenon that describes how people tend to only leave ratings when they have a strong opinion on something.
These users who left no rating likely had a neutral opinion on the recipe, and therefore didn't feel compelled to give it a rating.

At the same time though, we found that the missingness of rating was actually dependent on other columns of the dataset. This means that the rating column is actually MAR and not NMAR.

Here is our permutation test to assess missingness dependency between rating and n_steps:

<iframe src="assets/missingness_plot1.html" width="800" height="420" frameBorder="0"></iframe>

Observed Difference in Mean Step Count: 1.1535

P-value: 0.00

Lets look at the kernel density estimation plots to visualize this dependency.

<iframe src="assets/missingness_plot2.html" width="800" height="420" frameBorder="0"></iframe>
<iframe src="assets/missingness_plot3.html" width="800" height="420" frameBorder="0"></iframe>

The missingness of rating seems to be dependent on the number of steps in a recipe! This could be due to many reasons. If we consider the fact that these are users who *did* leave a review, but *didn't* leave a rating, the true reason for this relationship is harder to track down. Users may feel that they didn't do the long and complex recipe justice, and therefore be reluctant to leave a rating because they don't trust their judgement. Or perhaps they had a higher expectation of the complex recipe, and were reluctant to show that it didn't turn out as well as they had hoped. Or maybe they figured 'I've been writing this stupid review for 10 minutes, I'm done'.

We performed a second permutation test to assess missingness dependency between rating and minutes:

<iframe src="assets/missingness_plot4.html" width="800" height="420" frameBorder="0"></iframe>

Observed Difference in Mean minutes: 51.4524

P-value: 0.1370

Rating missingness did not seem to be dependent on the minutes column.

---

## Hypothesis Testing

Before we begin our hypothesis test, we will first define what we call a balance score. The balance score is a rating that we give each recipe based on its balance between time to cook, taste, and nutritional profile. We will using recipe rating as a proxy for taste.

We will manually define a formula to calculate this balance score, which is a score from 1-10, with 10 being the most healthy, easy to cook, and protein dense meals, and 1 being unhealthy, hard to cook, and less nutritional meals.

This score is mostly subjective, as we are setting the weights ourselves. This is okay though, because our goal is to plug in our own recipes, and find out if its a recipe we would like to cook.

### Weight Assignments:
- **W_RATING_HIGH** = 0.4  
- **W_RATING_MEDIUM** = 0.2  
- **W_RATING_LOW** = 0.0  
- **W_TIME** = -0.01 (Penalizes long cooking times)  
- **W_N_STEPS** = -0.06 (Penalizes overly complex recipes)  
- **W_SAT_FAT** = -0.06 (Less saturated fat is better)  
- **W_CALORIES** = -0.01 (Penalizes excessive calories)  
- **W_PROTEIN** = 0.2 (More protein is better)  
- **W_CARBS** = -0.01 (Too many carbs reduce balance)  
- **W_SUGAR** = -0.04 (Excess sugar is penalized)  
- **W_SODIUM** = -0.03 (Excess sodium is penalized)  

---

### Balance Score Formula:

balance_score = (W_RATING_HIGH * rating_high) + (W_RATING_MEDIUM * rating_medium) + (W_RATING_LOW * rating_low) + (W_TIME * time_factor) + (W_N_STEPS * n_steps) + (W_SAT_FAT * sat_fat) + (W_CALORIES * calories) + (W_PROTEIN * protein) + (W_CARBS * carbs) + (W_SUGAR * sugar) + (W_SODIUM * sodium)

Note: We used a log decay on each feature to ensure no outliers.

The resulting dataframe:

|     id | name                                 | ... |   avg_rating | rating_category   | rating_low   | rating_medium   | rating_high   |   balance_score |
|-------:|:-------------------------------------|----:|-------------:|:------------------|:-------------|:----------------|:--------------|----------------:|
| 333281 | 1 brownies in the world    best ever | ... |            4 | Low               | True         | False           | False         |         1.53785 |
| 453467 | 1 in canada chocolate chip cookies   | ... |            5 | High              | False        | False           | True          |         8.25504 |
| 306168 | 412 broccoli casserole               | ... |            5 | High              | False        | False           | True          |         8.87765 |
| 286009 | millionaire pound cake               | ... |            5 | High              | False        | False           | True          |         8.40649 |
| 475785 | 2000 meatloaf                        | ... |            5 | High              | False        | False           | True          |         8.75722 |

Lets check out the distribution of our balance scores:
<iframe src="assets/hypothesis_plot1.html" width="800" height="600" frameBorder="0"></iframe>

Now that we have our balance scores calculated, lets perform our hypothesis test. For this section, we had a friend calculate balance scores for 30 random recipes by hand, using his own intuition, given the information that 'balance' in this context is a balance between time, healthiness, and rating as a proxy for taste. 

We would like to test if our algorithmically calculated balance scores reflect what someone with similar priorities in their recipes might score a recipe.

Our chosen statistic is the correlation between human balance scores and algorithmically calculated balance scores.

- Null Hypothesis (H0): 

There is no significant correlation between human calculated balance scores based off intuition and the algorithmically calculated balance scores. 

H(0) : p = 0

(Where p represents the correlation coefficient between human human balance scores and algorithm balance scores.)

- Alternative Hypothesis (HA):

​
There is a significant correlation between human calculated balance scores based off intuition and the algorithmically calculated balance scores. 

H(A) : p ≠ 0

Our p-value threshhold is .05.

<iframe src="assets/hypothesis_plot2.html" width="800" height="630" frameBorder="0"></iframe>

We got a P-Value of .0048, which definitely falls under our significance level. Therefore, we will reject our null hypothesis in favor of the alternative. This suggests that our formula for calculating a holistic balance score for each recipe is generalizable to a population with similar values of taste vs ease vs health in their cooking. If our algorithm can predict how people value the 'balance' of certain recipes, then this bodes well for our ability to predict how they'd holistically rate them. 

---

## Framing a Prediction Problem

As cool as our balance score feature from the hypothesis test was, we sadly cannot use it for our prediction. We ultimately want to predict how a user would rate a recipe. Our balance score could be a great feature for this, but it uses rating category to quantify the 'taste' of an recipe. We can't use rating to predict rating Back to the drawing board! We noticed that the vast majority of our reviews were 5 stars, and those that weren't, were 4 stars. This lead us to framing a new classification problem. We decided to create 3 separate bins for rating, and try to predict the class of each recipe in terms of one of these bins. We have low (<= 4.6 stars), medium (4.6-4.99 stars), and high (5 stars). Since we have three buckets, this is multiclass classification. We think a good model for this task would be a random forest from SKLearn. 

We are chose to predict the rating category of each recipe because that's what struck us as 'coolest' and most useful, since we could use these predicted ratings to calculate balance scores for our own recipes, which have never been rated before. We used features like the time the recipe takes, the number of steps in the recipe, and the nutritional information of the recipe. We chose to evaluate our model based on accuracy, since the goal of our prediction is to assign a rating category as accurately as possible, and we want our predicted ratings to best reflect to the real rating. Minimizing false positives or false negatives is not the greatest priority, since this is a low stakes prediction task. 

---

## Baseline Model

For our base random forest model, we used the 'minutes', 'n_steps', 'n_ingredients', 'calories', 'sugar', 'carbs', 'sat_fat', 'sodium', and 'protein' features. These are all discrete quantitative features. We used 100 trees per forest, with a random state of 10, so that we can recreate the same results if necessary. We used a test/train split of 80/20, meaning that we trained on 80% of our data and tested the accuracy on the following 20%, again using a random state of 10 for reproducability in our final model. Our model had a classification accuracy of 57%, meaning we correctly identified the bucket the recipe fell into at 57% of the time. This isn't great, but it's much better than randomly assigning them, which would yield a 33% accuracy. Therefore, we feel that this baseline model is good, considering the lack of correlation between rating and other features present in this dataset. For those reasons, we feel it does a good job.

--

## Final Model

When it came time to craft our final, new and improved model, we started by engineered some new features in hopes of better accuracy. We performed log, square, and quadratic transformations on each quantitative feature present in our base model, in hopes that these new features might help our model better distinguish between the 3 classes. As a performance metric for these new features, we used their F score, which tells us how well these features distinguish between the classes. F scores being the variance within groups divided by the variance between groups. At this point we had many features, and we had to cut down on them to avoid overfitting to our data, and ensure we weren't adding redundant or irrelevant features that would increase the variance of the model. We chose the 17 top features with the best F scores, and instead of hand picking through them ourselves, we utilizeded SKLearn's random forest model with built in recursive feature elimination. This works by recursively removing the least important feature until we reached our target number. We set our preferred # of features to 10, and ended up with the following optimized features: log(protein), log(minutes), log(sugar), log(sodium), log(calories), n_steps * n_ingredients, n_ingredients * sat_fat, n_steps * sat_fat, calories, sugar.

Next, we needed to optimize model hyperparameters. To do this, we used SKLearns GridSearchCV, which used cross fold validation to test 54 combinations of hyperparameters. The hyperparameters we tested were n_estimators, max_features, min_samples_split, and min_samples_leaf, with bootstrap enabled. The optimal hyperparamters we found were max_features = sqrt, classifier__min_samples_leaf = 4, min_samples_split = 10, n_estimators = 300.

Finally, we put it all together into one final model, using everything we had learned so far. We implemented feature subsetting, where, for each tree, we select a random subset of features that the tree will use to make decisions. This prevents the tree from relying too much on strong predictors, and increases the diversity among trees, leading to a more generalizable model. And finally, we added bootstrapping, or Bagging as its called, to make sure each tree is trained on a random bootstrapped sample of the training data. This also improves variance among trees, which leads to better generalization.

With all of these improvements, our final model had an accuracy of 58%, a whopping 1% improvement over our base model.

We suspect that this primarily due to the fact that our features really do not tell us much about the target, and our base model had already made use of almost all of the meaningful patterns present in the data. 
Another suspicion is the uneven distribution of rating categories. With such a large number of ratings in the ‘high’ category, the model may be biased to predict high in most cases. Currently about 60% of the recipes are rated ‘high’, 30% are rated ‘low’, and around 10% rated ‘medium’. We attempted to redefine our bins such that we had a 60/20/20 split, but found this didn’t affect the results. 

---

## Fairness Analysis  

To check if our model performs worse for certain groups, we compared the precision of predictions across the low, medium, and high rating categories. We ran a permutation test to determine whether the observed differences in precision were statistically significant.  

### Observed Statistic  
0.591

### Null Hypothesis (H₀)  
Our model is fair in terms of precision between rating groups. Any differences in precision between low, medium, and high categories are due to random chance, not systematic bias.  

The maximum difference in precision across rating categories is approximately zero:  
max(Precision) - min(Precision) ≈ 0 for all categories (low, medium, high).  

### Alternative Hypothesis (Hₐ)  
Our model is unfair, meaning there is a significant difference in precision across rating categories. The observed disparity is greater than what would be expected by chance.  
 
max(Precision) - min(Precision) > 0.10.  

Precision for a given category is calculated as:  
Precision = (True Positives in category) / (Predicted Positives in category)

### Results & Interpretation  

Observed Precision by Group:
'High': 0.591, 'Low': 0.387, 'Medium': 0.0

Observed Range of Precision Differences: 0.5914

Permutation Test p-value: 0.0010

Since the p-value is below 0.05, we reject the null hypothesis, meaning our model exhibits precision disparities across rating categories.  

The major issue here is that our model is very unfair toward medium rated recipes, it doesn’t predict the medium category at all! Those poor medium-rated recipes...  

Although this is bad news for the performance of our random forest classifier, this is a low-stakes prediction task, so a misclassified rating doesn’t have very tangible real-world consequences.  

---

## Thanks for checking out our project!
