
#### Contributors:

Ryan Lindberg (lindbergryan04@gmail.com) and Ethan Haus (ethanhaus@gmail.com)

---

## Introduction

Since transferring to UC San Diego, our cooking habits have taken a hit. With busy schedules and limited time, we've started prioritizing convenience over nutrition, which is something we’d like to change. The goal of this project is to find a way to optimize recipe selection, so we can cook meals that are tasty, quick to prepare, and nutritionally balanced without spending hours in the kitchen.

To do this, we analyzed a large dataset of 234,429 recipes with 15 features and built a machine learning model to predict how highly a recipe will be rated (out of 5 stars). There’s no shortage of recipes available online, but sorting through them to figure out which are actually worth making can be a pain—especially if a recipe has no reviews. A good prediction model could fill in the gaps, helping users filter through recipes they've saved and prioritize the ones that are most likely to be good.

Our approach started with exploratory data analysis (EDA) to figure out which recipe features were actually useful for predicting ratings. We focused on cooking time, number of steps, and key nutritional values (calories, sugar, sodium, protein, and saturated fat). We also introduced a new metric: the balance score, which takes into account taste (rating), effort (prep time & complexity), and healthiness to give each recipe a holistic score.

To predict rating categories, we trained a Random Forest classifier to categorize recipes as low (≤4.6 stars), medium (4.6-4.99 stars), or high (5 stars). This allows us to generate predicted ratings for new recipes, including our own, so we can calculate balance scores for them and make better cooking decisions.

This project utilizes EDA, feature engineering, statistical hypothesis testing, and machine learning to develop a practical tool for smarter recipe selection. With this, we can finally answer the question: What’s actually worth cooking?

---

## Cleaning and EDA


Our first step was to replace the ratings of '0' with np.nan. We then created an 'avg_rating' column to represent the average rating for each recipe. We did this by grouping by each recipeID, since most recipes had multiple reviews, then we simply took the mean after grouping. Then, we dropped some columns we weren't going to use. These ended up being 'review', 'contributor_id', 'user_id', 'date', 'submitted', 'description', and 'steps'. After, we looked at all the columns with string lists, and converted them to lists. Our plan was then to take these lists, and turn the list values into individual columns. We ended up only needing to do this with the 'nutrition' column, and turned all the different elements of 'nutrition' into their own columns. Since we'd extracted all that we needed from 'nutrition' we dropped the column. Our data ended up looking like this, minus the 'tags','ingredients','total_fat' and 'carbs' columns, which we couldn't include in the report in order for our dataframe to horizontally fit the page.

| name                                 |   minutes |   n_steps |   avg_rating |   calories |   sugar |   sodium |   protein |   sat_fat |
|:-------------------------------------|----------:|----------:|-------------:|-----------:|--------:|---------:|----------:|----------:|
| 1 brownies in the world    best ever |        40 |        10 |            4 |      138.4 |      50 |        3 |         3 |        19 |
| 1 in canada chocolate chip cookies   |        45 |        12 |            5 |      595.1 |     211 |       22 |        13 |        51 |
| 412 broccoli casserole               |        40 |         6 |            5 |      194.8 |       6 |       32 |        22 |        36 |
| 412 broccoli casserole               |        40 |         6 |            5 |      194.8 |       6 |       32 |        22 |        36 |
| 412 broccoli casserole               |        40 |         6 |            5 |      194.8 |       6 |       32 |        22 |        36 |

For our Univariate Analysis we wanted more insight on our 'minutes' and 'ratings' columns. Since our average ratings were so high, we wanted to see a distribution of all the values. And since we figured that the amount of effort would heavily factor into people's perceptions of the recipe and therefore their rating, we felt effort was best quantified by how long it took to complete the recipe, ergo 'minutes' column. So we decided to view the distribution to see how to best weight the feature. 

<iframe src="assets/eda-minutes.html" width="800" height="400" frameBorder="0"></iframe>

As you can see in the graph above, the graph is skewed right EDIT, with the majority of our values normally distributed around a rough mean of 30 minutes. There are definetely some outlier recipes that take much longer, but it's clear to see that most of our recipes are around 10 minutes to an hour. 

When initially starting this project, we were looking to see if any columns were correlated with each other. We thought this would be great to know in helpig us find columns with the best predictive power for ratings. 

<iframe src="assets/eda-heatmap5.html" width="800" height="600" frameBorder="0"></iframe>


As you can see, there was nothing that had a strong correlation with 'rating' or 'avg_rating' other than... 'rating' and 'avg_rating'. Our strongest was the minutes column which had a whopping $R^2$ of 0.01. At least it was positive! Upon seeing this, we knew we needed to create and explore some aggregate statistics. 


We made a pivot table that took the mean and count of the 'avg_rating' column for every value of 'n_steps', which we used as our index. We were hoping to find some sort of pattern between the number of steps and rating. Maybe people tended to rate recipes higher that required less work? I probably would. Or maybe it was the opposite case that the more you put into it, the better the result? Nope. Score one for nihilism. We didn't see any clear trends in the 'avg_rating' mean, although it's hard to judge because of our distribution of recipe's 'n_steps' has the vast majority of it's contents between 5 steps and 10 steps, which leaves us with inconsistent standards for the 'avg_rating'.

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

The rating column has 15036 missing values. This is due to the fact that some users did not leave a 1-5 rating with their review, causing it to default to 0. These 0 values were replaced with np.nan during the cleaning process. This might look like NMAR because the value of the missing rating could be dependent on the missing rating itself. This missingness is common with reviews and ratings, and is known as 'review bias,' which is the phenomenon that describes how people tend to only leave ratings when they have a strong opinion on something. These users who left no rating likely had a neutral opinion on the recipe, and therefore didn't feel compelled to give it a rating. At the same time though, we found that the missingness of rating was actually dependent on other columns of the dataset. This means that the rating column is actually MAR and not NMAR. 

<iframe src="assets/missingness_plot1.html" width="800" height="600" frameBorder="0"></iframe>

Observed Difference in Mean Step Count: 1.1535

P-value: 0.00

Lets look at the kde plots:

<iframe src="assets/missingness_plot2.html" width="800" height="600" frameBorder="0"></iframe>
<iframe src="assets/missingness_plot3.html" width="800" height="600" frameBorder="0"></iframe>

It appears that the missingness of rating may be dependent on the number of steps in a recipe! This could be due to many reasons. If we consider the fact that these are users who *did* leave a review, but *didn't* leave a rating, the true reason for this relationship is harder to track down. Users may feel that they didn't do the long and complex recipe justice, and therefore be reluctant to leave a rating because they don't trust their judgement. Or perhaps they had a higher expectation of the complex recipe, and were reluctant to show that it didn't turn out as well as they had hoped. Or perhaps they figured 'I've been writing this stupid review for 10 minutes, I'm done'.

<iframe src="assets/missingness_plot4.html" width="800" height="600" frameBorder="0"></iframe>

Observed Difference in Mean minutes: 51.4524

P-value: 0.1370

Rating missingness does not seem to be dependent on the minutes column.

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

<iframe src="assets/hypothesis_plot2.html" width="800" height="600" frameBorder="0"></iframe>

We got a P-Value of .0048, which definetely falls under our significance level. Therefore, we will reject our null hypothesis in favor of the alternative. This suggests that our formula for calculating a holistic balance score for each recipe is generalizable to a population with similar values of taste vs ease vs health in their cooking. If our algorithm can predict how people value the 'balance' of certain recipes, then this bodes well for our ability to predict how they'd holistically rate them. 

---

## Framing a Prediction Problem

As cool as our balance score feature from the hypothesis test was, we sadly cannot use it for our prediction. We ultimately want to predict how a user would rate a recipe. Our balance score could be a great feature for this, but it usees rating category to quantify the 'taste' of an recipe. We can't use rating to predict rating Back to the drawing board! We noticed that the vast majority of our reviews were 5 stars, and those that weren't, were 4 stars. This lead us to framing a new classification problem. We decided to create 3 seperate bins for rating, and try to predict the class of each recipe in terms of one of these bins. We have low (<= 4.6 stars), medium (4.6-4.99 stars), and high (5 stars). Since we have three buckets, this is multiclass classification. We think a good model for this task would be a random forest from SKLearn. 

We are chose to predict the rating category of each recipe because that's what struck us as 'coolest' and most useful, since we could use these predicted ratings to calculate balance scores for our own recipes, which have never been rated before. We used features like the time the recipe takes, the number of steps in the recipe, and the nutritional information of the recipe. We are chose to evaulate our model based on accuracy, simce the goal of our prediction is to assign a rating category as accurately as possible, and we want our predicted ratings to best reflect to the real rating. Minimizing false positives or false negatives is not the greatest priority, since this is a low stakes prediction task. 

---

## Baseline Model

For our base random forest model, we used the 'minutes', 'n_steps', 'n_ingredients', 'calories', 'sugar', 'carbs', 'sat_fat', 'sodium', and 'protein' features. These are all discrete quantitative features. We used 100 trees per forest, with a random state of 10, so that we can recreate the same results if necessary. We used a test/train split of 80/20, meaning that we trained on 80% of our data and tested the accuracy on the following 20%, again using a random state of 10 for reproducability in our final model. Our model had a classification accuracy of 57%, meaning we correctly identified the bucket the recipe fell into at 57% of the time. This isn't great, but it's much better than randomly assigning them, which would yield a 33% accuracy. Therefore, we feel that this baseline model is good, considering the lack of correlation between rating and other features present in this dataset. For those reasons, we feel it does a good job.

--

## Final Model

When it came time to craft our final, new and improved model, we started by engineered some new features in hopes of better accuracy. We performed log, square, and quadratic tranformations on each quantitative feature present in our base model, in hopes that these new features might help our model better distinguish between the 3 classes. As a performance metric for these new features, we used their F score, which tells us how well these features distinguish between the classes. F scores being the variance within groups divided by the variance between groups. At this point we had many features, and we had to cut down on them to avoid overfitting to our data, and ensure we weren't adding redundent or irrelevant features that would increase the variance of the model. We chose the 17 top features with the best F scores, and instead of hand picking through them ourselves, we utilizeded SKLearn's random forest model with built in recursive feature elimination. This works by recursively removing the least important feature until we reached our target number. We set our preferred # of features to 10, and ended up with the following optomized features: log(protein), log(minutes), log(sugar), log(sodium), log(calories), n_steps * n_ingredients, n_ingredients * sat_fat, n_steps * sat_fat, calories, sugar.

Next, we needed to optimize model hyperparameters. To do this, we used SKLearns GridSearchCV, which used cross fold validation to test 54 combinations of hyperparameters. The hyperparameters we tested were n_estimators, max_features, min_samples_split, and min_samples_leaf, with bootstrap enabled. The optimal hyperparamters we found were max_features = sqrt, classifier__min_samples_leaf = 4, min_samples_split = 10, n_estimators = 300.

Finally, we put it all together into one final model, using bootstrapping, 

