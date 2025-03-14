
by Ryan Lindberg (lindbergryan04@gmail.com) and Ethan Haus (ethanhaus@gmail.com)

---

## Introduction

In this project, we used a large dataset of recipes, and tried use their various features to train a model that could predict how a new recipe would be rated (out of 5 stars). There's no shortage of recipe's available online, we felt that this could offer a nice way for users to be able to filter threw some they saved that sounded good, and see what was actually worth making, if they didn't have any feedback.  Our model predicted the recipes based on the numerical columns that we used actually utilized. Our dataset had 234429 rows and 15 columns, of which we used 4. We also created a new column, avg_rating, which was used as well. We used the minutes, total minutes required to complete the recipe, and the n_steps, or number of steps in the recipe, with various weights to quantify how difficult it was to make a recipe. We used certain elements of the nutrition column (calories, sugar, sodium, protein and saturated fat), a column which provided nutritional information, to quantify how healthy each recipe is. Using these two stats, and factoring in the average rating, made a balance score, which we thought of as a holistic evaluation of a recipe. We used a random forsest network to calculate the optimal hyperparameters for our model

---

## Cleaning and EDA


Our first step was to replace the ratings of '0' with np.nan. We then created an 'avg_rating' column to represent the average rating of each recipe. We did this by grouping by each recipeID, since most recipes had multiple reviews, then we simply took the mean after grouping. Then, we dropped some columns we weren't going to use. These ended up being 'review', 'contributor_id', 'user_id', 'date', 'submitted', 'description', and 'steps'. After, we looked at all the columns with string lists, and converted them to lists. Our plan was then to take these lists, and turn the list values into individual columns. We ended up only needing to do this with the 'nutrition' column, and turned all the different elements of 'nutrition' into their own columns. Since we'd extracted all that we needed from 'nutrition' we dropped the column. Our data ended up looking like this, minus the 'tags','ingredients','total_fat' and 'carbs' columns, which we couldn't include in the report in order for our dataframe to horizontally fit the page.

| name                                 |   minutes |   n_steps |   avg_rating |   calories |   sugar |   sodium |   protein |   sat_fat |
|:-------------------------------------|----------:|----------:|-------------:|-----------:|--------:|---------:|----------:|----------:|
| 1 brownies in the world    best ever |        40 |        10 |            4 |      138.4 |      50 |        3 |         3 |        19 |
| 1 in canada chocolate chip cookies   |        45 |        12 |            5 |      595.1 |     211 |       22 |        13 |        51 |
| 412 broccoli casserole               |        40 |         6 |            5 |      194.8 |       6 |       32 |        22 |        36 |
| 412 broccoli casserole               |        40 |         6 |            5 |      194.8 |       6 |       32 |        22 |        36 |
| 412 broccoli casserole               |        40 |         6 |            5 |      194.8 |       6 |       32 |        22 |        36 |

For our Univariate Analysis we wanted more insight on our 'minutes' and 'ratings' columns. Since our average ratings were so high, we wanted to see a distribution of all the values. And since we figured that the amount of effort would heavily factor into people's perceptions of the recipe and therefore their rating, we felt effort was best quantified by how long it took to complete the recipe, ergo 'minutes' column. So we decided to view the distribution to see how to best weight the feature. 

<iframe src="assets/eda-minutes.html" width="800" height="600" frameBorder="0"></iframe>

As you can see in the graph above, the graph is skewed right EDIT, with the majority of our values normally distributed around a rough mean of 30 minutes. There are definetely some outlier recipes that take much longer, but it's clear to see that most of our recipes are around 10 minutes to an hour. 

When initially starting this project, we were looking to see if any columns were correlated with each other. We thought this would be great to know in helpig us find columns with the best predictive power for ratings. 

<iframe src="assets/eda-heatmap4.html" width="800" height="600" frameBorder="0"></iframe>


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

It appears that the missingness of rating may be dependent on the number of steps in a recipe! This could be due to many reasons. If we consider the fact that these are users who *did* leave a review, but *didn't* leave a rating, the true reason for this relationship is harder to track down. Users may feel that they didn't do the long and complex recipe justice, and therefore be reluctant to leave a rating because they don't trust their judgement. Or perhaps they had a higher expectation of the complex recipe, and were reluctant to show that it didn't turn out as well as they had hoped.

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


Now that we have our balance score, lets perform our hypothesis test. For this section, we had a friend calculate 30 random recipes by hand, using his own intuition, given the information that 'balance' is a defined as a balance between time, healthiness, and rating as a proxy for taste. 

We would like to test if our algorithmically calculated balance scores reflect what someone with similar priorities in their diet might score a recipe.

Our test statistic is the correlation between human balance scores and algorithmically calculated balance scores.

- Null Hypothesis (H0): 

There is no significant correlation between human calculated balance scores based off intuition and the algorithmically calculated balance scores. 

H(0) : p = 0

(Where p represents the correlation coefficient between human human balance scores and algorithm balance scores.)

- Alternative Hypothesis (HA):
​
There is a significant correlation between human calculated balance scores based off intuition and the algorithmically calculated balance scores. 

H(A) : p ≠ 0$

Our p-value threshhold is .05.

<iframe src="assets/hypothesis_plot1.html" width="800" height="600" frameBorder="0"></iframe>
<iframe src="assets/hypothesis_plot2.html" width="800" height="600" frameBorder="0"></iframe>

We got a P-Value of .0048, which definetely falls under our significance level. Therefore, we will reject our null hypothesis in favor of the alternative. This suggests that our formula for calculating a holistic balance score for each recipe is generalizable to a population with similar values of taste vs ease vs health in their cooking. 

---


