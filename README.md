
by Ryan Lindberg (lindbergryan04@gmail.com) and Ethan Haus (ethanhhaus@gmail.com)

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

<iframe src="assets/missingness_plot2" width="800" height="600" frameBorder="0"></iframe>

<iframe src="assets/missingness_plot3" width="800" height="600" frameBorder="0"></iframe>

<iframe src="assets/missingness_plot4" width="800" height="600" frameBorder="0"></iframe>


---

## Hypothesis Testing

Before we begin our hypothesis test, we will first define what we call a balance score. The balance score is a rating that we give each recipe based on its balance between time to cook, taste, and nutritional profile. We will using recipe rating as a proxy for taste.

We will manually define a formula to calculate this balance score, which returns scores from 1-10, with 10 being the most healthy, easy to cook, and protein dense meals, and 1 being unhealthy, hard to cook, and less nutritional meals.

This score is mostly subjective, as we are setting the weights ourselves. This is okay though, because our goal is to plug in our own recipes, and find out if its a recipe we would like to cook.

$$
\begin{aligned}
W_{\text{RATING\_HIGH}} &= 0.4 \\
W_{\text{RATING\_MEDIUM}} &= 0.2 \\
W_{\text{RATING\_LOW}} &= 0.0 \\
W_{\text{TIME}} &= -0.01 \quad &\text{(Penalizes long cooking times)} \\
W_{\text{N\_STEPS}} &= -0.06 \quad &\text{(Penalizes overly complex recipes)} \\
W_{\text{SAT\_FAT}} &= -0.06 \quad &\text{(Less saturated fat is better)} \\
W_{\text{CALORIES}} &= -0.01 \quad &\text{(Penalizes excessive calories)} \\
W_{\text{PROTEIN}} &= 0.2 \quad &\text{(More protein is better)} \\
W_{\text{CARBS}} &= -0.01 \quad &\text{(Too many carbs reduce balance)} \\
W_{\text{SUGAR}} &= -0.04 \quad &\text{(Excess sugar is penalized)} \\
W_{\text{SODIUM}} &= -0.03 \quad &\text{(Excess sodium is penalized)}
\end{aligned}
$$


$$
\text{balance\_score} =
    \text{rating\_high} + \text{rating\_medium} + \text{rating\_low} +
    \left(W_{\text{TIME}} \cdot \text{time\_factor} \right) +
    \left(W_{\text{N\_STEPS}} \cdot \log\_decay(\text{n\_steps}, \text{max\_values}_{\text{n\_steps}}) \right) +
    \left(W_{\text{SAT\_FAT}} \cdot \log\_decay(\text{sat\_fat}, \text{max\_values}_{\text{sat\_fat}}) \right) +
    \left(W_{\text{CALORIES}} \cdot \log\_decay(\text{calories}, \text{max\_values}_{\text{calories}}) \right) +
    \left(W_{\text{PROTEIN}} \cdot \log\_decay(\text{protein}, \text{max\_values}_{\text{protein}}) \right) +
    \left(W_{\text{CARBS}} \cdot \log\_decay(\text{carbs}, \text{max\_values}_{\text{carbs}}) \right) +
    \left(W_{\text{SUGAR}} \cdot \log\_decay(\text{sugar}, \text{max\_values}_{\text{sugar}}) \right) +
    \left(W_{\text{SODIUM}} \cdot \log\_decay(\text{sodium}, \text{max\_values}_{\text{sodium}}) \right)
$$



|     id | name                                 | ... |   avg_rating | rating_category   | rating_low   | rating_medium   | rating_high   |   balance_score |
|-------:|:-------------------------------------|----:|-------------:|:------------------|:-------------|:----------------|:--------------|----------------:|
| 333281 | 1 brownies in the world    best ever | ... |            4 | Low               | True         | False           | False         |         1.53785 |
| 453467 | 1 in canada chocolate chip cookies   | ... |            5 | High              | False        | False           | True          |         8.25504 |
| 306168 | 412 broccoli casserole               | ... |            5 | High              | False        | False           | True          |         8.87765 |
| 286009 | millionaire pound cake               | ... |            5 | High              | False        | False           | True          |         8.40649 |
| 475785 | 2000 meatloaf                        | ... |            5 | High              | False        | False           | True          |         8.75722 |




- Null Hypothesis ($H_{0}$): 

There is no significant correlation between human calculated balance scores based off intuition and the algorithmically calculated balance scores. 

$H_{0}$ : $p = 0$

(Where $p$ represents the correlation coefficient between human human balance scores and algorithm balance scores.)

- Alternative Hypothesis ($H_{A}$):

​
There is a significant correlation between human calculated balance scores based off intuition and the algorithmically calculated balance scores. 

$H_{0}$ : $p ≠ 0$

Our test statistic is the difference of balance scores calculated by humans vs the balance scores calculated by algorithm. We are going to test at the popular 95% confident level, or .05. We got a P-Value of .0048, which definetely falls under our significance level. Therefore, we will reject our null hypothesis in favor of the alternative. This means that our formula for calculating a holistic statistic for each recipe is generalizable  to a population with similar values of taste vs ease vs health in their cooking. 


<iframe src="assets/hypothesis_plot1" width="800" height="600" frameBorder="0"></iframe>



<iframe src="assets/hypothesis_plot2" width="800" height="600" frameBorder="0"></iframe>



#### Hypothesis Test Conclusion:

We got a P-value of .0048, a Pearson's R of .511. This means we reject the null in favor of the alternative hypothesis, meaning that our formula for calculating the balance score is likely generalizable to a population with similar values of taste vs ease vs health in their cooking.


---


