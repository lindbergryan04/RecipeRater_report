# Recipe Rater

by Ryan Lindberg (lindbergryan04@gmail.com) and Ethan Haus (ethanhhaus@gmail.com)

---

## Introduction

In this project, we used a large dataset of recipes, and tried use their various features to train a model that could predict how a new recipe would be rated (out of 5 stars). There's no shortage of recipe's available online, we felt that this could offer a nice way for users to be able to filter threw some they saved that sounded good, and see what was actually worth making, if they didn't have any feedback.  Our model predicted the recipes based on the numerical columns that we used actually utilized. Our dataset had 234429 rows and 15 columns, of which we used 4. We also created a new column, avg_rating, which was used as well. We used the minutes, total minutes required to complete the recipe, and the n_steps, or number of steps in the recipe, with various weights to quantify how difficult it was to make a recipe. We used certain elements of the nutrition column (calories, sugar, sodium, protein and saturated fat), a column which provided nutritional information, to quantify how healthy each recipe is. Using these two stats, and factoring in the average rating, made a balance score, which we thought of as a holistic evaluation of a recipe. We used a random forsest network to calculate the optimal hyperparameters for our model

---

## Cleaning and EDA


Our first step was to replace the ratings of '0' with np.nan. We then created an 'avg_rating' column to represent the average rating of each recipe. We did this by grouping by each recipeID, since most recipes had multiple reviews, then we simply took the mean after grouping. Then, we dropped some columns we weren't going to use. These ended up being 'review', 'contributor_id', 'user_id', 'date', 'submitted', 'description', and 'steps'. After, we looked at all the columns with string lists, and converted them to lists. Our plan was then to take these lists, and turn the list values into individual columns. We ended up only needing to do this with the 'nutrition' column, and turned all the different elements of 'nutrition' into their own columns. Since we'd extracted all that we needed from 'nutrition' we dropped the column. 

NEED DATAFRAME

For our Univariate Analysis we wanted more insight on our 'minutes' and 'ratings' columns. Since our average ratings were so high, we wanted to see a distribution of all the values. And since we figured that the amount of effort would heavily factor into people's perceptions of the recipe and therefore their rating, we felt effort was best quantified by how long it took to complete the recipe, ergo 'minutes' column. So we decided to view the distribution to see how to best weight the feature. 

<iframe src="eda-minutes.html" width="800" height="600" frameBorder="0"></iframe>



As you can see in the graph above, the graph is skewed right EDIT, with the majority of our values normally distributed around a rough mean of 30 minutes. There are definetely some outlier recipes that take much longer, but it's clear to see that most of our recipes are around 10 minutes to an hour. 

When initially starting this project, we were looking to see if any columns were correlated with each other. We thought this would be great to know in helpig us find columns with the best predictive power for ratings. 

<iframe src="eda-heatmap.html" width="800" height="600" frameBorder="0"></iframe>



---

## Assessment of Missingness

Here's what a Markdown table looks like. Note that the code for this table was generated _automatically_ from a DataFrame, using

```py
print(counts[['Quarter', 'Count']].head().to_markdown(index=False))
```

| Quarter     |   Count |
|:------------|--------:|
| Fall 2020   |       3 |
| Winter 2021 |       2 |
| Spring 2021 |       6 |
| Summer 2021 |       4 |
| Fall 2021   |      55 |

---

## Hypothesis Testing


---
