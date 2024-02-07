# Statistical analysis: NBA players

### Introduction ###
In the framework of sports data science, the dataset of basketball players at hand represents all players from three seasons of the NBA and their stats for a certain season. 
The rows of the dataset represent all players from three seasons of the NBA and the columns their stats for a certain season (see table below for full variable explanation). 
As opposed to other team sports, for instance soccer, basketball players play both offense and defense, regardless of their assigned position. 
During a game, a team will generally field five basic positions: point guard (PG), shooting guard (SG), small forward (SF), power forward (PF) and center (C). 
Each of these positions has their own tasks. Note that a player can often play different positions (we’ve provided the player’s main position in the variable Pos1) 
and that in modern basketball a player will sometimes be assigned a hybrid position that doesn’t perfectly fit with one of the five classical positions above. 
Further note that it’s often the case that the tasks of different positions overlap. Therefore, we’ve also provided a variable Role (Back or Front), that somewhat represents
the player’s main role in the team.

The following summary gives a description of the variables:
- Player.x: Name of the player.
- Pos1: Main position of the player.
- Tm: Name of the player’s team.
- Season: 2016-17, 2017-18 or 2018-19.
- Conference: Western or Eastern Conference.
- Role: Back or Front.
- Play: If the player played in the All-Star Game (Yes or No).
- Age: Age of the player.
- MP: Minutes played per game.
- X3P.: The number of three-point field goals divided by the number of three-point field goal attempts.
- X2P.: The number of two-point field goals divided by the number of two-point field goal attempts.
- FT.: The number of scored free throws divided by the number of free throw attempts.
- ORB: Offensive rebounds per game. DRB Defensive rebounds per game. AST Assists per game.
- STL: Steals per game.
- BLK: Blocks per game.
- TOV: Turnovers per game.
- PF: Personal fouls per game. Salary Yearly salary in USD.

The following project is aimed at performing various analysis:

- Exploratory and preprocessing step: the dataset is explored through statistics and plots. The dataset is comprised both of continuous and categorical variables.
  By plotting the continuous variables depending on the level of a categorical variable, we observe that the variable Play nicely separates almost all continuous variables in
  two groups. On the other hand, Role partially does so, whereas Pos1 doesn't separate the continuous variables in well-defined groups.
  We remove outliers that could have a strong influence on the outcome, through the Minimum Covariance Determinant (more robust than traditional methods) and transform the
  quantitative predictors to a more normal distribution, employing Box-Cox transformations.
  For the following steps, the cleaned and transformed dataset is retained and variables Player.x, Tm, Season and Conference are removed.

- PCA: we perform a Principal Component Analysis. The data matrix only consists of the continuous variables from the transformed and cleaned dataset.
  Since a correlation plot shows linear relations between some of the varibles, we assume that the first k principal components will be able to capture a great proportion
  of the variance.
  The analysis is carried out both with the covariance and correlation matrix, finally choosing to adopt the latter, since the variables present very different scales,
  affecting which variables contribute to the first principal components. The number of components is chosen as equal to 5, based on the visualization of a screeplot and the
  choice of retaining at least 80% of the total variance through the PCs.
  By visualizing scatterplots of the scores and biplots for the first 5 principal components, the groups determined by the variable Play are once again visible, whereas the
  groups determined by the categorical variable Role are only partially visible.
  By detecting outliers through classical PCA, only orthogonal outliers are exposed (outlying orthogonal distance/projection error), whereas no bad leverage points are identified.
  By performing the same analysis with a robust PCA method, which individuates a subspace robustly and uses MCD (minimum Covariance Determinant) to detect outliers, a number of
  bad leverage points and orthogonal outliers are in fact found.

- Clustering: in this step, instances are grouped through partitioning cluster methods and hierarchical cluster methods. Furthermore, variables are divided in groups through
  hierarchical cluster methods.
  By performing a partitioning cluster analysis on the observations, the generated groups are very similar when employing k-means and k-medoids. According to the silhouette
  score, which is employed as metric to judge the fit of the grouping to the dataset, the optimal number of clusters is 2. Standardizing the variables yields worse results
  than not standardizing them. Once again, the variable Play partially separates the two groups in accordance with the clustering method.
  Hierarchical cluster analysis is comprised of divisive analysis and agglomerative analysis (with complete, average and single linkage). Divisive analysis and agglomerative
  analysis with average linkage yield similar results, according to the silhouette score and divisive/agglomerative coefficient. These methods also produce similar groupings to
  the previously employed partitioning methods.
  A hierarchical cluster analysis is also carried out on variables, detecting two groups, which partition the variables according to the correlations among them: linearly related
  variables are clustered together, whereas variables that don't show correlations with the others are partitioned in a separate group.
  By plotting a heatmap, based on the hierarchical clustering of variables and the hierarchical clustering of observations, we can distinguish two groups in which the
  instances are divided and two groups in which the variables are divided. This reflects in a total of four groups on the heatmap, where the determined groups are recognizable.

- Linear regression: for the linear regression setting, the dataset of continuous variables is split into a training set and a validation set, through a 75/25 split.
  The model is regressed on other continuous variables in order to predict Salary.
  A linear model is first fitted on all variables. Through R^2 (coefficient of multiple determination) and adjusted R^2 (coefficient of multiple determination adjusted by the
  degrees of freedom of the sum of squares), the most relevant variables are detected. The findings are similar to what results from using Stepwise AIC (Akaike Information
  Criterion). By measuring multicollinearity through the Variance Inflation Factor and the condition number, it can be seen that, although some variables are linearly relkated,
  the model fit only on the chosen subset of variables doesn't suffer from multicollinearity.
  By checking the model assumptions (i.e. linear assumption and residuals normally distributed) through a scatterplot of the residuals and a normal Q-Q plot, although the errors
  are almost normally distributed, the error variance is not constant, suggesting heteroscedasticity. In order to solve the issue, the response variable Salary is transformed
  through a simple log transformation, so that the interpretability of the model isn't affected.
  To handle outliers, to which the least squares estimator is sensitive, Least Trimmed Squares is used, in order to reduce the majority of the residuals, not all of them.
  After the transformation and removing outliers, the residuals show a random pattern around zero in the residuals vs fitted plot.
  Performing an analysis of variance (ANOVA) on the final model created previously yields a significant extra sum of squares (SSR) for all variables considered, except PF.
  Furthermore, all variables have one degree of freedom, since there is no categorical variable employed in the model. Moreover, Age, MP, TOV, DRB and X3P have high F-values,
  indicating a high ratio of variance explained by the model over the variance not explained. Finally, the F-test indicates that all variables but PF contribute to
  explaining the variability in the response variable. As a consequence, removing PF wouldn’t significantly worsen the predictive power of the model.
  The model is then evaluated through R^2, adjusted R^2, RMSE (on the training set) and RMSEP (on the validation set).
  Based on the final model, we compute a 99% confidence interval and prediction interval for the average Salary of a player with Age = 26, MP = 21.7, X3P. = 0.31,
  X2P. = 0.50, FT. = 0.75, ORB = 0.9, DRB = 3.0, AST = 2.1, STL = 0.7, BLK = 0.4, TOV = 1.2 and PF = 1.8. 
  The model is re-trained on the whole dataset (without a training-validation split). Before feeding the new data to the model, the variables are transformed according
  to the same Cox-Box transformations that were used on the whole dataset. After obtaining the confidence interval for the mean response and the prediction interval
  for the unknown response, the transformation of the dependent variable is reversed, to obtain the predictions on the original scale. 

- Classification: in this part, a training-test split is again used to train the models and test their performance. Three classifiers are employed: LDA, QDA and
  k-Nearest Neighbours, in order to classify the categorical variables Play, Role and Pos1, based on the continuous predictors.
  The three classifiers are evaluated through confusion matrices, mosaic plots, ROC (receiver-operating characteristic) curves and AUC (Area Under the Curve).
  They yield different results for the three variables, with k-NN performing poorly on Role, but doing better on Play and Pos1. The optimal k is determined through
  10-fold cross validation and plotted on a fitting graph. Overall, however, the simplifying assumption underlying LDA (Linear Discriminant Analysis) of a normal distribution
  in the dataset and homoscedasticity in the classes produces the best results for classifying Role and Play. On the other hand, the multiclass classification necessary
  for Pos1 is best performed by QDA (Quadratic Discriminant Analysis), as the underlying assumption is more relaxed, by not assuming constant covariance matrix between classes.

---

### Requirements to run the project ###

An jupiyter environment with:
- R kernel
- the possibility to install the libraries mentioned in the .ipynb files

---

### How to run the project ###

Download the zip folder containing the .ipynb files and the dataset NBA. Modify the paths to the directory in the .ipynb files to your actual directory. 
I advise you to run the project on Jupyter Notebook, due to the specifics in the kernel.

The files should be run in this order:
1. Exploratory analysis
2. PCA
3. Clustering Analysis
4. Linear regression
5. Classification

due to the specifics in the libraries installations.
