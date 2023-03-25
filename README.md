# NBA_MVP_ML
NBA MVP Machine Learning Model Development &amp; Prediction

**Objective**

The goal of this project was 2-fold. First and foremost, to predict the NBA MVP winner of the 2022 season by predicting the award share score of each player. The highest award share score would be the MVP winner. The second goal was to predict the top 3 MVP finalists correctly. The approach would try different regressors and identify the most accurate models used in the prediction based on the mean root squared error, r2 score, 1st place prediction accuracy, and top three MVP prediction accuracy. Note, the python code is attached, above to this repository.

**Data Selection**

To conduct this project, a dataset from Kaggle was sourced and selected, The 1982-2022 NBA Player Statistics with MVP Votes: https://www.kaggle.com/datasets/robertsunderhaft/nba-player-season-statistics-with-mvp-win-share. The NBA Player statistics dataset was developed by web scraping https://www.basketball-reference.com.

**Data Exploration & Analysis**

Conducted basic statistical analyses and reviews with the original data set such as:
* The dataset had 55 columns and 17697 rows.
* 52 numerical and 3 non-numerical variables.
* Mean, Median, and Standard Deviation of the players' stats of the top 10 MVP award share scores for each season.
* Distribution of MVP winners vs non-winners.
* Analysis of features like age, win share, and points scored of MVP winners vs non-MVPs.
* Distribution of player volume by season.

![image](https://user-images.githubusercontent.com/89919659/227672505-55f9b578-d439-4249-9b1a-ed0c4612ff3e.png)

**Data Cleaning & Transformation**
* Removed all rows with any NULL values.
* Add a binary column that would identify the MVP of each season in the dataset with a number 1.
* Added a Boolean column that would identify the MVP of each season in the dataset.
* Feature Analysis and Final Input Data
* Conducted Feature Analysis via the Univariate method to find the top 20 features we should use within our machine learning model
* Assessed those 20 features through Pair Plot and Correlation to find any collinearity. Removed 5 of the 20 features due to collinearity
* Split the data into Train, Test, and Validation. Where the Training data was from seasons 2012 to 2020. Test and Validation/prediction data were seasons 2021 and 2022, respectively.
* Used 2012 to 2020 as training data because the ratio between train and test/validation came to 80/20.
			Train data set 2012 to 2020 = 80%.
			Test data set 2021 = 10%.
			Validation/ objective prediction season set 2022 = 10%
* Further, split our training/test/validation into target and predictor variables.
* Applied feature scaling using the StandardScaler function.

**Predictive Model Development & Analysis**

* Employed regression modeling to predict the award share score of each player.
* Choose the top 4 performing models from this first model run: Random Forest, Extra Trees Regressor, XGBoost Boosting Regressor, and CatBoost Regressor. 
* Developed, Hyperparameter Tuned, and fitted the top 4 models with the train/test/validation data sets.
* Reviewed the output of the predictions on the Test and Validation seasons, 2021 and 2022, respectively. Along with the RMSE and R^2 values of each model.

**Conclusion**

XGBoost Regressor with the following Hyperparameters: {'subsample': 0.5, 'objective': 'reg:squarederror', 'n_estimators': 100, 'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.5} performed the best considering all 3 of our performance criteria and was selected as our final prediction model.

XGBoost Regressor performance: 
* Predicted 2022 season MVP Winner Correctly
* Predicted the top 3 finalists, just not in the correct order for 2nd and 3rd place.
* R2 = 0.8860
* RMSE = 0.0191

Learning curve
* Plotted the learning curve for the training data from 2012 to 2020 based on RMSE.
* Using the final selected ML model XGBoost with the hyperparameters stated above.
* The plot result shows the model has a high variance. This means, the model focuses too much on specific patterns in the training dataset and does not generalize well on unseen data, therefore can be prone to overfitting.
* To reduce the high variance in our case was to increase the Training data size. For example, we can use all the data from 1982 to 2020 as Train and keep the seasons as we have done for Test (2021 season) and Validation/Objective Predict (2022 season).
* To further improve the gap between the training and validation plot of the learning curve we removed low-impact features to reduce the models' complexity.
* This tuning improved the learning curve result and the training and validation curves started to converge at 6000 samples.

The top 5 features that highly impacted the regression model were as follows:
* Point per Game (pts_per_g)
* Win Shares (ws)
* Field Goals per Game (fga_per_g)
* Defensive Win Shares (dws)
* Assist Percentage (ast_pct)

![image](https://user-images.githubusercontent.com/89919659/227674670-a43ab995-1cd7-49b2-b059-5a4b13fef282.png)

Here where the final predictions for the 2022 season:

![nbanvp](https://user-images.githubusercontent.com/89919659/213782999-2f753d7e-ad5c-47a7-af0b-c0234257f1e9.PNG)
* note: any column with _PRED was the ML predicted value or ranking
