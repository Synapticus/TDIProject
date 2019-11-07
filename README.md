This website (https://warm-fjord-40415.herokuapp.com/) is submitted as the TDI capstone project for Sean Patrick.

It covers an investigation into officer-involved shootings using a machine learning model to predict lethal vs. nonlethal outcomes based on circumstantial factors.

The datasets used are the Vice News OIS dataset (https://news.vice.com/en_us/article/a3jjpa/nonfatal-police-shootings-data) and Denver police shooting dataset (https://www.denvergov.org/opendata/dataset/city-and-county-of-denver-denver-police-officer-involved-shootings).

The audience of this study is law enforcement and officer-involved shooting victim advocacy groups, with the intent of revealing patterns in shooting survivorship that were not initially obvious. To this end, an analysis of feature importances was performed finding that there is a significant racial component in officer-involved shooting survivorship.

Visualizations of the dataset include a number of bar graphs showing the proportion of fatal to nonfatal shootings according to various attributes, in addition to a bubble plot showing the frequency of officer-involved shootings by officer and subject race.

The model used to make predictions was a random forest classifier, with hyperparameters n_estimators=30, max_depth=18 determined through grid search cross validation. Interactivity with the model is presented in the form of live predictions made from user-selected inputs.