#Random Forest Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)


#fitting the regression model to the dataset
#create our regressor
install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1], y = dataset$Salary,
                         ntree = 100)



#predicting results with regression model
Y_pred = predict(regressor, data.frame(Level = 6.5))





#visualising regresssion results(for hogher resolution and smoother curve)
library(ggplot2)
X_grid = seq(min(dataset$Level), max(dataset$level), 0.1)

ggplot() + geom_point(aes(x = dataset$Level, y = dataset$Salary)
                      , color = 'red') +
  geom_line(aes(x = X_grid, y = predict(regressor, newdata = data.frame(Level = X_grid)))
            , color = 'blue') +
  ggtitle('truth or bluff (RAndom Forest Regression Model)') +
  xlab('level') + ylab('salary')