#Decision Tree Regression 

#regression template 

#polynomial Regression
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
install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ ., data = dataset,
                  control = rpart.control(minsplit = 1))


#predicting results with regression model
Y_pred = predict(regressor, data.frame(Level = 6.5))







#visualising regresssion results(for higher resolution and smoother curve)
library(ggplot2)
X_grid = seq(min(dataset$Level), max(dataset$level), 0.1)

ggplot() + geom_point(aes(x = dataset$Level, y = dataset$Salary)
                      , color = 'red') +
  geom_line(aes(x = X_grid, y = predict(regressor, newdata = data.frame(Level = X_grid)))
            , color = 'blue') +
  ggtitle('truth or bluff (Decision Tree Regression Model)') +
  xlab('level') + ylab('salary')







