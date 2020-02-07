library(purrr)
library(tidyr)
library(ggplot2)
library(caret)
library(plyr)
library(dplyr)
library(grid)
library(gridExtra)
library(pROC)
library(e1071)
library(olsrr)
library(rpart.plot)
library(rpart)
set.seed(4321) # 12345
# options(scipen=999) # turn off scientific notation (such as 1.000e-15)


# Load in data
train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)


###### EDA/Visualizations ######
# histogram matrix of numeric variables
X <- train[, 1:11] # create a subset dataframe consist of only the predictors
par(mfrow = c(3, 4)) # split the graph into 9 subplots with 3 rows & 3 columns
for (i in 1:ncol(X)) {
  hist(X[ ,i], xlab = names(X[i]), main = paste(names(X[i])), col="steelblue")
}

# distribution of labels
table(train$fake)












###### Models Training: Logistic Regression (Linear) ######
# Model 1: without any processing and using all variables
# relevel 'fake' for later purpose
train$fake <- ifelse(train$fake==1, 'yes', 'no')
test$fake <- ifelse(test$fake==1, 'yes', 'no')

# relevel 'profile.pic'
train$profile.pic <- ifelse(train$profile.pic==1, 'yes', 'no')
test$profile.pic <- ifelse(test$profile.pic==1, 'yes', 'no')

# relevel 'private'
train$private <- ifelse(train$private==1, 'yes', 'no')
test$private <- ifelse(test$private==1, 'yes', 'no')

# make categorical variables factors
train$fake <- as.factor(train$fake)
test$fake <- as.factor(test$fake)

train$profile.pic <- as.factor(train$profile.pic)
test$profile.pic <- as.factor(test$profile.pic)

train$private <- as.factor(train$private)
test$private <- as.factor(test$private)













# train model with all variables
lr <- glm(data=train,
          fake~.,
          family='binomial') # need to take out or transform some variables
summary(lr) # these coefficients are in log-odds scale

# exp(coef(lr)) # these are in odds scale (more interpretable): for 1 unit increase in each variable;
# an account is X times more likely to be fake.
# If exp(coef(lr)) < 1: interpret by 1/exp(coef(lr)) - X times less likely to be fake.
# predict
predictions <- predict(lr, type='response')
# confusion matrix
confusion.matrix <- table(train$fake, predictions > 0.5)
confusion.matrix

# accuracy
accuracy <- (confusion.matrix[1] + confusion.matrix[4]) / sum(confusion.matrix)
accuracy










###### Pre-processing (transformation, standardization, etc.) ######
# excluding 'external.URL' (this variable causes perfect separation problem)

# identify Near Zero Variance variables
nzv <- nearZeroVar(train, saveMetrics=T)
nzv # 'nums.length.fullname' & 'name..username' are near zero variance variables.

# can consider excluding these variables.
# exclude the near 0 variance variables
no_nzv <- preProcess(train, method='nzv')
train_no_nzv <- predict(no_nzv, train)
test_no_nzv <- predict(no_nzv, test)

# check for multi-collinearity among variables
# same the continuous variable separately
vars <- c('nums.length.username', 'fullname.words', 'description.length',
          'X.posts', 'X.followers', 'X.follows')
# correlation matrix
varCor <- cor(train_no_nzv[vars])
round(varCor, 2)

# Print the number of correlated predictors higher than a correlation threshold
highCorr <- sum(abs(varCor[lower.tri(varCor)]) > 0.5)
highCorr # no highly correlated variables

# use VIF to indicate multicollinearity
# VIF <= 1: not correlated. 1 < VIF < 5: moderately correlated. VIF > 5: highly correlated
vif <- ols_coll_diag(lm(data=train_no_nzv,
                 fake~. -profile.pic -external.URL -private)) # exclude the categorical variables
vif.table <- data.frame('Variables'=colnames(train_no_nzv[vars]), 'VIF'=vif$vif_t[3]) # no multicollinearity

# plot bar chart of each variable VIF with multicollinearity threshold
par(mfrow=c(1,1)) # reset the plot to 1 whole plot
barplot(height=vif.table$VIF,
        width=0.1,
        main='VIF of each variable',
        xlab='Variables',
        ylab='VIF',
        ylim=c(0, 7),
        col='steelblue')
labels <- vif.table$Variables
mp <- barplot(height=vif.table$VIF,
              width=0.5,
              main='VIF of each variable',
              xlab='Variables',
              ylab='VIF',
              ylim=c(0, 7),
              col='steelblue')
text(mp, par('usr')[3], labels = labels, srt = 40, adj = c(1.1,1.1), xpd = TRUE, cex=.75)
abline(h=5, lty=2)












# Transform some variables using Box-Cox:
# 'nums.length.username', 'fullname.words',
# 'description.length', 'X.posts', 'X.followers', 'X.follows'
# First of all, let's have a look at the distribution of the continuous variables
X <- train_no_nzv[vars] # create a subset dataframe consist of only the predictors
par(mfrow = c(2, 3)) # split the graph into 9 subplots with 3 rows & 3 columns
for (i in 1:ncol(X)) {
  hist(X[, i], xlab = names(X[i]), main = paste(names(X[i])), col="steelblue")
}

# make a copy of the data before processing
cleaned.train <- train_no_nzv
cleaned.test <- test_no_nzv

# apply to train set
cleaned.train[vars] <- cleaned.train[vars] + 1 # add 1 to all these variables because cannot take log(0)
cleaned.test[vars] <- cleaned.test[vars] + 1

# boxcox
boxcox <- preProcess(cleaned.train[vars], method='BoxCox')
cleaned.train <- predict(boxcox, cleaned.train) # train set
cleaned.test <- predict(boxcox, cleaned.test) # test set

# plot the Box-Cox transformed variables
X <- cleaned.train[vars] # create a subset dataframe consist of only the predictors
par(mfrow = c(2, 3)) # split the graph into 9 subplots with 3 rows & 3 columns
for (i in 1:ncol(X)) {
  hist(X[, i], xlab = names(X[i]), main = paste(names(X[i])), col="steelblue")
}

# center and scale (standardization) the predictors
standardization <- preProcess(cleaned.train[vars], method=c('center', 'scale'))
cleaned.train <- predict(standardization, cleaned.train) # train set
cleaned.test <- predict(standardization, cleaned.test) # test set

# plot the new transformed variables
X <- cleaned.train[vars] # create a subset dataframe consist of only the predictors
par(mfrow = c(2, 3)) # split the graph into 9 subplots with 3 rows & 3 columns
for (i in 1:ncol(X)) {
  hist(X[, i], xlab = names(X[i]), main = paste(names(X[i])), col="steelblue")
}

# Outliers: using Spatial Sign - bring outliers closer to the majority of the data
X <- cleaned.train[-c(203, 41, 185, 6, 35, 46, 42, 26, 
                      166, 60, 259, 28, 44, 25, 45, 183), vars] # create a subset dataframe consist of only the continuous predictors
                                                                # exclude some rows for demonstration purpose
par(mfrow = c(2, 3)) # split the graph into 6 subplots with 2 rows & 3 columns
for (i in 1:ncol(X)) {
  boxplot(X[, i], xlab = names(X[i]), main = paste(names(X[i])), col="steelblue")
} # some variables still have some outliers

# scatter plot of X.posts & X.follows
par(mfrow=c(1,1))
plot(x=cleaned.train$X.follows[-c(203, 41, 185, 6, 35, 46, 42, 166, 60, 259, 28, 44)], # exclude these rows 
     y=cleaned.train$X.followers[-c(203, 41, 185, 6, 35, 46, 42, 166, 60, 259, 28, 44)], # for demonstration purpose
     main = "X.follows & X.followers with Outliers",
     xlab = "X.follows", ylab = "X.followers",
     pch = 19, col='steelblue')

# spatial sign
cleaned.train2 <- spatialSign(cleaned.train[vars]) # train set
cleaned.train2 <- data.frame(cleaned.train2) 

cleaned.test2 <- spatialSign(cleaned.test[vars]) # test set
cleaned.test2 <- data.frame(cleaned.test2)

# put the variables after doing spatial sign into the data frame
cleaned.train[vars] <- cleaned.train2[vars] # train set
cleaned.test[vars] <- cleaned.test2[vars] # test set

# replot the boxplot of our continuous variables after doing spatial sign
X <- cleaned.train[vars]
par(mfrow = c(2, 3)) # split the graph into 6 subplots with 2 rows & 3 columns
for (i in 1:ncol(X)) {
  boxplot(X[, i], xlab = names(X[i]), main = paste(names(X[i])), col="steelblue")
} # no more outliers

# plot the scatter plot of X.follows & X.followers after spatial sign
par(mfrow=c(1,1))
plot(x=cleaned.train$X.follows[-c(203, 41, 185, 6, 35, 46, 42, 166, 60, 259, 28, 44)], # exclude these rows 
     y=cleaned.train$X.followers[-c(203, 41, 185, 6, 35, 46, 42, 166, 60, 259, 28, 44)], # for demonstration purpose
     main = "X.follows & X.followers after Spatial Sign",
     xlab = "X.follows", ylab = "X.followers",
     pch = 19, col='steelblue')

















###### Model Training: lr, knn, decision tree, random forest ######
# logistic regression using using the processed data
lr <- glm(data=cleaned.train,
          fake~. -external.URL,
          family=binomial)
summary(lr)

# predict
new.predictions <- predict(lr, type='response')
# confusion matrix
confusion.matrix <- table(cleaned.train$fake, new.predictions > 0.5)
confusion.matrix

# accuracy
new.accuracy <- (confusion.matrix[1]+confusion.matrix[4])/sum(confusion.matrix)
new.accuracy # accuracy is improved compared to the model without any processing










# using Cross-validation with 'train' by 'caret' package
control <- trainControl(method='repeatedcv',
                        number=10, # 10 folds cv
                        repeats=5) # repeat 5 times
                        #classProbs=T,
                        #summaryFunction = twoClassSummary) # Declare this is binary classification summary.
                                                            # so we can set different metric other than Accuracy.
metric <- 'Accuracy'

fit.lr <- train(data=cleaned.train,
                fake~. -external.URL,
                method='glm',
                trControl=control,
                metric=metric)

summary(fit.lr)
fit.lr

# predictions
lr.predictions <- predict(fit.lr, type='raw')

lr.predictionsProb <- round(predict(fit.lr, type='prob'), 8) # need this for ROC curve

# confusion matrix of model trained by cross-validation
lr.confusion <- confusionMatrix(data=lr.predictions,
                                reference=cleaned.train$fake,
                                positive='yes',
                                mode='everything')
lr.confusion




###### Model Training: kNN (Non-linear) ######
fit.knn <- train(data=cleaned.train,
                 fake ~ . -external.URL,
                 method='knn',
                 trControl=control,
                 tuneGrid=expand.grid(k=seq(1, 15, by=2)),
                 metric=metric)
fit.knn

# plot in-sample accuracy with different k neighbors
plot(fit.knn)

# predictions
knn.predictions <- predict(fit.knn, type='raw')

knn.predictionsProb <- round(predict(fit.knn, type='prob'), 8) # need this for ROC curve

# confusion matrix of model trained by cross-validation
knn.confusion <- confusionMatrix(data=knn.predictions,
                                 reference=cleaned.train$fake,
                                 positive='yes',
                                 mode='everything')
knn.confusion






###### Model Training: Decision Tree (rpart) ######
# Because tree models do not require pre-processing, we can use original data to fit model
fit.rpart <- train(data=train,
                   fake ~ .,
                   method='rpart',
                   trControl=control,
                   tuneGrid=expand.grid(cp=seq(0.01, 0.1, by=0.005)),
                   metric=metric)
fit.rpart

# plot in-sample accuracy with different complexity parameters
plot(fit.rpart) # cp=0.01: best model (the lower cp; the more complex the tree)

# plot the selected features by our rpart model
plot(varImp(fit.rpart)) # important attributes in the classification process used in the tree)

# predictions
rpart.predictions <- predict(fit.rpart, type='raw')

rpart.predictionsProb <- round(predict(fit.rpart, type='prob'), 8) # need this for ROC curve

# confusion matrix of model trained by cross-validation
rpart.confusion <- confusionMatrix(data=rpart.predictions,
                                   reference=train$fake,
                                   positive='yes',
                                   mode='everything')
rpart.confusion

# plot final decision tree
rpart.plot(fit.rpart$finalModel,
           type=4,
           clip.right.labs=F,
           branch=.3)













###### Model Training: Random Forest ######
# mtry: number of randomly selected variables used at each tree. The higher, the less random
# ntree: number of trees in this Random Forest
tuneGrid.rf <- expand.grid(mtry=c(1:10))
fit.rf <- train(data=train,
                fake ~ .,
                method='rf',
                trControl=control,
                tuneGrid=tuneGrid.rf,
                metric=metric)
fit.rf
plot(fit.rf)

plot(varImp(fit.rf)) # important variables used by random forest model


# predictions
rf.predictions <- predict(fit.rf, type='raw')

rf.predictionsProb <- round(predict(fit.rf, type='prob'), 8) # need this for ROC curve

# confusion matrix of model trained by cross-validation
rf.confusion <- confusionMatrix(data=rf.predictions,
                                reference=cleaned.train$fake,
                                positive='yes',
                                mode='everything')
rf.confusion



















###### Performance Measure ######

###### Apparent Performance of each model ######
# summarize accuracy of models from cross-validation
results <- resamples(list('Decision Tree'=fit.rpart,
                          'Logistic Regression'=fit.lr,
                          'Random Forest'=fit.rf,
                          'kNN'=fit.knn))
summary(results)

# compare accuracy of models (Apparent Performance)
dotplot(results)

# accuracy
metrics.accuracy <- round(rbind(lr.confusion$overall[1],
                                knn.confusion$overall[1],
                                rpart.confusion$overall[1],
                                rf.confusion$overall[1]), 3)
# kappa
metrics.kappa <- round(rbind(lr.confusion$overall[2],
                             knn.confusion$overall[2],
                             rpart.confusion$overall[2],
                             rf.confusion$overall[2]), 3)
# f1 score
metrics.f1 <- round(rbind(lr.confusion$byClass[7],
                          knn.confusion$byClass[7],
                          rpart.confusion$byClass[7],
                          rf.confusion$byClass[7]), 3)
# metrics table
metrics <- data.frame(cbind(metrics.accuracy, metrics.kappa, metrics.f1))

# Add a colum to indicate this is metrics of Train set or Test set
metrics$Dataset <- c('Train', 'Train', 'Train', 'Train')

# rename the rows to the names of each model
rownames(metrics) <- c('Logistic Regression',
                       'kNN',
                       'Decision Tree',
                       'Random Forest')
# use the rownames as a variable
metrics$Model <- rownames(metrics)

# barplot of metrics by each model
# accuracy
ggplot(metrics, aes(x=reorder(metrics$Model, -metrics$Accuracy), y=metrics$Accuracy)) +
  geom_bar(stat = "identity", fill='steelblue') + 
  coord_cartesian(ylim = c(0, 1)) + 
  labs(y = "Accuracy", x = "Models") +
  geom_hline(yintercept=1,
             linetype='dashed',
             color='red',
             size=0.5) +
  geom_text(aes(label=metrics$Accuracy), 
            vjust=1.6, 
            color="white", 
            size=4.5) +
  ggtitle('Accuracy of each model') +
  theme(plot.title = element_text(hjust = 0.5))


# Kappa
ggplot(metrics, aes(x=reorder(metrics$Model, -metrics$Kappa), y=metrics$Kappa)) +
  geom_bar(stat = "identity", fill='steelblue') + 
  coord_cartesian(ylim = c(0, 1)) + 
  labs(y = "Kappa", x = "Models") +
  geom_hline(yintercept=1,
             linetype='dashed',
             color='red',
             size=0.5) +
  geom_text(aes(label=metrics$Kappa), 
            vjust=1.6, 
            color="white", 
            size=4.5) +
  ggtitle('Kappa of each model') +
  theme(plot.title = element_text(hjust = 0.5))

# F1 score
ggplot(metrics, aes(x=reorder(metrics$Model, -metrics$F1), y=metrics$F1)) +
  geom_bar(stat = "identity", fill='steelblue') + 
  coord_cartesian(ylim = c(0, 1)) + 
  labs(y = "F1 Score", x = "Models") +
  geom_hline(yintercept=1,
             linetype='dashed',
             color='red',
             size=0.5) +
  geom_text(aes(label=metrics$F1), 
            vjust=1.6, 
            color="white", 
            size=4.5) +
  ggtitle('F1 Score of each model') +
  theme(plot.title = element_text(hjust = 0.5))












###### Lift Chart ######
## Generate the test set results
lift_results <- data.frame(fake = cleaned.test$fake)
lift_results$lr <- predict(fit.lr, newdata=cleaned.test, type = "prob")[, 'no'] # only use the probability of 'fake' class (probability observation being fake)
lift_results$knn <- predict(fit.knn, newdata=cleaned.test, type = "prob")[, 'no']
lift_results$rpart <- predict(fit.rpart, newdata=test, type = "prob")[, 'no']
lift_results$rf <- predict(fit.rf, newdata=test, type = 'prob')[, 'no']
head(lift_results)

# plot lift chart
trellis.par.set(caretTheme())
lift_obj <- lift(fake ~ lr + knn + rpart + rf, data = lift_results)
xyplot(lift_obj, 
     lwd=2,
     auto.key = list(columns = 4,
                     lines = TRUE,
                     points = FALSE))








###### Test set ######
# Logistic Regression
test.lr.predictions <- predict(fit.lr, newdata=cleaned.test, type='raw')

test.lr.Prob <- round(predict(fit.lr, newdata=cleaned.test, type='prob'), 8) # need this for ROC curve

# confusion matrix of Logistic Regression
test.lr.confusion <- confusionMatrix(data=test.lr.predictions,
                                     reference=cleaned.test$fake,
                                     positive='yes',
                                     mode='everything')
test.lr.confusion

# knn
test.knn.predictions <- predict(fit.knn, newdata=cleaned.test, type='raw')

test.knn.Prob <- round(predict(fit.knn, newdata=cleaned.test, type='prob'), 8) # need this for ROC curve

# confusion matrix of kNN
test.knn.confusion <- confusionMatrix(data=test.knn.predictions,
                                      reference=cleaned.test$fake,
                                      positive='yes',
                                      mode='everything')
test.knn.confusion

# Decision Tree
test.rpart.predictions <- predict(fit.rpart, newdata=test, type='raw')

test.rpart.Prob <- round(predict(fit.rpart, newdata=test, type='prob'), 8) # need this for ROC curve

# confusion matrix of Decision Tree
test.rpart.confusion <- confusionMatrix(data=test.rpart.predictions,
                                        reference=test$fake,
                                        positive='yes',
                                        mode='everything')
test.rpart.confusion

# Random Forest
test.rf.predictions <- predict(fit.rf, newdata=test, type='raw')

test.rf.Prob <- round(predict(fit.rf, newdata=test, type='prob'), 8) # need this for ROC curve

# confusion matrix of Decision Tree
test.rf.confusion <- confusionMatrix(data=test.rf.predictions,
                                     reference=test$fake,
                                     positive='yes',
                                     mode='everything')
test.rf.confusion









###### ROC ######

# logistic regression
roc(cleaned.test$fake, test.lr.Prob$yes, plot=T, 
    legacy.axes=T, xlab="False Positive Rate", ylab="True Positive Rate",
    main='ROC', col="lightcoral", print.auc=T, print.auc.y=0.43)

# rf
plot.roc(test$fake, test.rf.Prob$yes, add=T,
         col="goldenrod1", print.auc=T, print.auc.y=0.37)

# add knn
plot.roc(cleaned.test$fake, test.knn.Prob$yes, add=T, 
         col="darkolivegreen4", print.auc=T, print.auc.y=0.31)

# add decision tree
plot.roc(test$fake, test.rpart.Prob$yes, add=T, 
         col="cornflowerblue", print.auc=T, print.auc.y=0.25)

# add legend
legend("bottomright", legend=c('Logistic Regression', 'Random Forest', 'kNN', 'Decision Tree'),
       col=c("lightcoral", "goldenrod1", "darkolivegreen4", "cornflowerblue"), lwd=4, cex = 0.9, 
       text.col=c("lightcoral", "goldenrod1", "darkolivegreen4", "cornflowerblue"), bty="n")










# Put the test metrics in a table
# accuracy
test.accuracy <- round(rbind(test.lr.confusion$overall[1],
                             test.knn.confusion$overall[1],
                             test.rpart.confusion$overall[1],
                             test.rf.confusion$overall[1]), 3)
# kappa
test.kappa <- round(rbind(test.lr.confusion$overall[2],
                          test.knn.confusion$overall[2],
                          test.rpart.confusion$overall[2],
                          test.rf.confusion$overall[2]), 3)
# f1 score
test.f1 <- round(rbind(test.lr.confusion$byClass[7],
                       test.knn.confusion$byClass[7],
                       test.rpart.confusion$byClass[7],
                       test.rf.confusion$byClass[7]), 3)
# metrics table
test.metrics <- data.frame(cbind(test.accuracy, test.kappa, test.f1))

# Add a column to indicate these metrics are Train or Test set
test.metrics$Dataset <- c('Test', 'Test', 'Test', 'Test')

# rename the rows to the names of each model
rownames(test.metrics) <- c('Logistic Regression',
                            'kNN',
                            'Decision Tree',
                            'Random Forest')
# use the rownames as a variable
test.metrics$Model <- rownames(test.metrics)

# combine train metrics & test metrics
metrics <- rbind(metrics, test.metrics)

# Plot Train and test metrics side by side for comparison
# accuracy
ggplot(metrics, aes(x=factor(reorder(metrics$Model, metrics$Accuracy)), y=metrics$Accuracy,
                    fill=factor(metrics$Dataset))) +
  geom_bar(stat = "identity", position = position_dodge(width = .5)) +
  coord_cartesian(ylim = c(0, 1.1)) +
  ggtitle("Train & Test set Accuracy") +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(y='Accuracy', x='Models') +
  labs(fill = "Dataset") +
  theme(legend.justification = "top") +
  geom_hline(yintercept=1,
             linetype='dashed',
             color='red',
             size=0.5)
# Kappa
ggplot(metrics, aes(x=factor(reorder(metrics$Model, metrics$Kappa)), y=metrics$Kappa,
                    fill=factor(metrics$Dataset))) +
  geom_bar(stat = "identity", position = position_dodge(width = .5)) +
  coord_cartesian(ylim = c(0, 1.1)) +
  ggtitle("Train & Test set Kappa") +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(y='Kappa', x='Models') +
  labs(fill = "Dataset") +
  theme(legend.justification = "top") +
  geom_hline(yintercept=1,
             linetype='dashed',
             color='red',
             size=0.5)
# Kappa
ggplot(metrics, aes(x=factor(reorder(metrics$Model, metrics$F1)), y=metrics$F1,
                    fill=factor(metrics$Dataset))) +
  geom_bar(stat = "identity", position = position_dodge(width = .5)) +
  coord_cartesian(ylim = c(0, 1.1)) +
  ggtitle("Train & Test set F1 score") +
  theme(plot.title = element_text(hjust = 0.5)) +
  labs(y='F1', x='Models') +
  labs(fill = "Dataset") +
  theme(legend.justification = "top") +
  geom_hline(yintercept=1,
             linetype='dashed',
             color='red',
             size=0.5)




































###### Decision Boundaries of each model for visualization ######
# Put everything in functions for plotting decision boundary of each model
nbp <- 1000 # resolution for plotting decision line. The higher the smoother the line
PredA <- seq(min(cleaned.train$X.follows), max(cleaned.train$X.follows), length = nbp)
PredB <- seq(min(cleaned.train$X.followers), max(cleaned.train$X.followers), length = nbp)
Grid <- expand.grid(X.follows = PredA, X.followers = PredB)

# function to plot the scatterplot and decision boundary line
PlotGrid <- function(pred,title) {

  pts <- (ggplot(data = cleaned.train, aes(x = X.follows, y = X.followers,
                                      color = fake)) +
            geom_contour(data = cbind(Grid, fake = pred), aes(z = as.numeric(fake)),
                         color = "red", breaks = c(1.5)) +
            geom_point(size = 4, alpha = .5) +
            ggtitle("Decision boundary") +
            theme(legend.text = element_text(size = 10))) +
    scale_x_continuous() +
    scale_y_continuous()

  grid.arrange(pts, top = textGrob(title, gp = gpar(fontsize = 20)))

}

# function to train models with 10-fold cv repeats 5 times
seed <- 1234
folds <- 10
repeats <- 5
control <- trainControl(method = "repeatedcv",
                          number = folds,
                          repeats = repeats)

# function to store accuracy of each iteration in cross-validation
accuracy <- function(Model, Name) {
  accuracy.df <- data.frame(t(postResample(predict(Model, newdata = cleaned.train), cleaned.train[["fake"]])),
                     Resample = "None", model = Name)
  rbind(accuracy.df, data.frame(Model$resample, model = Name))
}

accuracy.df <- data.frame()

# function to train and plot decision boundary of each model in 1 step
train.display <- function (accuracy.df, Name, Formula, Method, ...) {
  set.seed(seed)
  Model <- train(as.formula(Formula), data = cleaned.train, method = Method, trControl = control, ...)
  Pred <- predict(Model, newdata = Grid)
  PlotGrid(Pred, Name)
  accuracy.df <- rbind(accuracy.df, accuracy(Model, Name))
}









###### Train and Plot decision boundary of each model ######
###### (only using 2 variables for showcasing - cannot plot more than 3 variables) ######

# Logistic Regression
lr.boundary <- train.display(accuracy.df, "Logistic Regression", "fake ~ X.follows+X.followers", "glm")

# Decision Tree
rpart.boundary <- train.display(accuracy.df, "Decision Tree", "fake ~ X.follows+X.followers", "rpart")

# kNN
acc.kNN <- data.frame()
kNN <- 9
for (k in kNN) {
  acc.kNN <- train.display(acc.kNN, sprintf("k-NN with k=%i", k),
                           "fake ~ X.follows+X.followers", "knn", tuneGrid = data.frame(k = c(k)))
}

# Random Forest
rf.boundary <- train.display(accuracy.df, "Random Forest", "fake ~ X.follows+X.followers", "rf")

