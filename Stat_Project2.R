library(caTools)
library(MASS)
#Divide the data into training and test set
df <- read.csv("proj2.csv") # Reads in Dataset, add  your own link to the file:
# df <- read.csv(...) # Reads in Dataset, add  your own link to the file:
sample <- sample.split(df$age, SplitRatio = 0.75)
df$DRK_YN <- as.factor(df$DRK_YN)
train  <- subset(df, sample == TRUE)
test   <- subset(df, sample == FALSE)

# Perform Logistic Regression, use variable selection or dimension reduction if necessary.
#Full Model:

full.model <- glm(as.factor(DRK_YN) ~., data = train, family = binomial)
coef(full.model)


library(MASS)
step.model <- full.model %>% stepAIC(trace = FALSE)
coef(step.model)


## RANDOM FOREST
library(dplyr)                                
library(caTools)
library(randomForest)
library(caret)
library(pROC)


data = read.csv("proj2.csv")

##choose only 5000 samples for faster computation
data <- subset(data[1:5000,])

#turn qualitative variables into quantitative variables
data$sex <- as.factor(data$sex)
data$DRK_YN <- as.factor(data$DRK_YN)

#seed = 60 for model creation, model tuning, final model
set.seed(60)

##split into training and testing data
split2 = sample.split(data,SplitRatio = 0.75)
data_test <-subset(data, split2 =="TRUE")
data_train <-subset(data, split2 =="FALSE")

#initial untuned model
#initial mtry ~ sqrt(23) 
model1 = randomForest(DRK_YN ~ ., data = data_train, ntree = 500, mtry = 5, importance = TRUE)
print(model1)
imp = importance(model1) #ranks importance of features
varImpPlot(model1, sort=TRUE, main="Variable Importance Plot") #plots importance ranking of features


##model tuning - mtry
set.seed(60)

datatune <- data_train[ , ! names(data_train) %in% c("DRK_YN")] #leave classifier out to avoid error

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid") #establish control, 10-fold validation

tunegrid <- expand.grid(.mtry=c(1:15)) #mtry with gridsearch - create grid to test 15 values of mtry (1-15)

rf_gridsearch <- train(DRK_YN ~ ., data=data_train, method="rf", metric="Accuracy", tuneGrid=tunegrid, trControl=control) #find best mtry based on accuracy
mtryopt <- rf_gridsearch$bestTune$mtry

print(rf_gridsearch)
plot(rf_gridsearch) #plot mtry vs accuracy



set.seed(60)

modelnew = randomForest(DRK_YN ~., data = data_train, ntree = 500, mtry = mtryopt, importance = TRUE)
print(modelnew)

##compare prediction rates:
predictions_new_test <- as.data.frame(predict(modelnew, data_test, type = "prob"))
predict_new <- predict(modelnew, data_test)
confusionMatrix(predict_new, data_test$DRK_YN)

predictions_new_test$predict <- names(predictions_new_test)[1:2][apply(predictions_new_test[,1:2],1,which.max)]
predictions_new_test$observed <- data_test$DRK_YN
head(predictions_new_test)
roc.new <- roc(ifelse(predictions_new_test$observed == "Y","Y","N"), as.numeric(predictions_new_test$Y))

predictions_old_test <- as.data.frame(predict(model1, data_test, type = "prob"))
predict_old <- predict(model1, data_test)
confusionMatrix(predict_old, data_test$DRK_YN)

predictions_old_test$predict <- names(predictions_old_test)[1:2][apply(predictions_old_test[,1:2],1,which.max)]
predictions_old_test$observed <- data_test$DRK_YN
head(predictions_old_test)
roc.old <- roc(ifelse(predictions_old_test$observed == "Y","Y","N"), as.numeric(predictions_old_test$Y))



plot(roc.new, col="red", lwd=3, main="New RF ROC curve Test")
plot(roc.old, col="blue", lwd=3, main="Old RF ROC curve Test")

## STOCHASTIC GRADIENT BOOSTING
library(xgboost)
library(caret)
library(pROC)
data = read.csv("proj2.csv")

##choose only 5000 samples for faster computation
data <- subset(data[1:5000,])

#turn qualitative variables into quantitative variables
data$sex <- as.factor(data$sex)
data$DRK_YN <- as.factor(data$DRK_YN)

#seed = 60 for model creation, model tuning, final model
set.seed(60)

##split into training and testing data
split2 = sample.split(data,SplitRatio = 0.75)
data_test <-subset(data, split2 =="TRUE")
data_train <-subset(data, split2 =="FALSE")

set.seed(60)

#stochastic gradient model
xgmodel <- train(
  DRK_YN ~., data = data_train, method = "xgbTree",
  trControl = trainControl("cv", number = 10) #parameters optimized via 10-fold validation
)

xgmodel$bestTune #displays best tuning parameters
varImp(xgmodel) #displays variable importance in descending order
confusionMatrix(xgmodel) #displays confusion matrix


#prediction rates
xgpredictions <- as.data.frame(predict(xgmodel, data_test, type = "prob"))
xgpredict <- predict(xgmodel, data_test)
confusionMatrix(xgpredict, data_test$DRK_YN)

xgpredictions$predict <- names(xgpredictions)[1:2][apply(xgpredictions[,1:2],1,which.max)]
xgpredictions$observed <- data_test$DRK_YN
head(xgpredictions)
roc.xg <- roc(ifelse(xgpredictions$observed == "Y","Y","N"), as.numeric(xgpredictions$Y))


plot(roc.xg, col="red", lwd=3, main="Stochastic Gradient ROC curve Test")

