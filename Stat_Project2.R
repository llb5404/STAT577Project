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
#set.seed(60)

##split into training and testing data
split2 = sample.split(data,SplitRatio = 0.75)
data_test <-subset(data, split2 =="TRUE")
data_train <-subset(data, split2 =="FALSE")

#initial untuned model
#initial mtry ~ sqrt(23) 
start.time_model1 <- Sys.time() 

model1 = randomForest(DRK_YN ~ ., data = data_train, ntree = 500, mtry = 5, importance = TRUE)

end.time_model1 <- Sys.time()

time.taken_model1 <- round(end.time_model1 - start.time_model1,2)
time.taken_model1

print(model1)
imp = importance(model1) #ranks importance of features
varImpPlot(model1, sort=TRUE, main="Variable Importance Plot") #plots importance ranking of features


##model tuning - mtry
#set.seed(60)

datatune <- data_train[ , ! names(data_train) %in% c("hear_left")] #leave classifier out to avoid error

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid") #establish control, 10-fold validation

tunegrid <- expand.grid(.mtry=c(1:15)) #mtry with gridsearch - create grid to test 15 values of mtry (1-15)

#record time for gridsearch
start.time_gs <- Sys.time() 

rf_gridsearch <- train(DRK_YN ~ ., data=data_train, method="rf", metric="Accuracy", tuneGrid=tunegrid, trControl=control) #find best mtry based on accuracy

end.time_gs <- Sys.time()

time.taken_gs <- round(end.time_gs - start.time_gs,2)
time.taken_gs

mtryopt <- rf_gridsearch$bestTune$mtry

print(rf_gridsearch)
plot(rf_gridsearch) #plot mtry vs accuracy



#set.seed(60)

start.time_new <- Sys.time() 

modelnew = randomForest(DRK_YN ~.-hear_left -hear_right -urine_protein, data = data_train, ntree = 500, mtry = mtryopt, importance = TRUE)

end.time_new <- Sys.time()

time.taken_new <- round(end.time_new - start.time_new,2)
time.taken_new

print(modelnew)

##compare prediction rates:
#Test Data
  #new model
  predictions_new_test <- as.data.frame(predict(modelnew, data_test, type = "prob"))
  predict_new <- predict(modelnew, data_test)
  confusionMatrix(predict_new, data_test$DRK_YN)
  
  predictions_new_test$predict <- names(predictions_new_test)[1:2][apply(predictions_new_test[,1:2],1,which.max)]
  predictions_new_test$observed <- data_test$DRK_YN
  head(predictions_new_test) #shows first few rows of predictions
  roc.new <- roc(ifelse(predictions_new_test$observed == "Y","Y","N"), as.numeric(predictions_new_test$Y))
  
  #old model
  predictions_old_test <- as.data.frame(predict(model1, data_test, type = "prob"))
  predict_old <- predict(model1, data_test)
  confusionMatrix(predict_old, data_test$DRK_YN)
  
  predictions_old_test$predict <- names(predictions_old_test)[1:2][apply(predictions_old_test[,1:2],1,which.max)]
  predictions_old_test$observed <- data_test$DRK_YN
  head(predictions_old_test) #shows first few rows of predictions
  roc.old <- roc(ifelse(predictions_old_test$observed == "Y","Y","N"), as.numeric(predictions_old_test$Y))

#Train Data
  #new model
  predictions_new_train <- as.data.frame(predict(modelnew, data_train, type = "prob"))
  predict_new_tr <- predict(modelnew, data_train)
  confusionMatrix(predict_new_tr, data_train$DRK_YN)
  
  predictions_new_train$predict <- names(predictions_new_train)[1:2][apply(predictions_new_train[,1:2],1,which.max)]
  predictions_new_train$observed <- data_train$DRK_YN
  head(predictions_new_train) #shows first few rows of predictions
  roc.new_tr <- roc(ifelse(predictions_new_train$observed == "Y","Y","N"), as.numeric(predictions_new_train$Y))

  #old model
  predictions_old_train <- as.data.frame(predict(model1, data_train, type = "prob"))
  predict_old_tr <- predict(model1, data_train)
  confusionMatrix(predict_old_tr, data_train$DRK_YN)
  
  predictions_old_train$predict <- names(predictions_old_train)[1:2][apply(predictions_old_train[,1:2],1,which.max)]
  predictions_old_train$observed <- data_train$DRK_YN
  head(predictions_old_train)
  roc.old_tr <- roc(ifelse(predictions_old_train$observed == "Y","Y","N"), as.numeric(predictions_old_train$Y))


plot(roc.new, col="red", lwd=3, main="New RF ROC curve Test")
plot(roc.old, col="blue", lwd=3, main="Old RF ROC curve Test")

plot(roc.new_tr, col="red", lwd=3, main="New RF ROC curve Train")
plot(roc.old_tr, col="blue", lwd=3, main="Old RF ROC curve Train")

## EXTREME GRADIENT BOOSTING
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
#set.seed(60)

##split into training and testing data
split2 = sample.split(data,SplitRatio = 0.75)
data_test <-subset(data, split2 =="TRUE")
data_train <-subset(data, split2 =="FALSE")

#set.seed(60)

#stochastic gradient model
start.time_sg <- Sys.time() 

xgmodel <- train(
  DRK_YN ~., data = data_train, method = "xgbTree",
  trControl = trainControl("cv", number = 10) #parameters optimized via 10-fold validation
)

end.time_sg <- Sys.time()

time.taken_sg <- round(end.time_sg - start.time_sg,2)
time.taken_sg

xgmodel$bestTune #displays best tuning parameters
varImp(xgmodel) #displays variable importance in descending order
confusionMatrix(xgmodel) #displays confusion matrix

start.time_sg_new <- Sys.time() 

xgmodel_new <- train(
  DRK_YN ~.-hear_left -hear_right -urine_protein, data = data_train, method = "xgbTree",
  trControl = trainControl("cv", number = 10) #parameters optimized via 10-fold validation
)

end.time_sg_new <- Sys.time()

time.taken_sg_new <- round(end.time_sg_new - start.time_sg_new,2)
time.taken_sg_new

xgmodel_new$bestTune #displays best tuning parameters
varImp(xgmodel_new) #displays variable importance in descending order
confusionMatrix(xgmodel_new) #displays confusion matrix


#prediction rates

  #test
  xgpredictions <- as.data.frame(predict(xgmodel, data_test, type = "prob"))
  xgpredict <- predict(xgmodel, data_test)
  confusionMatrix(xgpredict, data_test$DRK_YN)
  
  xgpredictions$predict <- names(xgpredictions)[1:2][apply(xgpredictions[,1:2],1,which.max)]
  xgpredictions$observed <- data_test$DRK_YN
  head(xgpredictions)
  roc.xg <- roc(ifelse(xgpredictions$observed == "Y","Y","N"), as.numeric(xgpredictions$Y))
  
  #train
  xgpredictions_train <- as.data.frame(predict(xgmodel, data_train, type = "prob"))
  xgpredict_train <- predict(xgmodel, data_train)
  confusionMatrix(xgpredict_train, data_train$DRK_YN)
  
  xgpredictions_train$predict <- names(xgpredictions_train)[1:2][apply(xgpredictions_train[,1:2],1,which.max)]
  xgpredictions_train$observed <- data_train$DRK_YN
  head(xgpredictions_train)
  roc.xg_train <- roc(ifelse(xgpredictions_train$observed == "Y","Y","N"), as.numeric(xgpredictions_train$Y))
  
  #test
  xgpredictions_new <- as.data.frame(predict(xgmodel_new, data_test, type = "prob"))
  xgpredict_new <- predict(xgmodel_new, data_test)
  confusionMatrix(xgpredict_new, data_test$DRK_YN)
  
  xgpredictions_new$predict <- names(xgpredictions_new)[1:2][apply(xgpredictions_new[,1:2],1,which.max)]
  xgpredictions_new$observed <- data_test$DRK_YN
  head(xgpredictions_new)
  roc.xg_new <- roc(ifelse(xgpredictions_new$observed == "Y","Y","N"), as.numeric(xgpredictions_new$Y))
  
  #train
  xgpredictions_new_train <- as.data.frame(predict(xgmodel_new, data_train, type = "prob"))
  xgpredict_new_train <- predict(xgmodel_new, data_train)
  confusionMatrix(xgpredict_new_train, data_train$DRK_YN)
  
  xgpredictions_new_train$predict <- names(xgpredictions_new_train)[1:2][apply(xgpredictions_new_train[,1:2],1,which.max)]
  xgpredictions_new_train$observed <- data_train$DRK_YN
  head(xgpredictions_new_train)
  roc.xg_new_train <- roc(ifelse(xgpredictions_new_train$observed == "Y","Y","N"), as.numeric(xgpredictions_new_train$Y))


plot(roc.xg, col="blue", lwd=3, main="Old XGradient ROC curve Test")
plot(roc.xg_train, col="blue", lwd=3, main="Old XGradient ROC curve Train")
plot(roc.xg_new, col="red", lwd=3, main="New XGradient ROC curve Test")
plot(roc.xg_new_train, col="red", lwd=3, main="New XGradient ROC curve Train")



