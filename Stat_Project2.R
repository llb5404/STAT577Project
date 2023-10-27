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



library(dplyr)                                
library(caTools)
library(randomForest)
library(caret)


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

datatune <- data_train[ , ! names(data_train) %in% c("DRK_YN")] #leave classifier out to avoid error
mtry <- tuneRF(datatune,data_train$DRK_YN, ntreeTry=500,
               stepFactor=1.5,improve=1e-5, trace=TRUE, plot=TRUE) #first try tuning with tuneRF
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid") #establish control, 10-fold validation

tunegrid <- expand.grid(.mtry=c(1:15)) #mtry with gridsearch - create grid to test 15 values of mtry (1-15)

rf_gridsearch <- train(DRK_YN ~ ., data=data_train, method="rf", metric="Accuracy", tuneGrid=tunegrid, trControl=control) #find best mtry based on accuracy
print(rf_gridsearch) #best mtry = 5 -- initial guess
plot(rf_gridsearch) #plot mtry vs accuracy

##model tuning - features

control2 = rfeControl(functions = rfFuncs, 
                      method = "repeatedcv", 
                      repeats = 3, 
                      number = 10) #establish control, 10-fold validation


result_rfe1 <- rfe(x = datatune, 
                   y = data_train$DRK_YN, 
                   sizes = c(1:23),
                   rfeControl = control2) #test all combinations of 23 features

# Print the results
result_rfe1 

# Print the selected features
predictors(result_rfe1) ## 20 predictors selected

# Print the results visually
ggplot(data = result_rfe1, metric = "Accuracy") + theme_bw()

set.seed(60)

modelnew = randomForest(DRK_YN ~. - hear_left -hear_right, data = data_train, ntree = 500, mtry = 5, importance = TRUE)
print(modelnew)



print(mtry)


