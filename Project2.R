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
