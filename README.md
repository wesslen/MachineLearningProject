# MachineLearningProject
R Code used for Machine Learning Project on CMPD Clearance Rates.

## Part 1: Set Working Directory, Load Model Dataset and Create Factors

### Set Working Directory
The user should set his or her working directory to where the model dataset is located.

```r
setwd("C:\\Users\\Gabriel\\Dropbox\\Machine Learning Project\\Data")
#setwd("C:/Users/rwesslen/Dropbox/Machine Learning Project/Data")
#setwd("~/Dropbox/Machine Learning Project/Data")
```

### Initiate Libraries 

```r
options(java.parameters = "-Xmx8192m")  #Increase memory -- you may want to ignore/comment out
library(rJava) 
library(ggplot2)
library(caret)
library(tabplot)
library(rpart)				        # Popular decision tree algorithm
library(rattle)					      # Fancy tree plot
library(rpart.plot)				    # Enhanced tree plots
library(RColorBrewer)			  	# Color selection for fancy tree plot
library(ebdbNet)
library(RWeka)
library(FSelector)              # Filter Variable Importance
library(e1071)                # SVM
library(spgwr)                # SWR (Spatially Weighted Regression)
library(kknn)                 # kNN
library(neuralnet)            # Neural Network
library(ROCR)
library(smbinning) 
library(ebdbNet)
library(maptools)
```

### Upload Dataset and Create Factors
```r
  Incident <- read.csv("INCIDENT.csv", head=TRUE)
  
  # dependent variable (Clear_Flag)
  outcome<-"Clear_Flag"

  # convert categorical variables to factors
  # Note: factors could have been automatically loaded as factors
  #       with the read.csv option "stringsAsFactors"
  
  Incident$Clear_Flag <- factor(Incident$ClearFlag, levels = 0:1, labels = c("No", "Yes"))
  Incident$YEAR <- factor(Incident$YEAR)
  Incident$ZipCode <- factor(Incident$ZipCode)
  
  Incident$Division <- factor(Incident$Division)
  Incident$RankPopDensity2010 <- factor(Incident$RankPopDensity2010)
  Incident$RankYouthPop2012 <- factor(Incident$RankYouthPop2012)
  Incident$RankWhitePop2010 <- factor(Incident$RankWhitePop2010)
  Incident$RankBachDeg2010 <- factor(Incident$RankBachDeg2010)
  Incident$RankVotPart2012 <- factor(Incident$RankVotPart2012)
  Incident$RankGrocProx2011 <- factor(Incident$RankGrocProx2011)
  Incident$RankTreeCanopy2012 <- factor(Incident$RankTreeCanopy2012)
  Incident$RankPublicAsst2010 <- factor(Incident$RankPublicAsst2010)
  Incident$RankAnimalControl2011 <- factor(Incident$RankAnimalControl2011)
  Incident$RankHHI2012 <- factor(Incident$RankHHI2012)
  Incident$RankHPI2012 <- factor(Incident$RankHPI2012)
  Incident$Black_Victim <- factor(Incident$Black_Victim)
  Incident$White_Victim <- factor(Incident$White_Victim)
  
  Incident$VULNERSUBST_FLAG <- factor(Incident$VULNERSUBST_FLAG)
  Incident$REFUSED_TREAT_FLAG <- factor(Incident$REFUSED_TREAT_FLAG)
  Incident$RankPropertyValue <- factor(Incident$RankPropertyValue)
  
  Incident$Within_Family_Victim_Flag <- factor(Incident$Within_Family_Victim_Flag)
  Incident$Outside_Family_Victim_Flag <- factor(Incident$Outside_Family_Victim_Flag)
  Incident$Unknown_Victim_Flag <- factor(Incident$Unknown_Victim_Flag)
  
  Incident$BUSINESS_FLAG <- factor(Incident$BUSINESS_FLAG)
  Incident$PUBLIC_FLAG <- factor(Incident$PUBLIC_FLAG)
  Incident$GOVT_FLAG <- factor(Incident$GOVT_FLAG)
  Incident$FIN_FLAG <- factor(Incident$FIN_FLAG)
  Incident$RELG_FLAG <- factor(Incident$RELG_FLAG)
  Incident$WALMART_FLAG <- factor(Incident$WALMART_FLAG)
  
  Incident$NCSTATE_FLAG <- factor(Incident$NCSTATE_FLAG)
  Incident$CHAR_FLAG <- factor(Incident$CHAR_FLAG)
  
  Incident$AddressSameFlag <- factor(Incident$AddressSameFlag)
  Incident$AddressReportFlag <- factor(Incident$AddressReportFlag)
  Incident$ReportByOfficerFlag <- factor(Incident$ReportByOfficerFlag)
  Incident$NameWithheldFlag <- factor(Incident$NameWithheldFlag)
  Incident$HomelessProxFlag <- factor(Incident$HomelessProxFlag)
  Incident$SchoolProxFlag <- factor(Incident$SchoolProxFlag)
  Incident$ChurchProxFlag <- factor(Incident$ChurchProxFlag)
  
  Incident$WinterWeatherFlag <- factor(Incident$WinterWeatherFlag)
  Incident$SevereWeatherFlag <- factor(Incident$SevereWeatherFlag)
  ```
  
### Check for near-zero variance variables
This is a check for any near-constant categorical variables. This uses a function from the caret package.
  
```r
nzv <- nearZeroVar(Incident, saveMetrics= TRUE)
nzv[nzv$nzv,][1:20,]
```
  
### Identify Candidate Predictor variables
  
```r
predictors <-c("CSS_Called","Place1","NIBRS_Hi_Class","Against",
  "Place2","Category",
  "Location_Type","Long","Lat","YEAR","WEEK","DAYOFWEEK","HOUR",
  "RankPopDensity2010","RankYouthPop2012","RankWhitePop2010","RankBachDeg2010",
  "RankVotPart2012","RankGrocProx2011","RankTreeCanopy2012","RankPublicAsst2010",
  "RankAnimalControl2011","RankHHI2012","RankHPI2012","Black_Victim","White_Victim",
  "Victim_Age_Binned","VULNERSUBST_FLAG","REFUSED_TREAT_FLAG",
  "RankPropertyValue","Within_Family_Victim_Flag","Outside_Family_Victim_Flag","Unknown_Victim_Flag",
  "RollSevenDayNorm","BUSINESS_FLAG","WALMART_FLAG","NCSTATE_FLAG","CHAR_FLAG","PUBLIC_FLAG",
  "AddressSameFlag","AddressReportFlag","ReportByOfficerFlag","NameWithheldFlag",
  "HomelessProxFlag","SchoolProxFlag","ChurchProxFlag","WinterWeatherFlag","SevereWeatherFlag")
```
  
## Part 2: Partition Data, Explore Data & Variable Importance using Chi-Squared
  
### Partition the Data into Training, Validation and Test Partition
```r
  # Create a 70 / 30 Training / Validation Partition
  set.seed(1234)
  inTrain <- createDataPartition(Incident$ClearFlag, p=0.7, list=FALSE) #inTrain has about 142,311 rows. 
  train <- Incident[inTrain,] #
  rest_of_set <- Incident[-inTrain,]

  # This will create another partition of the 30% of the data, so ~20%-validation and ~10%-test
  inValidation <- createDataPartition(rest_of_set$ClearFlag, p=0.71, list=FALSE)
  valid <- rest_of_set[inValidation,]
  test <- rest_of_set[-inValidation,]
```
  
### Exploratory analysis with tabplot

```r
#Crime Type
coltoview <-c("NIBRS_Hi_Class","Category","Against","Group")

tableplot(train[,c(outcome,coltoview)])

#Location
coltoview <- c("CSS_Called","Place1","Place2","Location_Type","Long","Lat","RollSevenDayNorm")

tableplot(train[,c(outcome,coltoview)])

#Location Ranks
coltoview <- c("RankPopDensity2010","RankYouthPop2012","RankWhitePop2010","RankBachDeg2010",
"RankVotPart2012","RankGrocProx2011","RankTreeCanopy2012","RankPublicAsst2010",
"RankAnimalControl2011","RankHHI2012")

tableplot(train[,c(outcome,coltoview)])


coltoview <- c("Black_Victim","White_Victim",
"Victim_Age","Victim_Age_Binned","VULNERSUBST_FLAG","REFUSED_TREAT_FLAG")

tableplot(train[,c(outcome,coltoview)])

coltoview <- c("RankPropertyValue","AddressSameFlag","AddressReportFlag","ReportByOfficerFlag","NameWithheldFlag",
"HomelessProxFlag","SchoolProxFlag","ChurchProxFlag")

tableplot(train[,c(outcome,coltoview)])
```

### Chi-Squared for Variable Importance (Filter Method)

```r
weights <- chi.squared(Clear_Flag~., train[-c(1,6,7,13)]) 
print(weights)
```

## Part 3: Non-H2O Modeling

### "Simple" CART

```r
SIMPLEpredictors <- c("Against","PUBLIC_FLAG","RollSevenDayNorm","Victim_Age_Binned","Place1")

fV <- paste(outcome, ' ~ ',paste(c(SIMPLEpredictors),collapse=' + '),sep='')
simpletmodel <-rpart(fV,method = "class", data=train,
                     control=rpart.control(cp=0.001,minsplit=1000,minbucket=1000,maxdepth=5))

# Training
Simpletrainhat <- predict(simpletmodel,newdata=train, type = "class")
Simpledt.train.CM <- confusionMatrix(Simpletrainhat, train$Clear_Flag)
print(Simpledt.train.CM)

# Validation
Simplevalidhat <- predict(simpletmodel,newdata=valid, type = "class")
Simpledt.valid.CM <- confusionMatrix(Simplevalidhat, valid$Clear_Flag)
print(Simpledt.valid.CM)
```

### CART

```r
fV <- paste(outcome, ' ~ ',paste(c(predictors),collapse=' + '),sep='')
tmodel <-rpart(fV,method = "class", data=train,
               control=rpart.control(cp=0.001,minsplit=1000,minbucket=1000,maxdepth=5))

fancyRpartPlot(tmodel) # Better visualization tool for decision trees
print(tmodel)

# Training
trainhat <- predict(tmodel,newdata=train, type = "class")
dt.train.CM <- confusionMatrix(trainhat, train$Clear_Flag)
print(dt.train.CM)

# Validation
validhat <- predict(tmodel,newdata=valid, type = "class")
dt.valid.CM <- confusionMatrix(validhat, valid$Clear_Flag)
print(dt.valid.CM)
```

## Part 4: H2O Modeling

### Initialize H2O and create Three H2o datasets for Training, Validation and Test

```r
#start h2o
suppressPackageStartupMessages(library(h2o))
localH20=h2o.init(nthreads = -1)
#h2o.removeAll()		# only important if already had h2o session running
#h2o.checkClient(localH20)

# create Three (training, valid/validation) h2o datasets
train.hex <-as.h2o(localH20,train)
valid.hex <-as.h2o(localH20,valid)
test.hex <- as.h2o(localH20,test)
```

### H2O Naive Bayes
```r
predictorsNB <-c("Place2","Location_Type","Against","Category",
                 "AddressReportFlag","RankPropertyValue","PUBLIC_FLAG")
fitBayes<-h2o.naiveBayes(x = predictorsNB, y = outcome, training_frame = train.hex, laplace = 3)

# Training
bayes.train.pred <- as.data.frame(h2o.predict(fitBayes,train.hex,type="probs"))
bayes.train.CM <- confusionMatrix(bayes.train.pred$predict,train$Clear_Flag)
print(bayes.train.CM)

# Validation
bayes.valid.pred <- as.data.frame(h2o.predict(fitBayes,valid.hex,type="probs"))
bayes.valid.CM <- confusionMatrix(bayes.valid.pred$predict,valid$Clear_Flag)
print(bayes.valid.CM)
```

### H2O GLM
```r
## Regularization: Lasso (alpha = 1)

fitGLM <-h2o.glm(y = outcome, x = predictors, training_frame = train.hex, 
                 family = "binomial", lambda_search = TRUE, alpha = 1)

# Training
GLM.train.pred <- as.data.frame(h2o.predict(fitGLM,train.hex,type="probs"))
GLM.train.CM <- confusionMatrix(GLM.train.pred$predict,train$Clear_Flag)
print(GLM.train.CM)

# Validation
GLM.valid.pred <- as.data.frame(h2o.predict(fitGLM,valid.hex,type="probs"))
GLM.valid.CM <- confusionMatrix(GLM.valid.pred$predict,valid$Clear_Flag)
print(GLM.valid.CM)

# Variable Importance
glm.varimp <- h2o.varimp(fitGLM)
print(glm.varimp)
```

### H2O GBM
```r
fitgbm<-h2o.gbm(y=outcome,x=predictors, training_frame = train.hex,key="mygbm", 
                distribution = "bernoulli", ntrees = 200,max_depth =5, 
                interaction.depth = 2, learn_rate = 0.2)

# Training
gbm.train.pred <- as.data.frame(h2o.predict(fitgbm,train.hex,type="probs"))
gbm.train.CM <- confusionMatrix(gbm.train.pred$predict,train$Clear_Flag)
print(gbm.train.CM)

# Validation
gbm.valid.pred <- as.data.frame(h2o.predict(fitgbm,valid.hex,type="probs"))
gbm.valid.CM <- confusionMatrix(gbm.valid.pred$predict,valid$Clear_Flag)
print(gbm.valid.CM)

# Variable Importance
gbm.varimp <- h2o.varimp(fitgbm)
print(gbm.varimp) 
```

### H2O Deep Learning
```r
fitDL <- h2o.deeplearning(x = predictors, y = outcome, training_frame = train.hex, 
                          hidden = c(200,200,200),variable_importances = TRUE)

# Training
DL.train.pred <- as.data.frame(h2o.predict(fitDL, train.hex,type="probs"))
DL.train.CM <- confusionMatrix(DL.train.pred$predict,train$Clear_Flag)
print(DL.train.CM)

# Validation
DL.valid.pred <- as.data.frame(h2o.predict(fitDL,valid.hex,type="probs"))
DL.valid.CM <- confusionMatrix(DL.valid.pred$predict,valid$Clear_Flag)
print(DL.valid.CM)

# Variable Importance
dl.varimp <- h2o.varimp(fitDL)
print(dl.varimp)
```

### H2O Random Forests
```r
fitRF <- h2o.randomForest(x = predictors, y = outcome, training_frame = train.hex, 
                          ntree = 50, max_depth = 10, min_rows = 5, nbins = 20)

# Training
RF.train.pred <- as.data.frame(h2o.predict(fitRF, train.hex,type="probs"))
RF.train.CM <- confusionMatrix(RF.train.pred$predict,train$Clear_Flag)
print(RF.train.CM)

# Validation
RF.valid.pred <- as.data.frame(h2o.predict(fitRF,valid.hex,type="probs"))
RF.valid.CM <- confusionMatrix(RF.valid.pred$predict,valid$Clear_Flag)
print(RF.valid.CM)

# Variable Importance
rf.varimp <- h2o.varimp(fitRF)
print(rf.varimp)
```

## Part 5: Predict on Test Datasets

### Prediction of all seven models on Test Dataset
```r
# Simple CART
Simpletesthat <- predict(simpletmodel,newdata=test, type = "class")
Simpledt.test.CM <- confusionMatrix(Simpletesthat, test$Clear_Flag)
print(Simpledt.test.CM)

# CART
testhat <- predict(tmodel,newdata=test, type = "class")
dt.test.CM <- confusionMatrix(testhat, test$Clear_Flag)
print(dt.test.CM)

# Naive Bayes
bayes.test.pred <- as.data.frame(h2o.predict(fitBayes,test.hex,type="probs"))
bayes.test.CM <- confusionMatrix(bayes.test.pred$predict,test$Clear_Flag)
print(bayes.test.CM)

# GLM
GLM.test.pred <- as.data.frame(h2o.predict(fitGLM,test.hex,type="probs"))
GLM.test.CM <- confusionMatrix(GLM.test.pred$predict,test$Clear_Flag)
print(GLM.test.CM)

# GBM
gbm.test.pred <- as.data.frame(h2o.predict(fitgbm,test.hex,type="probs"))
gbm.test.CM <- confusionMatrix(gbm.test.pred$predict,test$Clear_Flag)
print(gbm.test.CM)

# Deep Learning
DL.test.pred <- as.data.frame(h2o.predict(fitDL,test.hex,type="probs"))
DL.test.CM <- confusionMatrix(DL.test.pred$predict,test$Clear_Flag)
print(DL.test.CM)

# Random Forests
RF.test.pred <- as.data.frame(h2o.predict(fitRF,test.hex,type="probs"))
RF.test.CM <- confusionMatrix(RF.test.pred$predict,test$Clear_Flag)
print(RF.test.CM)
```

## Part 6: ROC Curves [To be completed]
