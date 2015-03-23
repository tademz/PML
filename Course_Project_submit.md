# Course Project

Declare some global variables.


```r
set.seed(383838)
library(ggplot2); 
p <- 0.75 # partition ratio
k <- 10 # fold number
```

# Data processing

Load the __training__ data set into `data.frame` and remove the rows with `new_window=="yes"`. Convert the class of `user_name` and `classe` into `factor`. 


```r
data <- read.csv('train.csv', as.is=TRUE, na.strings=c('', NA))
data <- data[data$new_window == "no", ]
data$classe <- factor(data$classe)
data$user_name <- factor(data$user_name)
```

Resulting dataset contains 19216 rows and 160 columns.

Dataset contains large number of summary columns. Since summary rows have been removed those columns contain only `NA` values and are no longer relevant and can be removed.


```r
keepCol <- complete.cases(t(data))
data <- data[, keepCol]
```

Several variables (like `num_window` or `cvtd_timestamp`) show clear functional relation with `classe` for each user. These variables are clearly related to the data collection process and I decided not to use any of these (as well as the user id) for my model.


```r
names(data[,1:7])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"
```

```r
data <- data[, 8:60]
```

From the resulting dataset I removed highly correlated variables using `caret::findCorrelation` function.


```r
# exclude highly-correlated columns
colExclude <- caret::findCorrelation(cor(data[, -ncol(data)]), cutoff=0.8)
data <- data[, -colExclude]
```

Only 41 columns left for training models. 
Dataset has been splited into train and test dataset using `caret::createDataPartition` with `p` equal to 0.75.


```r
inTrain <- caret::createDataPartition(data$classe, p=p, list=FALSE)
training <- data[inTrain, ]
testing <- data[-inTrain, ]
```

Exploratory data analysis on the training data set showed a trend of multiple class-based clusters. Random forests might be suitable for training models.


```r
gridExtra::grid.arrange(
    qplot(x=accel_forearm_x, y=yaw_forearm, data = training,col=classe),
    qplot(x=magnet_dumbbell_x, y=yaw_dumbbell,data = training,col=classe),
    qplot(x=total_accel_arm, y=yaw_arm,data = training,col=classe),
    qplot(x=roll_arm, y=pitch_arm,data = training,col=classe),
    ncol=2
)
```

![plot of chunk unnamed-chunk-7](./Course_Project_submit_files/figure-html/unnamed-chunk-7.png) 

# Model building and Cross-validation

First, I tried to train models with trees.


```r
# setting the attribute 'number' as k, for k-folds validation.
# savePred will save all the predictions in each fold, allowing us to check what actually happend while training.
ctrl <- caret::trainControl(method="cv", allowParallel = T, savePred=T, classProb=T, number = k)
modelFitRpart <- caret::train(classe~., data=training, method = "rpart", trControl = ctrl)
```

```
## Loading required package: rpart
## Loading required package: lattice
```

```r
head(modelFitRpart$pred)
```

```
##   pred obs      A       B C D E rowIndex      cp Resample
## 1    A   A 0.9541 0.04586 0 0 0        3 0.03016   Fold01
## 2    A   A 0.9541 0.04586 0 0 0        8 0.03016   Fold01
## 3    A   A 0.9541 0.04586 0 0 0       11 0.03016   Fold01
## 4    A   A 0.9541 0.04586 0 0 0       22 0.03016   Fold01
## 5    A   A 0.9541 0.04586 0 0 0       37 0.03016   Fold01
## 6    A   A 0.9541 0.04586 0 0 0       52 0.03016   Fold01
```

```r
predictionRpart <- predict(modelFitRpart, testing)
cmatrixRpart <- caret::confusionMatrix(predictionRpart, testing$classe)
cmatrixRpart$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      4.988e-01      3.460e-01      4.845e-01      5.130e-01      2.847e-01 
## AccuracyPValue  McnemarPValue 
##     5.948e-214      0.000e+00
```

However, the result is not so satisfying (the accuracy is 0.4988). Next, I proceed to build a model with random forest.


```r
# setting the attribute 'number' as k, for k-folds validation.
# allowParallel=T will reduce the time for training. 
# verboseIter=T will show the log of progress (taken out for the conciseness of the report).
ctrl <- trainControl(method = "cv", number = k, allowParallel = T, savePred=T, classProb=T)
modelFitRF <- caret::train(classe ~ ., data=training, method = "rf", trControl = ctrl)
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
head(modelFitRF$pred)
```

```
##   pred obs     A B C D     E rowIndex mtry Resample
## 1    A   A 1.000 0 0 0 0.000        7    2   Fold01
## 2    A   A 0.998 0 0 0 0.002       17    2   Fold01
## 3    A   A 0.998 0 0 0 0.002       34    2   Fold01
## 4    A   A 1.000 0 0 0 0.000       35    2   Fold01
## 5    A   A 0.996 0 0 0 0.004       49    2   Fold01
## 6    A   A 1.000 0 0 0 0.000       55    2   Fold01
```

```r
predictionRF <- predict(modelFitRF, testing)
cmatrixRF <- caret::confusionMatrix(predictionRF, testing$classe)
cmatrixRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1367    4    0    0    0
##          B    0  923   10    0    0
##          C    0    2  826   13    0
##          D    0    0    2  772    3
##          E    0    0    0    1  879
## 
## Overall Statistics
##                                        
##                Accuracy : 0.993        
##                  95% CI : (0.99, 0.995)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.991        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.994    0.986    0.982    0.997
## Specificity             0.999    0.997    0.996    0.999    1.000
## Pos Pred Value          0.997    0.989    0.982    0.994    0.999
## Neg Pred Value          1.000    0.998    0.997    0.997    0.999
## Prevalence              0.285    0.193    0.175    0.164    0.184
## Detection Rate          0.285    0.192    0.172    0.161    0.183
## Detection Prevalence    0.286    0.194    0.175    0.162    0.183
## Balanced Accuracy       0.999    0.995    0.991    0.990    0.998
```

The results of random forest (`rf`) are better than trees (`rpart`) because the accuracy of using `rf` is 0.9927) with the 95% confidence interval for accuracy between (0.9899, 0.9949).

# Final results

Use the best model of random forest one on the testing data set, and examine its performance. 


```r
predictionFinal <- predict(modelFitRF, newdata = testing)
cmatrixFinal <- caret::confusionMatrix(predictionFinal, testing$classe)
cmatrixFinal
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1367    4    0    0    0
##          B    0  923   10    0    0
##          C    0    2  826   13    0
##          D    0    0    2  772    3
##          E    0    0    0    1  879
## 
## Overall Statistics
##                                        
##                Accuracy : 0.993        
##                  95% CI : (0.99, 0.995)
##     No Information Rate : 0.285        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.991        
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.994    0.986    0.982    0.997
## Specificity             0.999    0.997    0.996    0.999    1.000
## Pos Pred Value          0.997    0.989    0.982    0.994    0.999
## Neg Pred Value          1.000    0.998    0.997    0.997    0.999
## Prevalence              0.285    0.193    0.175    0.164    0.184
## Detection Rate          0.285    0.192    0.172    0.161    0.183
## Detection Prevalence    0.286    0.194    0.175    0.162    0.183
## Balanced Accuracy       0.999    0.995    0.991    0.990    0.998
```

The final model allows us to make predictions with the accuracy of 99.27 %. The 95% confidence interval for accuracy is between (0.9899, 0.9949).

# Making predictions for unknown data


```r
testingDS <- read.csv("test.csv", as.is=TRUE, na.strings=c('', NA))
answers <- predict(modelFitRF,testingDS)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
