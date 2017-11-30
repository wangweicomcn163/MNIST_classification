#this script is the classifer for digit recognization. 
library(caret)
#=================load training and testing data sets==========================
rawdata <- read.csv('Digit recognizer/train.csv')
#testing <- read.csv('Digit recognizer/test.csv')



#=================split raw data===============================================
rawdata$label <- as.factor(rawdata$label)
set.seed(2017-03-21)
intrain <- createDataPartition(rawdata$label,p=0.8,list = FALSE)
training <- rawdata[intrain,]
testing <- rawdata[-intrain,]



#=================function to display image====================================
display <- function(x){
        x <- unlist(x[,2:785])
        dim(x) <- c(28, 28)
        image(x[, nrow(x):1], axes = FALSE,
              col = grey(seq(0, 1, length = 256)))
}

#=================Display sample images========================================
#display the first thirty samples from training dataset
par(mar = c(rep(0.1, 4)))   #set margin to 0.1
par(mfrow = c(6, 5))   #set plot array to 6 (row) by 5 (col)
for (i in 1:30) {
        display(training[i,])
}


#=================train rf in caret============================================
library(randomForest)
library(foreach)
library(doSNOW)
ctrl <- trainControl(method = 'repeatedcv',
                     number = 2,
                     repeats = 1,
                     verboseIter=TRUE,
                     allowParallel = F)

#use multicore and caret packages for parallel computing
ptm <- proc.time()  #start the clock
cl <- makeCluster(2) #register 2 cores
registerDoSNOW(cl)
set.seed(0)
modelfit.rf <-
        train(label ~ .,
              data = training,
              method = 'rf',
              trControl = ctrl,
              ntree=150)
stopCluster(cl)
proc.time()-ptm   #stop the clock
#predict
pred.rf <- predict(modelfit.rf,testing)
confusionMatrix(pred.rf,testing$label)

#accuracy 0.963. ~4500sec


#=================train rf in randomforest=====================================
#use multicore, foreach and randomforest packages for parallel computing
cl <- makeCluster(2) #register 2 cores
registerDoSNOW(cl)
ptm <- proc.time()  #start the clock
set.seed(234)
rf <- foreach(ntree = rep(50, 3),
                .combine = combine,
                .packages = 'randomForest',
                .verbose=TRUE) %dopar% {
                        randomForest(training[,-1], training[, 1], 
                                     ntree = ntree,
                                     do.trace = TRUE)
                }
stopCluster(cl)
proc.time()-ptm   #stop the clock
#print confusion matrix of predition by rf
confusionMatrix(predict(rf,testing),testing$label)

#training time~570sec,accuracy 96.4
