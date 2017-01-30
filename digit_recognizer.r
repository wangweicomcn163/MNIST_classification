#this script is the classifer for digit recognization. 
library(caret)
#=================load training and testing data sets==========================
rawdata <- read.csv('Digit recognizer/train.csv')
#testing <- read.csv('Digit recognizer/test.csv')


#=================Display sample images========================================
#display the first thirty samples from training dataset
par(mar = c(rep(0.1, 4)))   #set margin to 0.1
par(mfrow = c(6, 5))   #set plot array to 6 (row) by 5 (col)
for (i in 1:30) {
        spl <- rawdata[i, 2:785]
        spl <- unlist(spl)
        dim(spl) <- c(28, 28)
        image(spl[, nrow(spl):1], axes = FALSE,
              col = grey(seq(0, 1, length = 256)))
}

#=================preprocess data==============================================
rawdata$label <- as.factor(rawdata$label)
zerovaridx <- nearZeroVar(rawdata,saveMetrics = TRUE)
rawdata_proc <- rawdata[,which(!zerovaridx$zeroVar)] #remove zerovar variables

#=================split raw data===============================================
intrain <- createDataPartition(rawdata_proc$label,p=0.75,list = FALSE)
training <- rawdata_proc[intrain,]
testing <- rawdata_proc[-intrain,]



#=================train rf in caret============================================
library(randomForest)
library(foreach)
library(doSNOW)
ctrl <- trainControl(method = 'repeatedcv',
                     number = 2,
                     repeats = 1,
                     verboseIter=TRUE,
                     allowParallel = TRUE)

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
rf <- foreach(ntree = rep(50, 3),
                .combine = combine,
                .packages = 'randomForest') %dopar% {
                        randomForest(training[,-1], training[, 1], 
                                     ntree = ntree)
                }
stopCluster(cl)
proc.time()-ptm   #stop the clock
#print confusion matrix of predition by rf
confusionMatrix(predict(rf,testing),testing$label)

#training time~570sec,accuracy 96.4




#=================Display sample images========================================
#display the first thirty samples from testing dataset
par(mar = c(rep(0.1, 4)))   #set margin to 0.1
par(mfrow = c(6, 5))   #set plot array to 6 (row) by 5 (col)
for (i in 1:30) {
        spl <- testing[i,]
        spl <- unlist(spl)
        dim(spl) <- c(28, 28)
        image(spl[, nrow(spl):1], axes = FALSE,
              col = grey(seq(0, 1, length = 256)))
}

dev.off() #reset the device to default.