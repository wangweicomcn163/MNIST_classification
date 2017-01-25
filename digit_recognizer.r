#this script is the classifer for digit recognization. 

#load training and testing data sets
training <- read.csv('Digit recognizer/train.csv')
testing <- read.csv('Digit recognizer/test.csv')

#convert labels to factors
training$label <- as.factor(training$label)

#display the first thirty samples from training dataset
par(mar = c(rep(0.1, 4)))   #set margin to 0.1
par(mfrow = c(6, 5))   #set plot array to 6 (row) by 5 (col)
for (i in 1:30) {
        sample <- training[i, 2:785]
        sample <- unlist(sample)
        dim(sample) <- c(28, 28)
        image(sample[, nrow(sample):1], axes = FALSE,
              col = grey(seq(0, 1, length = 256)))
}

#train rf model
library(caret)
library(randomForest)
library(foreach)
library(doSNOW)
ctrl <- trainControl(method = 'repeatedcv',
                     number = 2,
                     repeats = 1)
set.seed(0)
rows <- sample(1:42000, 10000)

ptm <- proc.time()  #start the clock
modelfit <-
        train(label ~ .,
              data = training[rows,],
              method = 'rf',
              trControl = ctrl,
              ntree=100)
proc.time()-ptm   #stop the clock



cl <- makeCluster(2) #register 2 cores
registerDoSNOW(cl)
ptm <- proc.time()  #start the clock
rf <- foreach(ntree = rep(100, 4),
                .combine = combine,
                .packages = 'randomForest') %dopar% {
                        randomForest(training[rows,-1], training[rows, 1], 
                                     ntree = ntree)
                }
proc.time()-ptm   #stop the clock
stopCluster(cl)
#print confusion matrix of predition by rf
confusionMatrix(predict(rf,training),training$label)


#predict
pred <- predict(modelfit,testing)

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
