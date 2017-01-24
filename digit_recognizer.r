#this script is the classifer for digit recognization. 

#configure multicore
library

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
ctrl <- trainControl(method = 'repeatedcv',
                     number = 10,
                     repeats = 1)
modelfit <-
        train(label ~ .,
              data = training[sample(1:42000, 1000), ],
              method = 'rf',
              trControl = ctrl)

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
