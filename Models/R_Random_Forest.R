library(randomForest)
library(rsample)
library(r2pmml)
library(caret)
library(xtable)
library(Metrics)
require(caTools)
setwd("~/Documents/Datalogi/Master Thesis/Ferry_Route_Optimization")

dataset <-read.csv("Data/Relative/relative_data.csv")

set.seed(1)

train <- sample(1:nrow(dataset), nrow(dataset)/2)

test <- dataset[-train, "fuel_consumption"]


r_mse = c(0.02955530, 0.02848476, 0.02853054, 0.02886243, 0.02888279, 0.02911002, 0.02933143, 0.02948338, 0.02835805, 0.02860505, 0.02877669, 0.02893323, 0.02917435, 0.02946092, 0.02951731, 0.02843712, 0.02858015, 0.02881386, 0.02894738, 0.02916407, 0.02941284, 0.02947304, 0.02836104, 0.02858483, 0.02876070, 0.02897612, 0.02917569, 0.02937070)
r_mae = c(0.10652669, 0.10076950, 0.09942246, 0.09942309, 0.09889954, 0.09863660, 0.09838028, 0.10646274, 0.10043085, 0.09976513, 0.09901807, 0.09868776, 0.09872663, 0.09851562, 0.10641472, 0.10066445, 0.09956209, 0.09918118, 0.09879197, 0.09863105, 0.09859110, 0.10634910, 0.10045547, 0.09955502, 0.09898029, 0.09880544, 0.09862411, 0.09853669)


a_mse = c(0.0450, 0.0310, 0.0311, 0.0311, 0.0313, 0.0317, 0.0323, 0.0459, 0.0310, 0.0309, 0.0312, 0.0314, 0.0318, 0.0323, 0.0461, 0.0310, 0.0309, 0.0311, 0.0314, 0.0318, 0.0322, 0.0462, 0.0309, 0.0308, 0.0311, 0.0315, 0.0318, 0.0323)
a_mae = c(0.1466, 0.1027, 0.1003, 0.0994, 0.0993, 0.0997, 0.1006, 0.1490, 0.1026, 0.1002, 0.0997, 0.0994, 0.0997, 0.1007, 0.1496, 0.1025, 0.1002, 0.0994, 0.0994, 0.0997, 0.1004, 0.1496, 0.1023, 0.0998, 0.0995, 0.0994, 0.0996, 0.1005)


dat <- data.frame(
  model_number = c(1:length(r_mse)),
  mse_diff = a_mse - r_mse,
  mae_diff = a_mae - r_mae
)


ggplot() + geom_bar(data=dat, aes(x=model_number, y=mse_diff), stat='identity', fill='blue') 

ggplot() + geom_bar(data=dat, aes(x=model_number, y=mae_diff), stat='identity', fill='blue') 





modelList <- list()
MSEList <- list()
MAEList <- list()
predList <- list()

ntreeList <- list()
mtryList <- list()


ntrees <- c(2000)
mtrys <- c(7)
i <- 1

for (ntree in ntrees){
  
  for (mtry in mtrys){
    
    
    rf <- randomForest(fuel_consumption~., data=dataset, subset=train, importance=TRUE, keep.forest=TRUE, ntree=ntree, mtry=mtry)
    key <- toString(ntree+mtry)
    modelList[[key]] <- rf

    pred <- predict(rf, newdata=dataset[-train,])
    # plot(pred, test, xlim=c(0.5,3), ylim=c(0.5,3), cex=0.5)
    # abline(0,1)
    predList[[key]] <- pred

    MSE <- mean((pred-test)^2)
    MSEList[[key]] <- MSE

    MAE <- mae(test, pred)
    MAEList[[key]] <- MAE


    ntreeList[[key]] = ntree
    mtryList[[key]] = mtry
    
    
    print(i)
    i <- i+1
    
  }
  
  
  
  
}

ntreeVector <- unlist(ntreeList, use.names = FALSE)
mtryVector <- unlist(mtryList, use.names = FALSE)

MSEVector <- unlist(MSEList, use.names = FALSE)

MAEVector <- unlist(MAEList, use.names = FALSE)


plot(ntrees, MSEList, ylab="MSE")
lines(ntrees[order(ntrees)], MSEVector[order(ntrees)])


plot(ntrees, MAEList, ylab="MAE")
lines(ntrees[order(ntrees)], MAEVector[order(ntrees)])

temp <- c("500", "1000", "1500", "2000")

RESData <- data.frame(
  ntree = rep(temp, each=1),
  mtry = mtryVector,
  mse = MSEVector,
  mae = MAEVector
)

RESData2 <- data.frame(
  ntree = rep(temp, each=7),
  mtry = c(1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7),
  mse = c(0.02955530, 0.02848476, 0.02853054, 0.02886243, 0.02888279, 0.02911002, 0.02933143, 0.02948338, 0.02835805, 0.02860505, 0.02877669, 0.02893323, 0.02917435, 0.02946092, 0.02951731, 0.02843712, 0.02858015, 0.02881386, 0.02894738, 0.02916407, 0.02941284, 0.02947304, 0.02836104, 0.02858483, 0.02876070, 0.02897612, 0.02917569, 0.02937070),
  mae = c(0.10652669, 0.10076950, 0.09942246, 0.09942309, 0.09889954, 0.09863660, 0.09838028, 0.10646274, 0.10043085, 0.09976513, 0.09901807, 0.09868776, 0.09872663, 0.09851562, 0.10641472, 0.10066445, 0.09956209, 0.09918118, 0.09879197, 0.09863105, 0.09859110, 0.10634910, 0.10045547, 0.09955502, 0.09898029, 0.09880544, 0.09862411, 0.09853669)
)



p1 <- ggplot(data = RESData2, aes(x=mtry, y=mse)) + geom_line(aes(colour=ntree))
p1 + theme(legend.position = "top")


p2 <- ggplot(data = RESData2, aes(x=mtry, y=mae)) + geom_line(aes(colour=ntree))
p2 + theme(legend.position = "top")

xtable(RESData)


hl <- c(1, 2, 3, 4, 5, 6, 7, 8, 9)
hl_mse <- c(0.0487, 0.0471, 0.0519, 0.0476, 0.0534, 0.0521, 0.0556, 0.0476, 0.0497)
hl_mae <- c(0.1407, 0.1300, 0.1349, 0.1219, 0.1291, 0.1252, 0.1330, 0.1219, 0.1172)

plot(hl, hl_mse, xlab="Number of Hidden Layers", ylab="MSE")
lines(hl[order(hl)], hl_mse[order(hl)])


plot(hl, hl_mae, xlab="Number of Hidden Layers", ylab="MAE")
lines(hl[order(hl)], hl_mae[order(hl)])


# pred <- predict(rf, newdata=dataset[-train,])
# plot(pred, test, xlim=c(0.5,3), ylim=c(0.5,3), cex=0.5)
# abline(0,1)
# MSE <- mean((pred-test)^2)
# MSE
# 
# MAE <- mae(test, pred)
# MAE
# 

# #validation split
# valid_split <- initial_split(dataset, 0.8)
# 
# #training data
# train <- analysis(valid_split)
# 
# #validation data
# valid <- assessment(valid_split)
# 
# x_test <- valid[setdiff(names(valid), "fuel_consumption")]
# y_test <- valid$fuel_consumption
# 
# #trainIndeces <- sample(nrow(dataset), 0.7 *nrow(dataset), replace = FALSE)
# #train <- dataset[trainIndeces,]
# #test <- dataset[-trainIndeces,]
# 
# #split <- sample.split(dataset$fuel_consumption, SplitRatio = 0.7)
# #train <- subset(dataset, split=TRUE)
# #test <- subset(dataset, split=FALSE)
# 
# rf <- randomForest(fuel_consumption ~ ., data=train, xtest=x_test, ytest=y_test, keep.forest=TRUE)
# rf
# 
# oob <- sqrt(rf$mse)
# oob
# validation <- sqrt(rf$test$mse)
# validation
# plot(rf)
# which.min(rf$mse)
# sqrt(rf$mse[which.min(rf$mse)])
# 
# #predTrain <- predict(rf, train)
# #predTrain
# 
# #MAE <- function(actual, predicted) {mean(abs(predicted - actual)) }
# #MAE(test$fuel_consumption, p1)
# 
# #MSE <- mean((p1-test$fuel_consumption)^2)
# #MSE
# 
# #make predictions
# predictions <- predict(rf, x_test)
# 
# plot(y_test, predictions)
# abline(coef = c(0,1))
# 
# MAE <- mae(y_test, predictions)
# MAE
# 
# importance(rf)


##TODO##
# test with different values of mtry
