library(reshape2) #Required for data transformation
library(memisc) #Required for finding missing values
library(glmnet) #Required for regularization models
library(gbm) #Required for boosting model
library(caret) #Required for train-test partition and pre-processing
library(xgboost) #Required for Extreme Gradient Boosting model
library(randomForest) #Required for Random Forest model
library(pls) #Required for Principal component regression
library(tidyverse) #Required for dplyr and ggplot2
library(dplyr) #Required for data manipulation
library(ggplot2) #Required for data visualization 
library(GGally) #For pairwise plots
library(factoextra) #Required for PCA visualization
library(MASS) 
library(gridExtra)
library(cluster)

#Importing csv file 
df <- read.csv(file = '/Users/robertocolangelo/Desktop/Progetto Brazilian Houses/Data Analysis final project/BrazHousesRent.csv')
df
# Dimensions of the dataset (# of Instances and columns)
dim(df) #10692 instances and 12 attributes

#Glimpse of the typology of each variable.
str(df) #4 categorical variables to substitute, the rest are numerical

#Computing some  statistics
summary(df)

#Null values 
df[df=="-"]<-NA
colSums(is.na(df))
which(colSums(is.na(df))>0)
names(which(colSums(is.na(df))>0))
#Only floor has Null values 

df<-na.omit(df)
#Let's remove them

#Removing duplicates and creating a new dataframe (df) with no duplicates
df<-unique(df)
dim(df) #The number of instances has diminished from 10692 to 7960

#eliminating unary columns (unary means with only one value in a column) (not relevant for our modeling)
for(i in list(seq=1,12,1)){
  if(length(unique(df[,i]))<2)
    df[,i]<-NULL
}
dim(df) #Same number of columns meaning that there are not unary columns


#Data Visualization
#Relationship between rent and taxes on property
dfr<-df
ggplot(aes(y = property.tax..R.., x = rent.amount..R..  ), data = dfr) + 
  geom_point(alpha = 0.35, pch = 1) + 
  labs(y= 'Taxes ',
       x= 'Rent',
       title= 'Relationship of Rent Vs. Taxes')
#There is an outlier --> We'll remove it and replot the relationship between the two variable

which(dfr$property.tax..R..>20000) 
dfr<-dfr[-c(4719,5006),]
#dfr<-dfr[-c(1683,6105,6493),] #Removing instances number 1683,6104,6492
ggplot(aes(y = property.tax..R.., x = rent.amount..R..  ), data = dfr) + 
  geom_point(alpha = 0.35, pch = 1) + 
  labs(y= 'Taxes ',
       x= 'Rent',
       title= 'Relationship of Rent Vs. Taxes')
#Removing high leverage point 
which(df$rent.amount..R..>40000)
dfr<-dfr[-2156,]

ggplot(aes(y = property.tax..R.., x = rent.amount..R..  ), data = dfr) + 
  geom_point(alpha = 0.35, pch = 1) + 
  labs(y= 'Taxes ',
       x= 'Rent',
       title= 'Relationship of Rent Vs. Taxes')
#Most of the rent amounts go from 0 to 15000 while taxes from 0 to 6000

#Barchart of number properties per city 
ggplot(df) + 
  geom_bar(aes( x = city,fill=city, ))+ 
  theme(axis.text.x = element_text(vjust=0.5, angle = 90))+
  labs(y= 'Number of Properties',
      x= 'Density',
      title= 'Number of properties per city')
#Most of the properties are in Sao Paulo
ggplot(df)+
  geom_violin(aes(x=city,y=rent.amount..R..,fill=city))+
  theme(axis.text.x = element_text(vjust=0.5, angle = 90))
#As can be seen from the violin distribution plot of rents the place which costs the most is Sao Paulo (The shape of the violin is less wide and longer) while it appears that the cheapest rents are in Campinas

#Boxplot of rent vs furniture
ggplot(df) +
  geom_boxplot( aes(x=furniture, y=rent.amount..R.., fill=furniture))+
  theme(axis.text.x = element_text(vjust=0.5, angle = 60))
#On average places with no furniture cost less but there are some not furnished place that cost even more top furnished places


#Graph of Space vs number of bathrooms
dfg<-df
which(df$area>10000)
dfg<-dfg[-c(2365,5786,8958),]
dfg$bathroom<-as_factor(dfg$bathroom)
ggplot(aes(y = area, x = bathroom,fill=bathroom), data = dfg) + 
  geom_boxplot() + 
  labs(y= 'area ',
       x= 'Number of bathrooms',
       title= 'House Space Vs. Number of Bathrooms')
which(dfg$area>1500)
dfg<-dfg[-c(1789,4476,6569,6906),]
dfg$bathroom<-as_factor(dfg$bathroom)
ggplot(aes(y = area, x = bathroom,fill=bathroom), data = dfg) + 
  geom_boxplot() + 
  labs(y= 'area ',
       x= 'Number of bathrooms',
       title= 'House Space Vs. Number of Bathrooms')

#Distribution of target variable
dfy<-df$rent.amount
distrib<-hist(dfy, breaks=20 , col="yellow", xlab="Rent",
              main="Distribution of Rent")
xfit<-seq(min(dfy),max(dfy),length=40)
yfit<-dnorm(xfit,mean=mean(dfy),sd=sd(dfy))
yfit <- yfit*diff(distrib$mids[1:2])*length(dfy)
lines(xfit, yfit, col="black ", lwd=2)

#Not close to a Gaussian distribution.Might consider to transform it with logarithm (To fulfill assumptions linear regression)

dfylog<-log(dfy)
dfylog
distrib<-hist(dfylog, breaks=20 , col="yellow", xlab="Rent",
              main="Distribution of Rent")
xfit<-seq(min(dfylog),max(dfylog),length=40)
yfit<-dnorm(xfit,mean=mean(dfylog),sd=sd(dfylog))
yfit <- yfit*diff(distrib$mids[1:2])*length(dfylog)
lines(xfit, yfit, col="black ", lwd=2)
#Gaussian Distribution
#We need to transform the target variable rent amount with log

#Cleaning and transforming categorical variables
str(df)
#Converting animal allowance and furniture as binary variables
df$animal<-ifelse(df$animal=='acept',1,0)
df$furniture<-ifelse(df$furniture=='furnished',1,0)

#might consider converting Cities with one Hot encoding (there is no hierarchical order...) and removing/converting floor to numeric because too many variable would be created with onehotencoding
df$floor<-as.numeric(df$floor)
dummy <- dummyVars(" ~ .", data=df)
df <- data.frame(predict(dummy, newdata = df))
dforiginal<-df

#Correlation matrix between all variables 
df[,]%>%
  cor()%>%
  melt()%>%
  ggplot(aes(Var1,Var2, fill=value))+
  scale_fill_gradient2(low = "#075AFF",
                       mid = "#FFFFCC",
                       high = "#FF0000")+
  geom_tile(color='black')+
  geom_text(aes(label=paste(round(value,2))),size=2, color='black')+
  theme(axis.text.x = element_text(vjust=0.5, angle = 65))+
  labs(title='Correlation between variables',
       x='',y='')
#Most important correlations with target variable rent amount are:
#rooms->Number of rooms
#fire.insurance--> monthly fire insurance cost, kind strange it is =1 
#parking.spaces --> private parking spaces included with the house

#Correlation without one hot encoded variables 
dfnocity<-df[,-c(1,2,3,4,5)]
dfnocity
dfnocity[,]%>%
  cor()%>%
  melt()%>%
  ggplot(aes(Var1,Var2, fill=value))+
  scale_fill_gradient2(low = "#075AFF",
                       mid = "#FFFFCC",
                       high = "#FF0000")+
  geom_tile(color='black')+
  geom_text(aes(label=paste(round(value,2))),size=2, color='black')+
  theme(axis.text.x = element_text(vjust=0.5, angle = 65))+
  labs(title='Correlation between variables',
       x='',y='')

#Pair plots
#pairs(df)
#if you want to visualize these pair plots you have to enlarge the bottom right corner panel

#TASK 1
#REGRESSION MODELS --> PREDICTING TARGET 'rent amount' 
# Train - test split
#Setting seed 
set.seed(12)
df$rent.amount..R..<-log(df$rent.amount..R..) #Trasforming output variable with log for linear regression assumptions
training <- df$rent.amount..R.. %>%
  createDataPartition(p = 0.75, list = FALSE)
trainset  <- df[training, ]
xtrain<-trainset[,-14]
ytrain<-trainset[,14]
testset<- df[-training, ]
xtest<-testset[,-14]
ytest<-testset[,14]
#Scaling variables
xtrain <- xtrain %>% scale()
xtest <- xtest %>% scale(center=attr(xtrain, "scaled:center"), 
                         scale=attr(xtrain, "scaled:scale"))

#BASELINE MODEL
base <- lm(rent.amount..R.. ~ rooms, data = trainset)
summary(base) #Rsquared== 0.266--> low
#Interpretation of the coefficients--> since we transformed the target with a log operation  the impact of an increase of 1 in X is of e^(beta1)
#if beta1= 0.38 impact of unit increase of X on Y--> e^(0.38)=1.462 , if we add 1 room the rent increases by 1.46*number of rooms
predictions<-predict(base,testset[,-14])
#We'll use Mean squared error as metric
baselineMSE=mean((ytest-predictions)^2) #MSE 0.42
baselineMSE

#model with most correlated variables with Y
base2 <- lm(rent.amount..R..~fire.insurance..R..+ rooms, data = trainset)
summary(base2)#full model has adjusted Rsquared of 0.833--> improved a lot, fire insurance is super powerful in the predictive model
predictions<-predict(base2,testset[,-14])
baseline2MSE=mean((ytest-predictions)^2)#MSE 0.09 -> way better.
baseline2MSE
#Adding the variable fire.insurance has a huge impact on prediction performance 

#full model (ALL INPUT VARIABLES)
fulllr<- lm(rent.amount..R..~., data = trainset)
summary(fulllr)#full model has adjusted Rsquared of 0.88-->really high on trainset 
predictions<-predict(fulllr,testset[,-14])
multivariatelrMSE=mean((predictions-ytest)^2)  #MSE 0.07
multivariatelrMSE

#Performing Akaike information criterion stepwise selection
#Forward and back
AICforback<- step(fulllr, direction = "both", trace = F)
summary(AICforback)
AICforbackMSE=mean(AICforback$residuals^2) #MSE 0.068 -> improved (removes the following variables: animal, property tax,area and hoa)
AICforbackMSE
#Back
modelstepaicback<- step( fulllr, direction = "backward", trace = F)
summary(modelstepaicback)
AICbackMSE=mean(modelstepaicback$residuals^2)
AICbackMSE #MSE 0.068
#Forward
modelstepaicforward<- step( fulllr, direction = "forward", trace = F)
summary(modelstepaicforward)
AICforMSE=mean(modelstepaicforward$residuals^2)
AICforMSE#MSE 0.068  
#It's the same with back,forward and both

#By looking at the summary animal  and property.tax don't seem to be statistically significant 
trainnew<-trainset[,c(-11,-15)]
testnew<-testset[,c(-11,-15)]

#Models with interaction terms of a subset of variables
interaction<-lm(rent.amount..R..~ rooms +I(rooms*area) +furniture+hoa..R..+floor+parking.spaces+bathroom+cityBelo.Horizonte+cityCampinas+cityPorto.Alegre+cityRio.de.Janeiro+citySão.Paulo,data=trainnew)
summary(interaction) #Adjusted Rsquared 0.55 interaction between area and rooms impact negatively on the prediction so let's not consider interactions 
lrinteractionMSE=mean(interaction$residuals^2) #MSE 0.259
lrinteractionMSE
#PCA 
#Visualizing number of components needed 
pca <- princomp(trainset[,-14], 
                cor = TRUE,
                score = TRUE)
pca$loadings
summary(pca)
fviz_eig(pca) #By using elbow method it is clear that 3 components are more than enough
#Principal component regression 
pcreg <- pcr(rent.amount..R..~., data = trainset, scale = TRUE, validation = "CV")
pcrpred <- predict(pcreg, testset[,-14], ncomp = 3)
pcrMSE=mean((pcrpred - testset[,14])^2) #MSE 0.18-> obviously we lose a bit of variability from creating new variables from the original ones but now we only have 3 dimensions (variables)
pcrMSE
#Regularization models 
folds <- 10
control <- trainControl(method="cv", number=folds)

#Ridge 10 folds
nlambdas <- 100
lambdas <- seq(0.001, 2, length.out = nlambdas)
trainmatrix<-as.matrix(trainset)
testmatrix<-as.matrix(testset)
x<-trainmatrix[,-14]
y<-trainmatrix[,14]
ridgereg10 <- cv.glmnet(x, y,nfolds=folds, alpha=0)
ridgereg10<- train(x, y, method = "glmnet", trControl = control,
                   tuneGrid = expand.grid(alpha = 0, 
                                          lambda = lambdas))
predictions<-predict(ridgereg10,as.matrix(testmatrix[,-14]))
ridgereg10MSE=mean((predictions - testmatrix[,14])^2 )#MSE 0.076 pretty good but not as good as linear regression
ridgereg10MSE
#Ridge 20 folds
folds<-20
nlambdas <- 100
lambdas <- seq(0.001, 2, length.out = nlambdas)
trainmatrix<-as.matrix(trainset)
testmatrix<-as.matrix(testset)
x<-trainmatrix[,-14]
y<-trainmatrix[,14]
ridgereg20<- cv.glmnet(x, y,nfolds=folds, alpha=0)
ridgereg20<- train(x, y, method = "glmnet", trControl = control,
                   tuneGrid = expand.grid(alpha = 0, 
                                          lambda = lambdas))
predictions<-predict(ridgereg20,as.matrix(testmatrix[,-14]))
ridgereg20MSE=mean((predictions - testmatrix[,14])^2 )#MSE 0.076 increasing the number of folds doesn't impact
ridgereg20MSE

#Lasso 10 folds
fold=10
lambdas <- 10^seq(2, -3, by = -.1)
nalphas <- 20
alphas <- seq(0, 1, length.out=nalphas)
ctrl <- trainControl(method="cv", number=folds)
lassoreg <- train(x, y, method = "glmnet", trControl = control,
                  tuneGrid = expand.grid(alpha = alphas, 
                                         lambda = lambdas))
lassoreg$bestTune$alpha#Best parameters 
lassoreg$bestTune$lambda
predictions<-predict(lassoreg,as.matrix(testmatrix [,-14]))
lassoregMSE=mean((predictions - testmatrix[,14])^2) #MSE 0.072 BETTER THAN RIDGE--> shrinking some coefficients directly to 0 gives bettere results
lassoregMSE
#Elastic net -> Takes a while to run
lambda.grid<-10^seq(2,-2,length=100)
alpha.grid<-seq(0,1,length=100) #Trying out 100 different values of alpha between 0 and 1 to understand which is the best alpha
grid<-expand.grid(alpha=alpha.grid,lambda=lambda.grid)
Elnet<-train(x, y, method = "glmnet", trControl = ctrl,tuneGrid = grid)
Elnet$bestTune #Best parameters
predictions<-predict(Elnet,as.matrix(testmatrix [,-14]))
ElasticnetMSE=mean((predictions - testmatrix[,14])^2)
ElasticnetMSE#MSE 0.072
#Alpha close to 0? More ridge than lasso
#Alpha close to 1? More Lasso than ridge

#Random Forest 
Rfreg <- randomForest(
  formula = rent.amount..R..~ .,
  data= trainset,
  importance=FALSE,
  ntree=600 #Setting number of trees to train Random Forest model.The higher the number the more the time to execute the code
)
predictions<-predict(Rfreg,testset[-14])
RfregMSE=mean((predictions - testset[,14])^2) #MSE 0.008 almost perfect on the testset
RfregMSE
# Gradient Boosting
boostreg= gbm(rent.amount..R.. ~.,
              data = trainset,
              distribution='gaussian',
              cv.folds = 10,
              shrinkage = .1,
              n.minobsinnode = 10,
              n.trees = 500 )
summary(boostreg)
predictions<-predict(boostreg,testset[,-14])
boostregMSE=mean((predictions - testset[,14])^2) #0.01 Random forest performed better but still super good
boostregMSE
#Xgboost 
xgboost = xgboost(data = trainmatrix[,-14], label = trainmatrix[,14], nrounds = 150, 
                  objective = "reg:squarederror", eval_metric = "error")

err_xg_tr = xgboost$evaluation_log$train_error
predxgboost<-predict(xgboost,testmatrix[,-14])
XGBMSE=mean((predxgboost - testset[,14])^2) #0.009, a little bit better than gradient boosting but still not as good as random forest
XGBMSE

#Let's compare performances of the different models

models<-c('baseline','baseline 2','Multivariate LR',"Step_forwardAIC", "Step_backwardAIC", "MixedAIC",'Interaction',"PCR", "Lasso",'Elastic Net',"Ridge 10","Ridge 20", "RandomForest", "Boosting",'XGBoost')
r=c(baselineMSE,baseline2MSE,multivariatelrMSE,AICforMSE, AICbackMSE, AICforbackMSE,lrinteractionMSE, pcrMSE, lassoregMSE,ElasticnetMSE, ridgereg10MSE,ridgereg20MSE, RfregMSE, boostregMSE,XGBMSE)
Performancesummary <- data.frame(Model = models, MSE = r)
Performancesummary[order(Performancesummary$MSE),][1:5,]#Prints 5 most performing models

Performanceplot<- ggplot(data = Performancesummary) + 
  geom_point(mapping = aes(x = Model, y=MSE,colour='Red'),show.legend=FALSE) +
  labs(x = 'Models', y = 'MSE')
Performanceplot+theme(axis.text.x=element_text(angle=90))
#The lower the y coordinate of the point, the Lower the MSE, the Better the Model performance

#TASK 2)
# Kmeans Clustering 
set.seed(13)
k2 <- kmeans(dforiginal,centers=2,nstart=10)
k2WSS<-k2$tot.withinss
k3 <- kmeans(dforiginal, centers = 3, nstart = 10)
k3WSS<-k3$tot.withinss
k4 <- kmeans(dforiginal, centers = 4, nstart = 10)
k4WSS<-k4$tot.withinss
k5 <- kmeans(dforiginal, centers = 5, nstart = 10)
k5WSS<-k5$tot.withinss
WSS1<-c(k2WSS,k3WSS,k4WSS,k5WSS)
Nclust<-c(2,3,4,5)
Clustercomparisonwss<-data.frame(WCSS=WSS1,K=Nclust)
Clustercomparisonwss[order(Clustercomparisonwss$WCSS),]
#Not really a great variation of Within Cluster Sum of Squares

# Let's plot the clusters graphically
p1 <- fviz_cluster(k2, geom = "point", data = dforiginal) + ggtitle("k = 2")
p2 <- fviz_cluster(k3, geom = "point",  data = dforiginal) + ggtitle("k = 3")
p3 <- fviz_cluster(k4, geom = "point",  data = dforiginal) + ggtitle("k = 4")
p4 <- fviz_cluster(k5, geom = "point",  data = dforiginal) + ggtitle("k = 5")
grid.arrange(p1, p2, p3, p4, nrow = 2)
#Really messy clusters that interpolate with one another

#Using all the variables doesn't giev good interpretable results, let's try with just two variables (rooms and rent amount)
dfcluster<-dforiginal[,c(7,14)]
km2 <- kmeans(dfcluster,centers=2,nstart=25)
km2WSS<-km2$tot.withinss
km3<- kmeans(dfcluster, centers = 3, nstart = 25)
km3WSS<-km3$tot.withinss
km4<- kmeans(dfcluster, centers = 4, nstart = 25)
km4WSS<-km4$tot.withinss
km5<- kmeans(dfcluster, centers = 5, nstart = 25)
km5WSS<-km5$tot.withinss
WSS2<-c(km2WSS,km3WSS,km4WSS,km5WSS)
Nclust<-c(2,3,4,5)
Clustercomparisonwss<-data.frame(WCSS=WSS2,K=Nclust)
Clustercomparisonwss[order(Clustercomparisonwss$WCSS),]
#Obviously the more clusters,the less Within cluster sum of squares

# Let's see the representation to compare
pm1 <- fviz_cluster(km2, geom = "point", data = dfcluster) + ggtitle("k = 2")
pm2 <- fviz_cluster(km3, geom = "point",  data = dfcluster) + ggtitle("k = 3")
pm3 <- fviz_cluster(km4, geom = "point",  data = dfcluster) + ggtitle("k = 4")
pm4 <- fviz_cluster(km5, geom = "point",  data = dfcluster) + ggtitle("k = 5")
grid.arrange(pm1, pm2, pm3, pm4, nrow = 2)
#We can see that the clusters are mostly affected by the rent amount

#Hierarchical clustering 
d1 <- dist(dforiginal, method = "euclidean")
hc1 <- hclust(d1, method = "complete" ) #Complete linkage
fviz_nbclust(dforiginal, FUN = hcut, method = "wss") #Takes a while to run and shows the best number of clusters is 2 according to elbow method
sub_grp1 <- cutree(hc1, k = 2)
plot(sub_grp1, cex = 0.6, hang = -1)
fviz_cluster(list(data = dforiginal, cluster = sub_grp1),geom = "point")
#The distinction between the different clusters is not so clear so let's try again with only a couple of variables

#Hierarchical clustering with only 2 variables
d2 <- dist(dfcluster, method = "euclidean")
hc2 <- hclust(d2, method = "complete" ) #Complete linkage
fviz_nbclust(dfcluster, FUN = hcut, method = "wss") #Takes a while to run and shows the best number of clusters is 3 according to elbow method
sub_grp2 <- cutree(hc2, k = 3)
plot(sub_grp2, cex = 0.6, hang = -1)
fviz_cluster(list(data = dfcluster, cluster = sub_grp2),geom = "point")

