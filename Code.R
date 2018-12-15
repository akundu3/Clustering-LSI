setwd("C:/Users/Aditya/Documents/Data/Reuters/training")
library (tm)
library (SnowballC)
library(cluster)
library(fpc)
require(graphics)

# Reading the data

acq <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/acq"), readerControl = list (reader = readPlain))
bop <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/bop"), readerControl = list (reader = readPlain))
carcass <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/carcass"), readerControl = list (reader = readPlain))
cocoa <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/cocoa"), readerControl = list (reader = readPlain))
coffee <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/coffee"), readerControl = list (reader = readPlain))
corn <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/corn"), readerControl = list (reader = readPlain))
cpi <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/cpi"), readerControl = list (reader = readPlain))
crude <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/crude"), readerControl = list (reader = readPlain))
dlr <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/dlr"), readerControl = list (reader = readPlain))
earn <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/earn"), readerControl = list (reader = readPlain))
gnp <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/gnp"), readerControl = list (reader = readPlain))
gold <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/gold"), readerControl = list (reader = readPlain))
grain <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/grain"), readerControl = list (reader = readPlain))
interest <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/interest"), readerControl = list (reader = readPlain))
livestock <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/livestock"), readerControl = list (reader = readPlain))
money_fx <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/money_fx"), readerControl = list (reader = readPlain))
money_supply <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/money_supply"), readerControl = list (reader = readPlain))
nat_gas <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/nat_gas"), readerControl = list (reader = readPlain))
oilseed <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/oilseed"), readerControl = list (reader = readPlain))
reserves <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/reserves"), readerControl = list (reader = readPlain))
ship <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/ship"), readerControl = list (reader = readPlain))
soybean <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/soybean"), readerControl = list (reader = readPlain))
sugar <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/sugar"), readerControl = list (reader = readPlain))
trade <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/trade"), readerControl = list (reader = readPlain))
veg_oil <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/veg_oil"), readerControl = list (reader = readPlain))
wheat <- Corpus(DirSource("/Users/Aditya/Documents/Data/Reuters/training/wheat"), readerControl = list (reader = readPlain))
alldata<- c( crude , grain , trade )
## Creating two preprocessing functions to analyze number of unique terms,
## before and after removing stopwords

# Preprocessing Function 1 (does not remove stop words)
prp <- function(x){
  alldata <- tm_map(alldata, removeNumbers)
  alldata <- tm_map(alldata, removePunctuation)
  alldata <- tm_map(alldata, content_transformer(tolower))
  alldata <- tm_map(alldata, stemDocument)
  alldata <- tm_map(alldata, stripWhitespace)
  return(x)
}

## Preprocessing function to remove stopwords
prpd<- function(x){
  x <- prp(x)
  dtm <- DocumentTermMatrix(alldata)
  y <- weightTfIdf(y)
  print(ncol(y))
  alldata <- tm_map(alldata, removeWords, stopwords("english"))
  y <- DocumentTermMatrix(x)
  dtmidf <- weightTfIdf(dtm)
  print(ncol(y))
  return(y)
}

acqdtm <- prpd(acq)
bopdtm <- prpd(bop)
carcassdtm <- prpd(carcass)
cocoadtm <- prpd(cocoa)
coffeedtm <- prpd(coffee)
corndtm <- prpd(corn)
cpidtm <- prpd(cpi)
crudedtm <- prpd(crude)
dlrdtm <- prpd(dlr)
earndtm <- prpd(earn)
gnpdtm <- prpd(gnp)
golddtm <- prpd(gold)
graindtm <- prpd(grain)
interestdtm <- prpd(interest)
livestockdtm <- prpd(livestock)
money_fxdtm <- prpd(money_fx)
money_supplydtm <- prpd(money_supply)
nat_gasdtm <- prpd(nat_gas)
oilseeddtm <- prpd(oilseed)
reservesdtm <- prpd(reserves)
shipdtm <- prpd(ship)
soybeandtm <- prpd(soybean)
sugardtm <- prpd(sugar)
tradedtm <- prpd(trade)
veg_oildtm <- prpd(veg_oil)
wheatdtm <- prpd(wheat)
alldatadtm <- prpd(alldata)

trade <- tm_map(trade, removeWords, stopwords("english"))
grain <- tm_map(grain, removeWords, stopwords("english"))
crude <- tm_map(crude, removeWords, stopwords("english"))
alldata <- tm_map(alldata, removeWords, stopwords("english"))
acq <- tm_map(acq, removeWords, stopwords("english"))
wheat <- tm_map(wheat, removeWords, stopwords("english"))

trade <- prp(trade)
grain <- prp(grain)
crude <- prp(crude)
alldata <- prp(alldata)
acq <- prp(acq)
wheat <- prp(wheat)

tradedtmc <- DocumentTermMatrix(trade)
graindtmc <- DocumentTermMatrix(grain)
crudedtmc  <- DocumentTermMatrix(crude)
acqdtmc <- DocumentTermMatrix(acq)
wheatdtmc <- DocumentTermMatrix(wheat)
alldatadtmc <- DocumentTermMatrix(alldata)

## taking corpuses with similar number of documents
data1 <- c(tradedtmc,graindtmc,crudedtmc)

## taking corpuses with a large differnce in the number of documents
data2 <- c(acqdtmc,wheatdtmc)
data3 <- c(crude,grain,trade,acq,wheat)

## Reducing high sparsity of the docment term matrices
data1 <- removeSparseTerms(data1, 0.99)
data2 <- removeSparseTerms(data2, 0.99)
dtmidf <- removeSparseTerms(dtmidf, 0.99)
data3 <- DocumentTermMatrix(data3)
data3 <- removeSparseTerms(data3,0.99)
pdf <- matrix(0, nrow = 5, ncol=1122)
rownames(pdf) <- c("crude", "grain", "trade", "acq", "wheat")
colnames(pdf) <- colnames(data3)
for(i in 1:1122){
  pdf[5,i] = sum(data3[2842:3053,i])
}
pdf1 <- pdf/1122
plot(pdf1[1,])
data1 <- weightTfIdf(data1)
data2 <- weightTfIdf(data2)
alldatadtmc <- weightTfIdf(alldatadtmc)

cl1 <- kmeans(data2, 2)
cl1
cl1$tot.withinss
plotcluster(data1, cl1$cluster)
points(cl1$centers, col = 1:2, pch = 8, cex = 2)

svd <- svd(alldatadtmc)
row.names(svd$u) <- row.names(alldatadtmc)
row.names(svd$v) <- colnames(alldatadtmc)
d <- diag(svd$d)

# Taking rows from the U matrix of SVD on all data for the corresponding topic
# subset of u for   crude(389 docs)        grain(433 docs)       trade(369 docs)
udata1 <- rbind(   svd$u[2191:2580,],      svd$u[5783:6216,],    svd$u[7970:8338,])
# subset of u for   acq(1650 docs)   wheat(212 docs)
udata2 <- rbind(   svd$u[1:1650,],   svd$u[8425:8637,])
udata1.scaled <- scale(udata1)
udata2.scaled <- scale(udata2)
scale(m, center=FALSE, scale=colSums(m))

#clustering after svd
test <- udata1.scaled[,1:50] %*% d[1:50,1:50] %*% t(svd$v[,1:50])
cl1 <- kmeans(test, 3)
plotcluster(test, cl1$cluster)
plotcluster(test, cl1$centers)
View(cl1$cluster)
View(cl1$centers)

dv <- d[1:20,1:20] %*% t(svd$v[,1:20])

wssplot <- function(data, nc, seed=1234){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")}
wssplot(test, 20)
wssplot(data2,20)


topics <- c("acq", "bop", "carcass", "cocoa","coffee","corn","cpi","crude","dlr","earn","gnp","gold","grain",
            "interest","livestock","money_fx","money_supply","nat_gas","oilseed","reserves",
            "ship","soybean","sugar","trade","veg_oil","wheat")





dtm <- DocumentTermMatrix(reuters)
dtm <- removeSparseTerms(dtm, 0.99)
dtm_tfxidf <- weightTfIdf(dtm)
svd <- svd(dtm_tfxidf,20,20)
row.names(svd$u) <- row.names(dtm)
row.names(svd$v) <- colnames(dtm)
d <- diag(svd$d)
d <- d[which (d!=0)][1:20]
d <- diag(d)
test <- svd$u %*% d %*% t(svd$v)

#clustering

a <- svd$v[,5]
final <-sort(abs(a), decreasing = TRUE)[1:100]
prop<-data.frame(final)
wordcloud(rownames(prop), prop[,1], scale=c(5,0.5), colors=brewer.pal(8, "Dark2"))
plot(final)
