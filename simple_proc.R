library(textreuse)
library(cluster)
library(factoextra)
library(NbClust)
library(NMF)
library(vegan)
library(clusteval)
library(clstutils)

asinTransform <- function(p) { asin(sqrt(p)) }
logitTransform <- function(p) { log(p/(1-p)) }

Metadata <- read.table("file_list.txt", stringsAsFactors = F, header = F)
rownames(Metadata) <- Metadata$V3
colnames(Metadata) <- c("hit","clus","file")

all_files.words <- TextReuseCorpus(dir = "All_files", meta = list(rownames(Metadata)),
                           tokenizer = tokenize_ngrams, skip_short = TRUE)


minhash <- minhash_generator(10000)
all_files.minhash <- TextReuseCorpus(dir = "All_files", tokenizer = tokenize_words,
                          minhash_func = minhash, keep_tokens = TRUE,
                          progress = FALSE)

all_files.words.dist <- pairwise_compare(all_files.words, ratio_of_matches, directional=T)
all_files.minhash.dist <- pairwise_compare(all_files.minhash, ratio_of_matches, directional=T)


all_files.words.dist <- dist(all_files.words.dist)
all_files.words.dist.asin <- asinTransform(all_files.words.dist)
all_files.words.dist.asin[is.na(all_files.words.dist.asin)] <- 0


#all_files.words.dist.logit <- logitTransform(all_files.words.dist)
#all_files.words.dist.logit[is.na(all_files.words.dist.logit)] <- 0

all_files.minhash.dist <- dist(all_files.minhash.dist)
all_files.minhash.dist.asin <- asinTransform(all_files.minhash.dist)
all_files.minhash.dist.asin[is.na(all_files.minhash.dist.asin)] <- 0

#all_files.minhash.dist.logit <- logitTransform(all_files.minhash.dist)
#all_files.minhash.dist.logit[is.na(all_files.minhash.dist.logit)] <- 0

adonis2(all_files.words.dist.asin ~ hit, data = Metadata, permutations=9999)

fviz_nbclust(as.matrix(all_files.words.dist.asin), kmeans,  k.max = 15, method = "wss")
fviz_nbclust(as.matrix(all_files.words.dist.asin), kmeans,  k.max = 15, method = "silhouette")

gap_stat <- clusGap(as.matrix(all_files.words.dist.asin), FUN = kmeans, nstart = 25,
                    K.max = 15, B = 10)
print(gap_stat, method = "firstmax")
fviz_gap_stat(gap_stat)

comp.pam <- pam(daisy(as.matrix(all_files.words.dist.asin)), 2, diss = TRUE)

clusplot(comp.pam, main = "ecology: Cluster plot, k = 2", 
         color = TRUE)

library("tm")
library("NMF")

mycorpus <- PCorpus(DirSource("All_files", encoding = "UTF-8"), dbControl=list(dbName="file_db"))

revs <- tm_map(mycorpus, content_transformer(tolower)) 
revs <- tm_map(revs, removeWords, stopwords("english")) 
revs <- tm_map(revs, removePunctuation) 
#revs <- tm_map(revs, removeNumbers) 
#revs <- tm_map(revs, stripWhitespace) 

dtm <- DocumentTermMatrix(revs)
dtm_m <- removeSparseTerms(dtm, 0.91)

m_dtm <- as.matrix(dtm_m)
m_dtm[is.na(m_dtm)] <- 0
m_dtm[is.null(m_dtm)] <- 0
m_dtm <- m_dtm[ rowSums(m_dtm)!=0, ] 

res3 <- nmf(m_dtm,2)
res2 <- nmf(m_dtm,3)
res4 <- nmf(as.matrix(all_files.words.dist.asin),2)

res <- res4

dims <- list( features=rownames(m_dtm), samples=colnames(m_dtm), basis=paste('', 1:nbasis(res), sep='') )

dimnames(res) <- dims
res.basis <- basis(res)
res.coef <- coef(res)

res.basis.temp <- cbind(res.basis,apply(res.basis, 1, min))
rownames(res.basis.temp) <- sub(".txt", "", rownames(res.basis.temp)) 
#colnames(res.basis.temp) <- c(1,2,3)
res.assign <- colnames(res.basis.temp)[apply(res.basis.temp,1,which.min)]

res.assign <- data.frame(res.assign)
rownames(res.assign) <- rownames(res.basis.temp)

# move to quantiles. 
res.quant <- quantile( res.coef, p = 0.75 )

res.report <- apply(res.assign, 1, function(x){ colnames(res.coef)[res.coef[x,] > res.quant] })

cluster_similarity(res.assign$res.assign,Metadata$clus,similarity = "rand")

