library(tm)
library(pdftools)

files <- list.files(pattern = "pdf$")

arts = lapply(files, pdf_text)

directory <- getwd()
txt_corpus = Corpus(DirSource(directory, pattern = ".pdf$"), readerControl = list(reader = readPDF))

#cat(txt[1])

txt_corpus = tm_map(txt_corpus, content_transformer(tolower))

txt_corpus = tm_map(txt_corpus, removePunctuation)
txt_corpus = tm_map(txt_corpus, stripWhitespace)


txt_corpus = tm_map(txt_corpus, removeWords, stopwords("english"))
txt_corpus = tm_map(txt_corpus, removeNumbers)
txt_corpus = tm_map(txt_corpus, stemDocument)

txt_corpus = TermDocumentMatrix(txt_corpus)


inspect(txt_corpus)

ft <- findFreqTerms(txt_corpus, lowfreq = 50, highfreq = Inf)
View(as.matrix(txt_corpus[ft,]))
ft.dm <- as.matrix(txt_corpus[ft,])
View(sort(apply(ft.dm, 1, sum), decreasing = TRUE))


#dtm = DocumentTermMatrix(txt_corpus)
#dtm = as.matrix(dtm)
#dtm = t(dtm)