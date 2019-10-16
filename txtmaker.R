files <- list.files(pattern = "pdf$")
arts = lapply(files, pdf_text)
typeof(arts)
lapply(arts, length)

head(arts[1])

for (name in files) {
  tname = gsub(".pdf", ".txt", name) 
  print(tname)
  for (item in arts) {
    sink(file = tname)
    print(item)
    sink()
  }
}
