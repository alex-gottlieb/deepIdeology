do_install <- FALSE
if (do_install) {
  install.packages(c("readr","streamR","twitteR","devtools", "keras"))
  library(devtools)
  install_version("rmongodb", version = "1.8.0", repos = "http://cran.us.r-project.org")
  install_github("SMAPPNYU/smappR")
  
  install_github("rstudio/keras")
  library(keras)
  install_keras()
}