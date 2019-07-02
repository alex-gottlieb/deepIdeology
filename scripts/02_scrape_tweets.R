library(readr)
library(twitteR)
library(smappR)
library(streamR)
library(jsonlite)

setwd("~/deepIdeology/")

scrape_tweets <- function(users, tweets_per_user, credentials_dir, out_dir) {
  if (!dir.exists(out_dir)) {
    dir.create(out_dir)
  }
  lapply(users$screen_name, 
         function(x) { 
           fname <- file.path(out_dir, paste0(x,'_tweets.json'))
           tryCatch(getTimeline(fname,
                                oauth_folder = credentials_dir, 
                                screen_name = x,
                                n = tweets_per_user), 
                    error = function(e) NA)
           }
        )
}

tweets_to_df <- function(tweet_dir, keep_retweets=FALSE) {
  files <- list.files(tweet_dir)                   
  tweets <- lapply(files, 
                   function(x) {
                     tryCatch(parseTweets(file.path(tweet_dir, x), legacy=TRUE), 
                              error=function(e) NA)
                     }
                   )
  tweets <- do.call("rbind",tweets)
  tweets$tweet_url <- sprintf("https://twitter.com/%s/status/%s", tweets$screen_name, tweets$id_str)
  
  if (!keep_retweets) {
    tweets <- tweets[!grepl("RT", tweets$text),]
  }
}

users <- read_csv("~/Thesis/elites-results.csv")
scrape_tweets(users, 
              tweets_per_user = 200,
              credentials_dir = "credentials",
              out_dir <- "data/elite_tweets")
df <- tweets_to_df("data/elite_tweets")
write_csv(df, "data/elite_tweets.csv")

