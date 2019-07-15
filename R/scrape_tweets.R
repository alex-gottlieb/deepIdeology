#' setup_dependencies
#'
#' This function should be called after package installation to properly set up dependencies.
#' @export
setup_dependencies <- function() {
  library(keras)
  install_keras()

  library(devtools)
  install_version("rmongodb", version = "1.8.0", repos = "http://cran.us.r-project.org")
  install_github("SMAPPNYU/smappR")
}

#' scrape_tweets
#'
#' This function scrapes the most recent n Tweets of a list of Twitter users.
#' @param screen_names Character vector of screen names of Twitter users.
#' @param ids Character or integer vector of IDs of Twitter users. Use either (but not both) of these two arguments.
#' @param tweets_per_user Number of tweets to scrape for each user.
#' @param credentials_dir Directory with Twitter OAuth tokens.
#' @param out_dir Name of directory to store scraped Tweets.
#' @export
#' @examples
#' data("tweets")
#' users <- unique(tweets$screen_name)
#' scrape_tweets(screen_names = users, tweets_per_user = 200, credentials_dir = "credentials", out_dir = "data/scraped_tweets")
scrape_tweets <- function(screen_names = NULL, ids = NULL, tweets_per_user, credentials_dir, out_dir) {
  if (!dir.exists(out_dir)) {
    dir.create(out_dir)
  }

  scrape_func <- function(x) {
    fname <- file.path(out_dir, paste0(x,'_tweets.json'))
    tryCatch(smappR::getTimeline(fname,
                                 oauth_folder = credentials_dir,
                                screen_name = x,
                                n = tweets_per_user),
             error = function(e) NA)
  }

  if (!is.null(screen_names)){
    lapply(screen_names, scrape_func)
  } else {
    lapply(ids, scrape_func)
  }
}

#' tweets_to_df
#'
#' This function takes a directory of JSON files containing scraped Tweets and returns a data.frame
#' @param tweet_dir Directory where scraped Tweets are stored
#' @param keep_retweets Optionally discard retweets.
#' @return data.frame of Tweets with metadata
#' @export
#' @examples
#' tweet_df <- tweets_to_df("data/scraped_tweets", keep_retweets = FALSE)
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

  return(tweets)
}

