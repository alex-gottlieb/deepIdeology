#' predict_ideology
#'
#' This function allows you to scale the ideological slant of Twitter posts.
#' @param tweets Character vector of Tweets.
#' @param model Neural network architecture to use. Options are "LSTM", "BiLSTM", or "C-BiLSTM".
#' @param embeddings Type of word embedding algorithm to use. Options are "w2v" (word2vec), "glove", or "random" (random initialization).
#' @param embedding_dim Length of word embeddings to use. Options are 25, 50, 100, or 200.
#' @param filter_political_tweets If Tweet collection may contain non-political Tweets, optionally filter them out before ideological scaling.
#' @return Vector of float values between 0 and 1, where values closer to 0 indiciate liberal ideological slant, values closer to 1 indicate conservative ideological slant, and values near 0.5 indicate a lack of ideological leaning. Non-political Tweets return return a NULL value.
#' @export
#' @details The data set on which the models are trained is roughly 75 percent Tweets from "elite" users (e.g. politicians, media outlets, think tanks, etc.), with the remaining 25 percent coming from "mass" users. In validating the models, it became apparent that they were much more capable of identifying slant from the former group, which in many ways presents an idealized scenario of clearly- (and often forcefully-) articulated ideological leanings along with (mostly) consistent grammar and spelling. Predictions of "mass" Tweets were largely clustered around the middle of the spectrum, not because they were necessarily more moderate, but because the models could not make a confident prediction either way. Accordingly, researchers should use caution when using this package to scale Tweets from groups other than poltiical elites.
#' @details The Tweets used to train the models were scraped and labeled in early 2018. The ideological spectrum is, of course, not a static entity, and where particular issues and actors fall on that spectrum can shift over time. Additionally, new issues and actors have emerged on the political scene since this data was collected, so stances on more recent topics (e.g. Brett Kavanaugh or the Green New Deal) that might provide a great deal of information to a political observer about someone's leanings would not provide any additional information to the model.
#' @examples
#' tweets <- c("Make no mistake- the President of the United States is actively sabotaging the health insurance of millions of Americans with this action.",
#'             "This MLK Day, 50 years after his death, we honor Dr. King's legacy. He lived for the causes of justice and equality, and opened the door of opportunity for millions of Americans. America is a better; freer nation because of it.",
#'             "I’m disappointed in Senate Democrats for shutting down the government. #SchumerShutdown")
#' preds <- predict_ideology(tweets, model="BiLSTM", embeddings="w2v")

predict_ideology <- function(tweets, model="BiLSTM", embeddings="w2v", embedding_dim=25, filter_political_tweets=FALSE) {
  stopifnot(model %in% list("LSTM", "BiLSTM", "C-BiLSTM"))

  cwd <- getwd()
  setwd("~/.deepIdeology/")
  # if Tweet collection contains non-political tweets, filter out before scaling ideology
  if (filter_political_tweets) {
    if (!file.exists("models/politics_classifier.h5")) {
      print("No pre-trained politics classifier exists. Training model now. This may take a moment.")
      prepare_politics_classifier()
    }

    model <- keras::load_model_hdf5("models/politics_classifier.h5")
    pol_ind <- model %>%
      keras::predict_classes(texts)
    sprintf("%i political Tweets identified out of %i total Tweets", table(preds)[2], length(preds))
  } else {
    pol_ind <- 1:length(tweets)
  }

  # load fit tokenizer, convert raw text to sequences
  if (!file.exists("tokenizers/ideo_tweet_tokenizer")) {
    data("ideo_tweets")
    tokenizer <- keras::text_tokenizer(num_words = 20000)
    tokenizer <- keras::fit_text_tokenizer(tokenizer, ideo_tweets$text)
    if (!dir.exists("tokenizers")) {
      dir.create("tokenizers")
    }
    keras::save_text_tokenizer(tokenizer, "tokenizers/ideo_tweet_tokenizer")
  }

  tokenizer <- keras::load_text_tokenizer("tokenizers/ideo_tweet_tokenizer")

  # load desired model
  model_name_map <- list(LSTM = "lstm", BiLSTM = "bi-lstm", CBiLSTM = "c-bi-lstm")
  model_fname <- sprintf("models/%s_%s_%sd.h5", model_name_map[[model]], embeddings, embedding_dim)

  if (!file.exists(model_fname)) {
    print("No pre-trained model with that configuration exists. Training model now. This may take a moment.")
    data("ideo_tweets")
    text_vecs <- texts_to_vectors(ideo_tweets$text, tokenizer)
    labels <- ideo_tweets$ideo_cat
    data <- train_test_split(text_vecs, labels)
    if (model == "BiLSTM") {
      bidirectional = TRUE
      convolutional = FALSE
    } else if (model == "C-BiLSTM") {
      bidirectional = TRUE
      convolutional = TRUE
    } else {
      bidirectional = FALSE
      convolutional = FALSE
    }
    train_lstm(data$X_train, data$y_train, embeddings = embeddings, embedding_dim = embedding_dim,
               bidirectional = bidirectional, convolutional = convolutional)
  }

  model <- keras::load_model_hdf5(model_fname)

  text_vecs <- texts_to_vectors(tweets, tokenizer)
  # generate predictions on new text
  preds <- model %>%
    keras::predict_proba(text_vecs)

  preds[-pol_ind] <- NULL

  setwd(cwd)
  return(preds[,1])
}

