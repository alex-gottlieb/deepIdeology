library(keras)
setwd("~/deepIdeology/")

predict_ideology <- function(tweets, model="LSTM", embedding_dim=25, use_glove_embeddings=FALSE, filter_political_tweets=FALSE) {
  stopifnot(model %in% list("LSTM", "BiLSTM", "C-BiLSTM"))
  
  # if Tweet collection contains non-political tweets, filter out before scaling ideology
  if (filter_political_tweets) {
    tokenizer <- load_text_tokenizer("tokenizers/pol_tweet_tokenizer")
    sequences <- texts_to_sequences(tokenizer, tweets)
    texts <- pad_sequences(sequences)
    
    model <- load_model_hdf5("models/politics_classifier.h5")
    preds <- model %>%
      predict_classes(texts)
    sprintf("%i political Tweets identified out of %i total Tweets", table(preds)[2], length(preds))
    tweets <- tweets[preds]
  }
  
  # load fit tokenizer, convert raw text to sequences
  tokenizer <- load_text_tokenizer("tokenizers/ideo_tweet_tokenizer")
  sequences <- texts_to_sequences(tokenizer, tweets)
  texts <- pad_sequences(sequences)
  
  # load desired model
  model_name_map <- list(LSTM = "lstm", BiLSTM = "bi-lstm", CBiLSTM = "c-bi-lstm")
  model_fname <- sprintf("models/%s_%sd.h5", model_name_map[[model]], embedding_dim)
  if (use_glove_embeddings) {
    model_fname <- sub("_", "_glove_", model_fname)
  }
  model <- load_model_hdf5(model_fname)
  
  # generate predictions on new text
  preds <- model %>%
    predict_proba(texts)
  
  return(preds[,1])
}

# tweets <- read_csv("data/elite_ideo_tweets.csv")
# ideo <- predict_ideology(tweets$text)
