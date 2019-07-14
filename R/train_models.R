library(keras)
library(readr)
library(text2vec)
library(glmnet)
library(reticulate)
library(purrr)

train_baseline <- function(X_train, y_train) {
  prep_fun <- tolower
  tok_fun <- word_tokenizer
  it_train <- itoken(X_train,
                     preprocessor = prep_fun,
                     tokenizer = tok_fun,
                     ids = y_train,
                     progressbar = FALSE)

  vocab <- create_vocabulary(it_train)
  vectorizer <- vocab_vectorizer(vocab)
  dtm_train <- create_dtm(it_train, vectorizer)
  tfidf <- TfIdf$new()
  dtm_train_tfidf <- fit_transform(dtm_train,tfidf)

  clf <- cv.glmnet(x=dtm_train_tfidf, y=y_train,
                   family="binomial")

  save(clf, file="models/logit.Rdata")
}

prepare_politics_classifier <- function() {
  data("pol_tweets")

  if (file.exists("tokenizers/pol_tweet_tokenizer")) {
    tokenizer <- load_text_tokenizer("tokenizers/pol_tweet_tokenizer")
  } else {
    tokenizer <- text_tokenizer()
    tokenizer <- fit_text_tokenizer(tokenizer,pol_tweets$Input.text)
    if (!dir.exists("tokenizers")) {
      dir.create("tokenizers")
    }
    save_text_tokenizer(tokenizer, "tokenizers/pol_tweet_tokenizer")
  }

  texts <- texts_to_vectors(pol_tweets, tokenizer)
  labels <- pol_tweets$pol
  word_index <- tokenizer$word_index
  data <- train_test_split(texts, labels)

  lstm <- keras_model_sequential()
  lstm %>%
    layer_embedding(input_dim = length(word_index)+1, output_dim=64) %>%
    layer_lstm(units=64, dropout=0.5, recurrent_dropout=0.3) %>%
    layer_dense(units=16, activation='relu') %>%
    layer_dropout(0.5) %>%
    layer_dense(units=1, activation='sigmoid')

  lstm %>% compile(loss='binary_crossentropy',optimizer='adam',metrics=c('accuracy'))

  if (!dir.exists("models")) {
    dir.create("models")
  }
  lstm %>% keras::fit(
    X_train, y_train,
    batch_size=64,
    epochs=2,
    validation_split=0.2,
    callbacks = list(callback_model_checkpoint(sprintf("models/politics_classifier.h5"),
                                               monitor = "val_loss",
                                               save_best_only = TRUE),
                     callback_early_stopping(monitor = "val_loss", patience=3))
  )
}

#' train_lstm
#'
#' This function trains the LSTM model to identify the ideological slant of Tweets.
#' @param X_train data.frame or matrix of vectorized Tweets
#' @param y_train Labels for training data. 0 for liberal, 1 for conservative.
#' @param embeddings Type of word embedding algorithm to use. Options are "w2v" (word2vec), "glove", or "random" (random initialization).
#' @param embedding_dim Length of word embeddings to use. Options are 25, 50, 100, or 200.
#' @param bidirectional Optionally train on text sequences in reverse as well as forwards.
#' @param convolutional Optionally apply convolutional filter to text sequences. Can only be used when bidirectional = TRUE
#' @export
#' @note Models are automatically saved in HDF5 format to a sub-folder of the root-directory called "models". File format is "\{model type\}_\{embedding type\}_\{embedding dimensionality\}d.h5".
#' @examples
#' # train a Bi-LSTM network using GloVe embeddings
#' data("ideo_tweets")
#' ideo_tokenizer <- text_tokenizer(num_words=20000)
#' ideo_tokenizer <- fit_text_tokenizer(ideo_tokenizer, ideo_tweets$text)
#' texts <- texts_to_vectors(ideo_tweets$text, ideo_tokenizer)
#' labels <- tweets$ideo_cat
#'
#' train_test <- train_test_split(texts, labels)
#' X_train <- train_test$X_train
#' y_trian <- train_test$y_train
#' train_ltsm(X_train, ty_train, embeddings="glove", bidirectional=TRUE)
train_lstm <- function(X_train, y_train, embeddings = "w2v", embedding_dim = 25, bidirectional = FALSE, convolutional = FALSE) {
  stopifnot(embedding_dim %in% list(25, 50, 100, 200))
  stopifnot(embeddings %in% list("random", "w2v", "glove"))

  out_fname <- sprintf("lstm_%sd.h5", embedding_dim)

  model <- keras_model_sequential()
  if (embeddings != "random") {
    embedding_fname <- sprintf("embeddings/tweet_%s_%sd.rda", embeddings, embedding_dim)

    if (!file.exists(embedding_fname)) {
      sprintf("Embedding file does not exist. Preparing %s-dimensional %s embeddings. This may take a moment", embedding_dim, embeddings)
      tokenizer <- load_text_tokenizer("tokenizers/ideo_tweet_tokenizer")
      if (embeddings == "glove") {
        prepare_glove_embeddings(embedding_dim, tokenizer)
      } else {
        data("ideo_tweets")
        prepare_w2v_embeddings(ideo_tweets$text, embedding_dim, tokenizer)
      }
    }
    emebedding_matrix <- get(load(embedding_fname))

    model %>%
      layer_embedding(input_dim = dim(embedding_matrix)[1], output_dim=embedding_dim,
                      weights = list(embedding_matrix))
    out_fname <- sprintf("lstm_%s_%sd.h5", embeddings, embedding_dim)
  } else {
    model %>%
      layer_embedding(input_dim = 20000+1, output_dim=64)
    out_fname <- sprintf("lstm_%sd.h5", embedding_dim)
  }
  if (convolutional) {
    model %>%
      layer_conv_1d(filters=64,
                    kernel_size = 3,
                    padding = 'valid',
                    activation = 'relu',
                    strides=1) %>%
      layer_max_pooling_1d(pool_size = 2)
    out_fname <- sprintf("c-bi-%s", out_fname)
  }
  if (bidirectional) {
    model %>%
      bidirectional(layer_lstm(units=64, dropout=0.3, recurrent_dropout=0.3))
    if (!convolutional) out_fname <- sprintf("bi-%s", out_fname)
  } else {
    model %>%
      layer_lstm(units=64, dropout=0.3, recurrent_dropout=0.3)
  }

  model %>%
    layer_dense(units=16, activation='relu') %>%
    layer_dropout(0.5) %>%
    layer_dense(units=1, activation='sigmoid')

  model %>% compile(loss='binary_crossentropy',optimizer='adam',metrics=c('accuracy'))

  if (!dir.exists("models")) {
    dir.create("models")
  }
  model %>% keras::fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    callbacks = list(callback_model_checkpoint(sprintf("models/%s", out_fname),
                                               monitor = "val_loss",
                                               save_best_only = TRUE),
                     callback_early_stopping(monitor = "val_loss", patience=3))
  )
}

#' evaluate
#'
#' This function evaluates the performance of a trained model.
#' @param model_path Path to HDF5 file containing model. Should be of the form "models/\{model type\}_\{embedding type\}_\{embedding dimensionality\}d.h5"
#' @param X_test data.frame or matrix of vectorized Tweets
#' @param y_test Labels for testing data. 0 for liberal, 1 for conservative.
#' @export
#' @return List of performance metrics. Currently, a confusion matrix, overall prediction accuracy, precision, recall, and F1 score are return.
#' @examples
#' data("ideo_tweets")
#' ideo_tokenizer <- text_tokenizer(num_words=20000)
#' ideo_tokenizer <- fit_text_tokenizer(ideo_tokenizer, ideo_tweets$text)
#' texts <- texts_to_vectors(ideo_tweets$text, ideo_tokenizer)
#' labels <- tweets$ideo_cat
#'
#' train_test <- train_test_split(texts, labels)
#'
#' evaluate("models/bi-lstm_w2v_25d.h5", train_test$X_test, train_test$y_test)
evaluate <- function(model_path, X_test, y_test) {
  model <- load_model_hdf5(model_path)
  preds <- model %>%
    predict_classes(X_test)

  res <- list()
  cm <-as.matrix(table(Actual = y_test, Predicted = preds))
  res[["Confusion Matrix"]] <- cm

  n <- sum(cm) # number of instances
  nc <- nrow(cm) # number of classes
  diag <- diag(cm) # number of correctly classified instances per class
  rowsums <- apply(cm, 1, sum) # number of instances per class
  colsums <- apply(cm, 2, sum) # number of predictions per class
  p <- rowsums / n # distribution of instances over the actual classes
  q <- colsums / n # distribution of instances over the predicted classes
  accuracy <- sum(diag) / n
  res[["Accuracy"]] <- accuracy

  precision <- diag / colsums
  recall <- diag / rowsums
  f1 <- 2 * precision * recall / (precision + recall)
  res[["Precision/Recall"]] <- data.frame(precision, recall, f1)

  return(res)
}

#' train_test_split
#'
#' Helper function to split data into training and testing sets.
#' @param X data.frame or matrix of data
#' @param y Labels (optional).
#' @param test_size Proportion of samples to set aside for testing.
#' @export
#' @return List of X_train, X_test, y_train, y_test
train_test_split <- function(X, y, test_size=0.2) {
  n_train <- floor((1-test_size)*nrow(X))
  train_ind <- sample(nrow(X),n_train)
  return(list(X_train=X[train_ind,], X_test=X[-train_ind,], y_train=y[train_ind], y_test=y[-train_ind]))
}

#' texts_to_vectors
#'
#' Helper function vectorize text data
#' @param texts Character vector of raw text data
#' @param tokenizer Pre-fit keras tokenizer
#' @export
#' @return matrix of vectorized texts
texts_to_vectors <- function(texts, tokenizer){
  sequences <- texts_to_sequences(tokenizer, texts)
  vecs <- pad_sequences(sequences)
  return(vecs)
}

# train_all_models <- function(X_train, y_train, tokenizer) {
#   for (embedding in c("random", "w2v", "glove")) {
#     for (bidirectional in c(TRUE, FALSE)) {
#       for (convolutional in c(TRUE, FALSE)) {
#         if (convolutional & !bidirectional) {
#           next
#         } else {
#           cat(sprintf("Training model with parameters:\n%s embeddings\r\nbidirectional=%s\r\nconvolutional=%s",
#                   embedding, bidirectional, convolutional))
#           train_lstm(X_train,
#                      y_train,
#                      embedding,
#                      embedding_dim = 25,
#                      tokenizer = tokenizer,
#                      bidirectional = bidirectional,
#                      convolutional = convolutional)
#         }
#       }
#     }
#   }
# }


