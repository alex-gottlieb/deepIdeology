prepare_politics_classifier <- function() {
  data("pol_tweets")

  if (file.exists("tokenizers/pol_tweet_tokenizer")) {
    tokenizer <- keras::load_text_tokenizer("tokenizers/pol_tweet_tokenizer")
  } else {
    tokenizer <- keras::text_tokenizer()
    tokenizer <- keras::fit_text_tokenizer(tokenizer,pol_tweets$Input.text)
    if (!dir.exists("tokenizers")) {
      dir.create("tokenizers")
    }
    keras::save_text_tokenizer(tokenizer, "tokenizers/pol_tweet_tokenizer")
  }

  texts <- texts_to_vectors(pol_tweets, tokenizer)
  labels <- pol_tweets$pol
  word_index <- tokenizer$word_index
  data <- train_test_split(texts, labels)

  lstm <- keras::keras_model_sequential()
  lstm %>%
    keras::layer_embedding(input_dim = length(word_index)+1, output_dim=64) %>%
    keras::layer_lstm(units=64, dropout=0.5, recurrent_dropout=0.3) %>%
    keras::layer_dense(units=16, activation='relu') %>%
    keras::layer_dropout(0.5) %>%
    keras::layer_dense(units=1, activation='sigmoid')

  lstm %>% keras::compile(loss='binary_crossentropy',optimizer='adam',metrics=c('accuracy'))

  if (!dir.exists("models")) {
    dir.create("models")
  }
  lstm %>% keras::fit(
    X_train, y_train,
    batch_size=64,
    epochs=2,
    validation_split=0.2,
    callbacks = list(keras::allback_model_checkpoint(sprintf("models/politics_classifier.h5"),
                                                     monitor = "val_loss",
                                                     save_best_only = TRUE),
                     keras::callback_early_stopping(monitor = "val_loss", patience=3))
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

  model <- keras::keras_model_sequential()
  if (embeddings != "random") {
    embedding_fname <- sprintf("embeddings/tweet_%s_%sd.rda", embeddings, embedding_dim)

    if (!file.exists(embedding_fname)) {
      sprintf("Embedding file does not exist. Preparing %s-dimensional %s embeddings. This may take a moment", embedding_dim, embeddings)
      tokenizer <- keras::load_text_tokenizer("tokenizers/ideo_tweet_tokenizer")
      if (embeddings == "glove") {
        prepare_glove_embeddings(embedding_dim, tokenizer)
      } else {
        data("ideo_tweets")
        prepare_w2v_embeddings(ideo_tweets$text, embedding_dim, tokenizer)
      }
    }
    emebedding_matrix <- get(load(embedding_fname))

    model %>%
      keras::layer_embedding(input_dim = dim(embedding_matrix)[1], output_dim=embedding_dim,
                             weights = list(embedding_matrix))
    out_fname <- sprintf("lstm_%s_%sd.h5", embeddings, embedding_dim)
  } else {
    model %>%
      keras::layer_embedding(input_dim = 20000+1, output_dim=64)
    out_fname <- sprintf("lstm_%sd.h5", embedding_dim)
  }
  if (convolutional) {
    model %>%
      keras::layer_conv_1d(filters=64,
                           kernel_size = 3,
                           padding = 'valid',
                           activation = 'relu',
                           strides=1) %>%
      keras::layer_max_pooling_1d(pool_size = 2)
    out_fname <- sprintf("c-bi-%s", out_fname)
  }
  if (bidirectional) {
    model %>%
      keras::bidirectional(layer_lstm(units=64, dropout=0.3, recurrent_dropout=0.3))
    if (!convolutional) out_fname <- sprintf("bi-%s", out_fname)
  } else {
    model %>%
      keras::layer_lstm(units=64, dropout=0.3, recurrent_dropout=0.3)
  }

  model %>%
    keras::layer_dense(units=16, activation='relu') %>%
    keras::layer_dropout(0.5) %>%
    keras::layer_dense(units=1, activation='sigmoid')

  model %>% keras::compile(loss='binary_crossentropy',optimizer='adam',metrics=c('accuracy'))

  if (!dir.exists("models")) {
    dir.create("models")
  }
  model %>% keras::fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    callbacks = list(keras::callback_model_checkpoint(sprintf("models/%s", out_fname),
                                                      monitor = "val_loss",
                                                      save_best_only = TRUE),
                     keras::callback_early_stopping(monitor = "val_loss", patience=3))
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
  model <- keras::load_model_hdf5(model_path)
  preds <- model %>%
    keras::predict_classes(X_test)

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
  sequences <- keras::texts_to_sequences(tokenizer, texts)
  vecs <- keras::pad_sequences(sequences)
  return(vecs)
}

