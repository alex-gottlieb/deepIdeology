library(keras)
library(readr)
library(text2vec)
library(glmnet)
library(reticulate)
library(purrr)

setwd("~/deepIdeology/")
source("scripts/word_embeddings.R")

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

train_lstm <- function(X_train, y_train, embeddings="w2v", embedding_dim=25, tokenizer = NULL, bidirectional=FALSE, convolutional=FALSE) {
  stopifnot(embedding_dim %in% list(25, 50, 100, 200))
  stopifnot(embeddings %in% list("random", "w2v", "glove"))
  stopifnot(embeddings != "random" & is.null(tokenizer))

  out_fname <- sprintf("lstm_%sd.h5", embedding_dim)
  
  model <- keras_model_sequential()
  if (embeddings != "random") {
    out_file <- sprintf( "data/embeddings/tweet_%s_%sd.Rdata", embeddings, embedding_dim)
    embedding_matrix <- get(load(out_file))

    model %>% 
      layer_embedding(input_dim = dim(embedding_matrix)[1], output_dim=embedding_dim,
                      weights = list(embedding_matrix))
    out_fname <- sprintf("lstm_glove_%sd.h5", embedding_dim)
  } else {
    model %>%
      layer_embedding(input_dim = length(tokenizer$num_words)+1, output_dim=64)
    out_fname <- sprintf("lstm_w2v_%sd.h5", embedding_dim)
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

train_test_split <- function(n_obs,test_size=0.2) {
  n_train <- floor((1-test_size)*n_obs)
  train_ind <- sample(n_obs,n_train)
  train_ind
}

texts_to_vectors <- function(texts, tokenizer){
  sequences <- texts_to_sequences(tokenizer, texts)
  vecs <- pad_sequences(sequences)
  return(vecs)
}

train_all_models <- function(X_train, y_train, tokenizer) {
  for (embedding in c("random", "w2v", "glove")) {
    for (bidirectional in c(TRUE, FALSE)) {
      for (convolutional in c(TRUE, FALSE)) {
        if (convolutional & !bidirectional) {
          next
        } else {
          cat(sprintf("Training model with parameters:\n%s embeddings\r\nbidirectional=%s\r\nconvolutional=%s", 
                  embedding, bidirectional, convolutional))
          train_lstm(X_train, 
                     y_train, 
                     embedding,
                     embedding_dim = 25, 
                     tokenizer = tokenizer, 
                     bidirectional = bidirectional,
                     convolutional = convolutional)
        }
      }
    }
  }
}

tweets <- read_csv("data/ideology_classifier_data.csv")
ideo_tokenizer <- load_text_tokenizer("tokenizers/ideo_tweet_tokenizer")
texts <- texts_to_vectors(tweets$text, ideo_tokenizer)
labels <- tweets$ideo_cat

train_ind <- train_test_split(nrow(tweets))

# X_train_raw <- tweets$text[train_ind]
# X_test_raw <- tweets$text[-train_ind]
X_train <- texts[train_ind,]
X_test <- texts[-train_ind,]
y_train <- labels[train_ind]
y_test <- labels[-train_ind]

if (!dir.exists("models")) {
  dir.create("models")
}

train_all_models(X_train, y_train, ideo_tokenizer)

