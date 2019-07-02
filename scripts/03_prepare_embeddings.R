library(keras)
library(readr)
library(miceadds)

setwd("~/deepIdeology/")

prepare_embedding_matrix <- function(embeddings_file, embedding_dim, word_index, out_file) {
  if (file.exists(out_file)) {
    obj <- load(out_file)
    return(get(obj))
  } else {
    embeddings_index <- new.env(parent = emptyenv())
    lines <- readLines(embeddings_file)
    for (line in lines) {
      values <- strsplit(line, ' ', fixed = TRUE)[[1]]
      word <- values[[1]]
      coefs <- as.numeric(values[-1])
      embeddings_index[[word]] <- coefs
    }
    
    embedding_matrix <- matrix(0L, nrow = length(word_index)+1, ncol = embedding_dim)
    for (word in names(word_index)) {
      index <- word_index[[word]]
      if (index >= length(word_index))
        next
      embedding_vector <- embeddings_index[[word]]
      if (!is.null(embedding_vector)) {
        # words not found in embedding index will be all-zeros.
        embedding_matrix[index,] <- embedding_vector
      }
    }
    
    save(embedding_matrix,file=out_file)
    return(embedding_matrix)
  }
}


tweets <- read_csv("data/elite_ideo_tweets.csv")
ideo_tokenizer <- text_tokenizer(num_words=20000)
ideo_tokenizer <- fit_text_tokenizer(ideo_tokenizer, tweets$text)
sequences <- texts_to_sequences(ideo_tokenizer, tweets$text)
texts <- pad_sequences(sequences)
labels <- tweets$ideo_cat
word_index <- ideo_tokenizer$word_index

train_test_split <- function(n_obs,test_size=0.2) {
  n_train <- floor((1-test_size)*n_obs)
  train_ind <- sample(n_obs,n_train)
  train_ind
}

train_ind <- train_test_split(nrow(tweets))

X_train <- texts[train_ind,]
X_test <- texts[-train_ind,]
y_train <- labels[train_ind]
y_test <- labels[-train_ind]

train_lstm <- function(X_train, y_train, embedding_dim=25, bidirectional=FALSE, convolutional=FALSE, use_glove_embeddings=FALSE) {
  model <- keras_model_sequential()
  if (use_glove_embeddings) {
    out_file <- sprintf( "data/embeddings/tweet_glove_%sd.Rdata", embedding_dim)
    glove_file <- sprintf("data/glove.twitter.27B/glove.twitter.27B.%sd.txt", embedding_dim)
    embedding_matrix <- prepare_embedding_matrix(embeddings_file = glove_file,
                                                 embedding_dim = embedding_dim,
                                                 word_index = word_index,
                                                 out_file = out_file)
    model %>% 
      layer_embedding(input_dim = dim(embedding_matrix)[1], output_dim=embedding_dim,
                      weights = list(embedding_matrix))
  } else {
      model %>%
        layer_embedding(input_dim = length(word_index)+1, output_dim=64)
  }
  
  model %>% 
    layer_lstm(units=64, dropout=0.3, recurrent_dropout=0.3) %>%
    layer_dense(units=16, activation='relu') %>%
    layer_dropout(0.5) %>%
    layer_dense(units=1, activation='sigmoid')

  model %>% compile(loss='binary_crossentropy',optimizer='adam',metrics=c('accuracy'))
  model %>% keras::fit(
    X_train, y_train,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    callbacks = list(callback_model_checkpoint("weights_best.h5", monitor = "val_loss", save_best_only = TRUE),
                     callback_early_stopping(monitor = "val_loss", patience=2))
  )
}


