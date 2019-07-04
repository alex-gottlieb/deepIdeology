library(keras)
library(dplyr)

setwd("~/deepIdeology/")

prepare_glove_embeddings <- function(embeddings_file, embedding_dim, tokenizer, out_file) {
    embeddings_index <- new.env(parent = emptyenv())
    lines <- readLines(embeddings_file)
    for (line in lines) {
      values <- strsplit(line, ' ', fixed = TRUE)[[1]]
      word <- values[[1]]
      coefs <- as.numeric(values[-1])
      embeddings_index[[word]] <- coefs
    }
    
    embedding_matrix <- matrix(0L, nrow = length(tokenizer$word_index)+1, ncol = embedding_dim)
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
    
    save(embedding_matrix, file=out_file)
}

prepare_w2v_embeddings <- function(texts, embedding_dim, tokenizer, out_file) {
  
  skipgrams_generator <- function(text, tokenizer, window_size, negative_samples) {
    gen <- texts_to_sequences_generator(tokenizer, sample(text))
    function() {
      skip <- generator_next(gen) %>%
        skipgrams(
          vocabulary_size = tokenizer$num_words, 
          window_size = window_size, 
          negative_samples = 1
        )
      x <- transpose(skip$couples) %>% map(. %>% unlist %>% as.matrix(ncol = 1))
      y <- skip$labels %>% as.matrix(ncol = 1)
      list(x, y)
    }
  }
  
  skip_window <- 5       # How many words to consider left and right.
  num_sampled <- 1       # Number of negative examples to sample for each word.
  
  input_target <- layer_input(shape = 1)
  input_context <- layer_input(shape = 1)
  
  embedding <- layer_embedding(
    input_dim = tokenizer$num_words + 1, 
    output_dim = embedding_dim, 
    input_length = 1, 
    name = "embedding"
  )
  
  target_vector <- input_target %>% 
    embedding() %>% 
    layer_flatten()
  
  context_vector <- input_context %>%
    embedding() %>%
    layer_flatten()
  
  dot_product <- layer_dot(list(target_vector, context_vector), axes = 1)
  output <- layer_dense(dot_product, units = 1, activation = "sigmoid")
  
  model <- keras_model(list(input_target, input_context), output)
  model %>% compile(loss = "binary_crossentropy", optimizer = "adam")
  summary(model)
  
  tweets <- read_csv("data/ideology_classifier_data.csv")
  model %>% fit_generator(skipgrams_generator(texts,
                                              tokenizer,
                                              skip_window,
                                              negative_samples),
                          steps_per_epoch=10000,
                          epochs=10,
                          callbacks = list(callback_model_checkpoint("data/tmp/tweet_w2v_25d.h5", 
                                                                     monitor = "loss", 
                                                                     save_best_only = TRUE),
                                           callback_early_stopping(monitor = "loss", patience=2))
  )

  model <- load_model_hdf5("data/tmp/tweet_w2v_25d.h5")  
  embedding_matrix <- get_weights(model)[[1]]
  words <- dplyr::data_frame(word=names(tokenizer$word_index),
                      id=as.integer(unlist(tokenizer$word_index)))
  words <- words %>% dplyr::filter(id <= tokenizer$num_words) %>% dplyr::arrange(id)
  row.names(embedding_matrix) <- c("UNK",words$word)
  save(embedding_matrix,file="data/embeddings/tweet_w2v_25d.Rdata")
}

if (!file.exists("tokenizers/ideo_tweet_tokenizer")) {
  tweets <- read_csv("data/ideology_classifier_data.csv")
  ideo_tokenizer <- text_tokenizer(num_words=20000)
  ideo_tokenizer <- fit_text_tokenizer(ideo_tokenizer, tweets$text)
  save_text_tokenizer(ideo_tokenizer, "tokenizers/ideo_tweet_tokenizer")
} else {
  ideo_tokenizer <- load_text_tokenizer("tokenizers/ideo_tweet_tokenizer")
}

if (!file.exists("data/embeddings/tweet_glove_25d.h5")) {
  paste("Preparing GloVe word embeddings. This may take a moment.")
  prepare_glove_embeddings("data/glove.twitter.27B/glove.twitter.27B.25d.txt",
                           embedding_dim = 25,
                           tokenizer = ideo_tokenizer,
                           out_file = "data/embeddings/tweet_glove_25d.h5")
}

if (!file.exists("data/embeddings/tweet_w2v_25d.h5")) {
  paste("Preparing word2vec embeddings. This may take a moment.")
  prepare_w2v_embeddings(tweets$text,
                         embedding_dim = 25,
                         tokenizer = ideo_tokenizer,
                         out_file = "data/embeddings/tweet_w2v_25d.h5")
  
}