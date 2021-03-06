#' prepare_glove_embeddings
#'
#' This function prepares an embedding matrix containing the words in the training data set from pre-trained GloVe embeddings.
#' @param embedding_dim Dimensionality of word embeddings. Options are 25, 50, 100, 200.
#' @param tokenizer Pre-fit keras text tokenizer.
#' @export
#' @details For more information on the GloVe embedding algorithm, visit https://nlp.stanford.edu/projects/glove/.
#' @note The GloVe embeddings are 1.3G zipped and 3.8G unzipped.
#' @note Embeddings are saved as Rdata to a folder called embeddings with the file format "tweet_glove_\{embedding_dim\}.rda"
prepare_glove_embeddings <- function(embedding_dim, tokenizer) {
  stopifnot(embedding_dim %in% list(25, 50, 100, 200))

  cwd <- getwd()
  setwd("~/.deepIdeology/")
  if (!dir.exists("glove.twitter.27B")) {
    dir.create("glove.twitter.27B")
    download <- menu(c("Yes", "No"), title="Cannot find pre-trained GloVe embeddings. Would you like to download now (1.3G)?")
    if (download == 1) download.file("http://nlp.stanford.edu/data/glove.twitter.27B.zip", "glove.twitter.27B/glove.twitter.27B.zip")
    unzip("glove.twitter.27B/glove.twitter.27B.zip", exdir="glove.twitter.27B")
    file.remove("glove.twitter.27B/glove.twitter.27B.zip")
  }

  embeddings_file <- sprintf("glove.twitter.27B/glove.twitter.27B.%sd.txt", embedding_dim)
  word_index <- tokenizer$word_index
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

  out_file <- sprintf("embeddings/tweet_glove_%sd.rda", embedding_dim)
  if (!dir.exists("embeddings")) {
    dir.create("embeddings")
  }
  save(embedding_matrix, file = out_file)
  setwd(cwd)
}


#' prepare_w2v_embeddings
#'
#' This function trains a word2vec model to create custom word embeddings from the training data set.
#' @param texts Character vector of raw text from training data.
#' @param embedding_dim Dimensionality of word embeddings. Options are 25, 50, 100, 200.
#' @param tokenizer Pre-fit keras text tokenizer.
#' @export
#' @details For a good introduction to word2vec model see Distributed Representations of Words and Phrases and their Compositionality (Mikolov et al., 2013)
#' @note Embeddings are saved as Rdata to a folder called embeddings with the file format "tweet_wv2_\{embedding_dim\}.rda"

prepare_w2v_embeddings <- function(texts, embedding_dim, tokenizer) {

  cwd <- getwd()
  setwd("~/.deepIdeology/")

  skipgrams_generator <- function(text, tokenizer, window_size, negative_samples) {
    gen <- keras::texts_to_sequences_generator(tokenizer, sample(text))
    function() {
      skip <- keras::generator_next(gen) %>%
        keras::skipgrams(
          vocabulary_size = tokenizer$num_words,
          window_size = window_size,
          negative_samples = 1
        )
      x <- purrr::transpose(skip$couples) %>% purrr::map(. %>% unlist %>% as.matrix(ncol = 1))
      y <- skip$labels %>% as.matrix(ncol = 1)
      list(x, y)
    }
  }

  skip_window <- 5       # How many words to consider left and right.
  num_sampled <- 1       # Number of negative examples to sample for each word.

  input_target <- keras::layer_input(shape = 1)
  input_context <- keras::layer_input(shape = 1)

  embedding <- keras::layer_embedding(
    input_dim = tokenizer$num_words + 1,
    output_dim = embedding_dim,
    input_length = 1,
    name = "embedding"
  )

  target_vector <- input_target %>%
    embedding() %>%
    keras::layer_flatten()

  context_vector <- input_context %>%
    embedding() %>%
    keras::layer_flatten()

  dot_product <- keras::layer_dot(list(target_vector, context_vector), axes = 1)
  output <- keras::layer_dense(dot_product, units = 1, activation = "sigmoid")

  model <- keras::keras_model(list(input_target, input_context), output)
  model %>% keras::compile(loss = "binary_crossentropy", optimizer = "adam")


  model %>% keras::fit_generator(skipgrams_generator(texts,
                                                     tokenizer,
                                                     skip_window,
                                                     negative_samples),
                          steps_per_epoch=10000,
                          epochs=10,
                          callbacks = list(keras::callback_model_checkpoint(sprintf("models/w2v_%sd.h5", embedding_dim),
                                                                            monitor = "loss",
                                                                            save_best_only = TRUE),
                                           keras::callback_early_stopping(monitor = "loss", patience=2))
  )

  model <- keras::load_model_hdf5(sprintf("models/w2v_%sd.h5", embedding_dim))
  embedding_matrix <- keras::get_weights(model)[[1]]
  words <- dplyr::data_frame(word=names(tokenizer$word_index),
                      id=as.integer(unlist(tokenizer$word_index)))
  words <- words %>% dplyr::filter(id <= tokenizer$num_words) %>% dplyr::arrange(id)
  row.names(embedding_matrix) <- c("UNK",words$word)

  out_file <- sprintf("embeddings/tweet_wv2_%sd.rda", embedding_dim)

  if (!dir.exists("embeddings")) {
    dir.create("embeddings")
  }
  save(embedding_matrix,file=out_file)
  setwd(cwd)
}
