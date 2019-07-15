# deepIdeology: Scale ideological slant of Tweets

This package allows users to identify the ideological leanings of Twitter posts with benchmark accuracy
using a Long Short-Term Memory recurrent neural network model trained on a data set of Tweets labeled through
the Amazon Mechanical Turk crowd-sourcing platform. The best-performing models are able to classify Tweets are 
liberal- or conservative-leaning with 86.90% accuracy and are able to capture both directionality and degree of
slant. This package was developed for a study of preference falsification on social media entitled "Private Partisan, 
Public Moderate: Preference Falsification on Twitter" (Gottlieb, 2018). Contact maintainer to request access.

## Installation and setup

To install the latest version of `r deepIdeology` from GitHub, run the following:
```{r}
devtools::install_github("alex-gottlieb/deepIdeology")
library(deepIdeology)
```
Following successful installation run the `r complete_setup()` command, which will finish installing the `r keras` module and set up the file caching
on which the package relies.

## predict_ideology

`r predict_ideology` is the core function of this package. It allows the user to scale the ideological slant of Tweets from 0 to 1, with values close 
to 0 indicating a strong liberal slant, values close to 1 indicating a strong conservative slant, and values around 0.5 indicating ideologically moderate or neutral content. The first time this function is called, it will likely take upwards of an hour to run, as the model will need to be trained and the word embeddings which are required to vectorize the raw text of the Tweets will need to be either downloaded (GloVe) or learned by another neural network
model (word2vec). Once a particular model or embedding configuration is used once, though, the files will be cached, allowing near-instantaneous
evaluations in the future.
