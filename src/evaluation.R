# Evaluation using CORA dataset
# dataset available at:
# https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
# Documentation available at:
# https://relational.fit.cvut.cz/dataset/CORA
# https://linqs.soe.ucsc.edu/data

# Here we use modified cora dataset found at https://github.com/thunlp/CANE/tree/master/datasets/cora

# classes:
    # Case_Based
    # Genetic_Algorithms
    # Neural_Networks
    # Probabilistic_Methods
    # Reinforcement_Learning
    # Rule_Learning
    # Theory


#### Chargement des données

data_filename <- '../dataset/cora_modified/data.txt'
group_filename <- '../dataset/cora_modified/group.txt'

data.group <- readLines(group_filename)
data.group <- as.matrix(data.group)
data.group <- factor(data.group)
empty <- which(data.group == '') # On relève les indices des cases vides
data.group <- data.group[-empty] # On enlève les données vides

data.text <- readLines(data_filename, encoding = "utf8")
# On enlève les données documents correspondant aux groupes non renseignés
data.text <- data.text[-empty] 
# On créé une liste de documents dont chaque élément est une liste de mots. Les
# caractères son mis en minuscule.
data.docsplit <- strsplit(tolower(data.text), ' ') 

#### Extraction des vecteurs de documents

source('modules.R')
source('baseline.R')
source('SIF_Document_Embedding.R')
source('doc_embedding_topic_model.R')

# load word embedding vectors
# file_vec <- "../dataset/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec"
word_embedding <- load_word_embedding_vector("../ressources/glove_cora.txt", nrows = NULL)
# word_embedding <- load_word_embedding_vector("../dataset/glove.6B/glove.6B.300d.txt", nrows = 50000)
# word_embedding <- load_word_embedding_vector("../dataset/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec", nrows = 50000)

# text8
text8 <- readLines('../dataset/text8/text8', warn=FALSE)
text8_cora <- c(text8, data.text)

# Calcul des pondérations tfidf et okapi
tfidf_avg <- weight_embedding(corpus = data.text, ponderation = 'tfidf')
okapi_avg <- weight_embedding(corpus = data.text, ponderation = "okapi")

doc_emb_bar <- sapply( 1:length(data.docsplit), 
                       function(doc_id) document_embedding(word_embedding, data.docsplit, doc_id, methode = "barycentre") )
doc_emb_tfidf <- sapply( 1:length(data.docsplit), 
                         function(doc_id) document_embedding(word_embedding, data.docsplit, doc_id, methode = "tfidf", weights = tfidf_avg) )
doc_emb_okapi <- sapply( 1:length(data.docsplit), 
                         function(doc_id) document_embedding(word_embedding, data.docsplit, doc_id, methode = "okapi", weights = okapi_avg) )

doc_emb_SIF <- SIF_document_embedding( data.docsplit, word_embedding, D_proba_w = text8_cora)
doc_emb_tm_main_topic <- tm_embedding_main_topic(data.text, word_embedding, k_topics = 10, method = 'LDA')
doc_emb_tm <- tm_embedding(data.text, word_embedding, k_topics = 10, method = 'LDA')

#### Tâche d'évaluation: Classification avec 2 méthodes de régression (Bayésien Naïf / SVM)
# Création des données d'entrainement et de test


### Entrainement et test de modèles avec Bayésien Naïf et SVM
library(MASS)
library(e1071)
source('evaluation_functions.R')

# ta <- 0.1 # Pourcentage de données d'entrainement
for(ta in c(0.1, 0.2, 0.3, 0.4, 0.5)){
  v_d <- readRDS(file = '../ressources/SIF_vectors.Rda')
  
  prf_nb_bar <- c()
  prf_svm_bar <- c()
  
  prf_nb_tfidf <- c()
  prf_svm_tfidf <- c()
  
  prf_nb_okapi <- c()
  prf_svm_okapi <- c()
  
  prf_nb_sif <- c()
  prf_svm_sif <- c()
  
  prf_nb_tm_main_topic <- c()
  prf_svm_tm_main_topic <- c()
  
  prf_nb_tm <- c()
  prf_svm_tm <- c()
  
  for(i in 1:10){
    svMisc::progress(i, max.value = 10 )
    # génération des même indices pour que les données soient comparées sur les même exemples
    ind <- generate_train_test_ind(doc_emb_bar, data.group, training_amount = ta ) 
    
    ###################
    ## Vecteurs barycentriques ##
    dataset <- assign_ind(doc_emb_bar, data.group, ind)
    
    # Bayésien Naïf
    nb_bar <- naiveBayes(y ~ ., data = dataset$train)
    pred_nb_bar <- predict(nb_bar, dataset$test)
    prf_nb_bar <- c(prf_nb_bar, prf(dataset$test$y, pred_nb_bar))
    
    # SVM
    svm_model_bar = svm(y ~ ., data = dataset$train )
    pred_svm_bar <- predict(svm_model_bar, dataset$test )
    prf_svm_bar <- c(prf_svm_bar, prf(dataset$test$y, pred_svm_bar))
  
    ###################
    ## Vecteurs tf-idf ##
    dataset <- assign_ind(doc_emb_tfidf, data.group, ind)
    
    # Bayésien Naïf
    nb_tfidf <- naiveBayes(y ~ ., data = dataset$train)
    pred_nb_tfidf <- predict(nb_tfidf, dataset$test )
    prf_nb_tfidf <- c(prf_nb_tfidf, prf(dataset$test$y, pred_nb_tfidf) )
    
    # SVM
    svm_model_tfidf = svm(y ~ ., data = dataset$train )
    pred_svm_tfidf <- predict(svm_model_tfidf, dataset$test )
    prf_svm_tfidf <- c(prf_svm_tfidf, prf(dataset$test$y, pred_svm_tfidf) )
  
    ###################
    ## Vecteurs okapi ##
    dataset <- assign_ind(doc_emb_okapi, data.group, ind)
    
    # Bayésien Naïf
    nb_okapi <- naiveBayes(y ~ ., data = dataset$train)
    pred_nb_okapi <- predict(nb_okapi, dataset$test )
    prf_nb_okapi <- c( prf_nb_okapi, prf(dataset$test$y, pred_nb_okapi) )
    
    # SVM
    svm_model_okapi = svm(y ~ ., data = dataset$train )
    pred_svm_okapi <- predict(svm_model_okapi, dataset$test )
    prf_svm_okapi <- c( pred_svm_okapi, prf(dataset$test$y, pred_svm_okapi) )
  
  
    ###################
    ## Vecteurs SIF ##
    dataset <- assign_ind(v_d, data.group, ind)
    
    # Bayésien Naïf
    nb_sif <- naiveBayes(y ~ ., data = dataset$train)
    pred_nb_sif <- predict(nb_sif, dataset$test )
    prf_nb_sif <- c( prf_nb_sif, prf(dataset$test$y, pred_nb_sif) )
    
    # SVM
    svm_model_sif = svm(y ~ ., data = dataset$train )
    pred_svm_sif <- predict(svm_model_sif, dataset$test )
    prf_svm_sif <- c( pred_svm_sif, prf(dataset$test$y, pred_svm_sif) )
  
  
    ###################
    ## Vecteurs TM main topic ##
    dataset <- assign_ind(doc_emb_tm_main_topic, data.group, ind)
    
    # Bayésien Naïf
    nb_tm_main_topic <- naiveBayes(y ~ ., data = dataset$train)
    pred_nb_tm_main_topic <- predict(nb_tm_main_topic, dataset$test )
    prf_nb_tm_main_topic <- c( prf_nb_tm_main_topic, prf(dataset$test$y, pred_nb_tm_main_topic) )
    
    # SVM
    svm_model_tm_main_topic = svm(y ~ ., data = dataset$train )
    pred_svm_tm_main_topic <- predict(svm_model_tm_main_topic, dataset$test )
    prf_svm_tm_main_topic <- c( pred_svm_tm_main_topic, prf(dataset$test$y, pred_svm_tm_main_topic) )
  
    
    ###################
    ## Vecteurs TM ##
    dataset <- assign_ind(doc_emb_tm, data.group, ind)
    
    # Bayésien Naïf
    nb_tm <- naiveBayes(y ~ ., data = dataset$train)
    pred_nb_tm <- predict(nb_tm, dataset$test )
    prf_nb_tm <- c( prf_nb_tm, prf(dataset$test$y, pred_nb_tm) )
    
    # SVM
    svm_model_tm = svm(y ~ ., data = dataset$train )
    pred_svm_tm <- predict(svm_model_tm, dataset$test )
    prf_svm_tm <- c( pred_svm_tm, prf(dataset$test$y, pred_svm_tm) )
  }
  
  mean_prf_nb_bar <- mean_prf(prf_nb_bar)
  mean_prf_svm_bar <- mean_prf(prf_svm_bar)
  
  mean_prf_nb_tfidf <- mean_prf(prf_nb_tfidf)
  mean_prf_svm_tfidf <- mean_prf(prf_svm_tfidf)
  
  mean_prf_nb_okapi <- mean_prf(prf_nb_okapi)
  mean_prf_svm_okapi <- mean_prf(prf_svm_okapi)
  
  mean_prf_nb_sif <- mean_prf(prf_nb_sif)
  mean_prf_svm_sif <- mean_prf(prf_svm_sif)
  
  mean_prf_nb_tm_main_topic <- mean_prf(prf_nb_tm_main_topic)
  mean_prf_svm_tm_main_topic <- mean_prf(prf_svm_tm_main_topic)
  
  mean_prf_nb_tm <- mean_prf(prf_nb_tm)
  mean_prf_svm_tm <- mean_prf(prf_svm_tm)
  
  print("naive bayes")
  print(paste("bar:", mean_prf_nb_bar$f1_score))
  print(paste("tfidf:", mean_prf_nb_tfidf$f1_score))
  print(paste("okapi:", mean_prf_nb_okapi$f1_score))
  print(paste("sif:", mean_prf_nb_sif$f1_score))
  print(paste("tm:", mean_prf_nb_tm_main_topic$f1_score))
  print(paste("tm:", mean_prf_nb_tm$f1_score))
  
  print("SVM")
  print(paste("bar:", mean_prf_svm_bar$f1_score))
  print(paste("tfidf:", mean_prf_svm_tfidf$f1_score))
  print(paste("okapi:", mean_prf_svm_okapi$f1_score))
  print(paste("sif:", mean_prf_svm_sif$f1_score))
  print(paste("tm:", mean_prf_svm_tm_main_topic$f1_score))
  print(paste("tm:", mean_prf_svm_tm$f1_score))
  
  out_csv <- data.frame(
    NB = list(bar = mean_prf_nb_bar$f1_score, 
              tfidf = mean_prf_nb_tfidf$f1_score,
              okapi = mean_prf_nb_okapi$f1_score,
              sif = mean_prf_nb_sif$f1_score,
              tm_main_topic = mean_prf_nb_tm_main_topic$f1_score,
              tm = mean_prf_nb_tm$f1_score),
    SVM = list(bar = mean_prf_svm_bar$f1_score, 
               tfidf = mean_prf_svm_tfidf$f1_score,
               okapi = mean_prf_svm_okapi$f1_score,
               sif = mean_prf_svm_sif$f1_score,
               tm_main_topic = mean_prf_svm_tm_main_topic$f1_score,
               tm = mean_prf_svm_tm$f1_score)
    )
  write.csv( format(t(out_csv)*100, digits=2, nsmall=2), 
             file = paste0('results_', toString(ta), 'training','.csv') 
          )
}

