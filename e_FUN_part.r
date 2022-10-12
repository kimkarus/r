train_keras<- function(dt_in, pi, pm){
  
  if(!pi$use_gpu) {
    Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)
    #reticulate::use_condaenv("keras-cpu")
  }
  library(dplyr)
  library(reticulate)
  #
  if(pi$seed > 0){
    set.seed(pi$seed)
  }
  #
  library("keras")
  library("tensorflow")
  if(!pi$use_gpu) {
    Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)
    #reticulate::use_condaenv("keras-cpu")
  }
  #
  is_seed_set <- FALSE
  if(pi$seed > 0){
    is_seed_set <- TRUE
    if(length(find("set_random_seed")) > 0){
      set_random_seed(pi$seed)
    } else {
      is_seed_set <- tryCatch({
        use_session_with_seed(pi$seed)
        is_seed_set <- TRUE
      }, error = function(e){
        is_seed_set <- FALSE
      })
      
    }
  }
  if(pi$os == "linux"){
    #reticulate::use_python("~/kimkarus/kimkarusenv/bin/python")
    ##use_python("/usr/local/bin/python")
    #reticulate::use_virtualenv("~/kimkarus/kimkarusenv")
    #use_condaenv("kimkarusenv")
    #reticulate::use_virtualenv("r-tensorflow")
    #reticulate::py_module_available("tensorflow")
  }
  
  if(pi$os == "windows") use_condaenv(condaenv = "r-reticulate", required = TRUE)
  
  
  
  x_train <- dt_in$prepr_train[,dt_in$best] %>% as.matrix()
  x_validation <- dt_in$prepr_validation[,dt_in$best] %>% as.matrix()
  
  
  
  #...................................HIDE........................
  
  
  
  model <- keras_get_layers(dt_in, model, pi, pm, x_train, y_train, x_validation, y_validation)
  
  model <- keras_get_optimizer(model, pi, pm)
  
  if(pi$use_gpu) callbacks = NULL
  
  if(pi$use_reduce_lr){
    callbacks = c(
      keras_get_callback_reduce_lr_on_plateau(pi,pm)
    )
  } else {
    callbacks = c(
      keras_get_callback_early_stopping(pi,pm)
    )
  }
  
  if(!pi$use_callbacks_with_earlystopping) callbacks = NULL
  
  
  
  if (pm$opt == "bdg") pm$Bs <- nrow(dt_in$prepr_train[,dt_in$best])
  
  if(pi$use_mix){
    sae <- model %>% fit(
      x_train, y_train, 
      epochs = pm$Ep, batch_size = pm$Bs, 
      validation_data = list(x_validation, y_validation),
      callbacks = callbacks,
      verbose = pi$verbose
      #shuffle = TRUE
    )
    
  } else {
    sae <- model %>% fit(
      x_train, y_train, 
      epochs = pm$Ep, batch_size = pm$Bs, 
      callbacks = callbacks,
      verbose = pi$verbose
      #shuffle = TRUE
    )
  }
  
  #history = sae %>% train_on_batch(
  #  x_train, y_train
  #)
  
  
  k_clear_session()
  
  return (model)
}

keras_get_callback_reduce_lr_on_plateau <- function(pi,pm){
  if(pi$ea_type_y == 2 || pi$ea_type_y == 3){
    #if(pi$ratio == 1.0) return(callback_reduce_lr_on_plateau(monitor='loss', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = c("auto", "min", "max")))
    if(pi$ratio == 1.0) return(callback_reduce_lr_on_plateau(monitor='loss', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = c("auto", "min", "max")))
    #return(callback_reduce_lr_on_plateau(monitor='val_loss', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = c("auto", "min", "max")))
    return(callback_reduce_lr_on_plateau(monitor='val_loss', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = c("auto", "min", "max")))
  } else {
    if(pi$soft){
      #if(pi$ratio == 1.0) return(callback_reduce_lr_on_plateau(monitor='accuracy', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = "max")) 
      if(pi$ratio == 1.0) return(callback_reduce_lr_on_plateau(monitor='accuracy', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = "max")) 
      #return(callback_reduce_lr_on_plateau(monitor='val_accuracy', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = "max")) 
      return(callback_reduce_lr_on_plateau(monitor='val_accuracy', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = "max"))
      #return(callback_reduce_lr_on_plateau(monitor='accuracy', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = "max"))
    } else{
      #if(pi$ratio == 1.0) return(callback_reduce_lr_on_plateau(monitor='loss', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = "min"))
      if(pi$ratio == 1.0) return(callback_reduce_lr_on_plateau(monitor='loss', factor=0.15, restore_best_weights = T, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = "min"))
      #return(callback_reduce_lr_on_plateau(monitor='loss', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = "min"))
      return(callback_reduce_lr_on_plateau(monitor='val_loss', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = "min"))
      #return(callback_reduce_lr_on_plateau(monitor='val_accuracy', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = "max"))
      #return(callback_reduce_lr_on_plateau(monitor='accuracy', factor=0.15, patience=pi$keras_patience, min_lr=0.00000000000000000000001, verbose=pi$verbose, mode = "max"))
    }
  }
  
}

keras_get_callback_early_stopping <-function(pi,pm){
  if(pi$ea_type_y == 2 || pi$ea_type_y == 3){
    if(pi$ratio == 1.0) return(callback_early_stopping(monitor = "loss", patience=pi$keras_patience, mode = c("auto", "min", "max"), restore_best_weights = T, verbose=pi$verbose))
    return(callback_early_stopping(monitor = "val_loss", patience=pi$keras_patience, mode = "min", restore_best_weights = T, verbose=pi$verbose))
    #return(callback_early_stopping(monitor = "loss", patience=pi$keras_patience, mode = "min", restore_best_weights = T, verbose=1))
    
  } else{
    if(pi$soft){
      if(pi$ratio == 1.0) return(callback_early_stopping(monitor = "accuracy", patience=pi$keras_patience, mode = "max", restore_best_weights = T, verbose=pi$verbose))
      return(callback_early_stopping(monitor = "val_accuracy", patience=pi$keras_patience, mode = "max", restore_best_weights = T, verbose=pi$verbose))
      #return(callback_early_stopping(monitor = "accuracy", patience=pi$keras_patience, mode = "max", restore_best_weights = T, verbose=1))
    } else {
      if(pi$ratio == 1.0) return(callback_early_stopping(monitor = "loss", patience=pi$keras_patience, mode = "min", restore_best_weights = T, verbose=pi$verbose))
      #return(callback_early_stopping(monitor = "loss", patience=pi$keras_patience, mode = "min", restore_best_weights = T, verbose=pi$verbose))
      return(callback_early_stopping(monitor = "val_loss", patience=pi$keras_patience, mode = "min", restore_best_weights = T, verbose=pi$verbose))
      #return(callback_early_stopping(monitor = "loss", patience=pi$keras_patience, mode = "min", restore_best_weights = T, verbose=1))
    }
  }
}

keras_get_optimizer <- function(sae, pi, pm){
  #if(pi$soft){
  #  sae <- keras_get_optimizer_sgd(sae, pi, pm)  
  #}else {
  if(pm$opt == "rmsprop") {
    sae <- keras_get_optimizer_rmsprop(sae, pi, pm)
  } else if(pm$opt == "adam"){
    sae <- keras_get_optimizer_adam(sae, pi, pm)  
  } else if(pm$opt == "nadam"){
    sae <- keras_get_optimizer_nadam(sae, pi, pm)  
  } else if(pm$opt == "sdg"){
    sae <- keras_get_optimizer_sgd(sae, pi, pm)
  } else {
    sae <- keras_get_optimizer_sgd(sae, pi, pm)
  }
  #}
  return(sae)
}

keras_get_out_layer <- function(sae, dt, pi, pm) {
  if(pi$ea_type_y==2 || pi$ea_type_y == 3){
    sae %>% layer_dense(units = ifelse(pi$n_y_predicts < 2, 1, pi$n_y_predicts), activation = pm$n_sae)  
  } else {
    if(pi$ea_type_y==0){
      if(pi$n_y_predicts > 1){
        sae %>% layer_dense(units = pi$n_y_predicts, activation = pm$n_out)
      } else {
        sae %>% layer_dense(units = ifelse(pi$soft, length(dt$ylevels), 1), activation = pm$n_out)
      }
    } else {
      sae %>% layer_dense(units = ifelse(pi$soft, length(dt$ylevels), 1), activation = pm$n_out)
    }
    
  }
  return(sae)
}

keras_get_layers <- function(dt, sae, pi, pm, x_train, y_train, x_validation, y_validation){
  sae <- keras_model_sequential()
  if(pi$ea_sig_type == 11){
    sae <- keras_get_layers_rnn(dt, sae, pi, pm, x_train, y_train, x_validation, y_validation) 
  } else if(pi$ea_sig_type == 15){
    sae <- keras_get_layers_lstm(dt, sae, pi, pm, x_train, y_train, x_validation, y_validation) 
  } else if(pi$ea_sig_type == 16){
    sae <- keras_get_layers_cnn(dt, sae, pi, pm, x_train, y_train, x_validation, y_validation) 
  }else{
    
    sae %>% layer_dense(units = dim(x_train)[2]*2)
    count_neurons <- pi$ea_count_neurons
    for(i in 1:pi$ea_count_hidden_layers){
      if(pm$n_act == "lrelu"){
        sae %>% layer_dense(units = count_neurons, activation = layer_activation_leaky_relu(), activity_regularizer = regularizer_l2(0.001))
      } else {
        sae %>% layer_dense(units = count_neurons, activation = pm$n_act, activity_regularizer = regularizer_l2(0.001))
      }
      if(pi$use_batch_normalization) sae %>% layer_batch_normalization()
      if(pi$use_dropout) sae %>% layer_dropout(pm$dropout)
    }
    
    sae <- keras_get_out_layer(sae, dt, pi, pm)
  }
  
  return(sae)
keras_get_layers_lstm <- function(dt, sae, pi, pm, x_train, y_train, x_validation, y_validation){
  
  #sae <- keras_model_sequential()
  #sae %>% layer_lstm(units=pi$ea_count_neurons, input_shape=c(pi$PrevBar,length(dt$best)), activation=pm$n_act, return_sequences = TRUE, activity_regularizer = regularizer_l2(0.001))
  sae %>% layer_lstm(units=pi$ea_count_neurons, input_shape=dim(x_train)[2:3], activation=ifelse(pm$n_act == "lrelu",layer_activation_leaky_relu(), pm$n_act), return_sequences = TRUE, activity_regularizer = regularizer_l2(0.001))
  if(pi$use_batch_normalization) sae %>% layer_batch_normalization()
  
  
  if(pi$ea_count_hidden_layers > 1){
    for(i in 1:pi$ea_count_hidden_layers){ 
      sae %>% layer_lstm(units=pi$ea_count_neurons, input_shape=dim(x_train)[2:3], activation=ifelse(pm$n_act == "lrelu",layer_activation_leaky_relu(), pm$n_act), return_sequences = TRUE)
    }
  } else {
    sae %>% layer_lstm(units=pi$ea_count_neurons, input_shape=dim(x_train)[2:3], activation=ifelse(pm$n_act == "lrelu",layer_activation_leaky_relu(), pm$n_act), return_sequences = TRUE)
  }
  if(pi$use_batch_normalization) sae %>% layer_batch_normalization()
  if(pi$use_dropout) sae %>% layer_dropout(pm$dropout)
  
  sae %>% layer_lstm(units=pi$ea_count_neurons, input_shape=dim(x_train)[2:3], activation=ifelse(pm$n_act == "lrelu",layer_activation_leaky_relu(), pm$n_act), return_sequences = FALSE)
  
  sae %>% layer_dense(units = pi$ea_count_neurons, activation = ifelse(pm$n_act == "lrelu", layer_activation_leaky_relu(), pm$n_act))
  
  if(pi$use_batch_normalization) sae %>% layer_batch_normalization()
  if(pi$use_dropout) sae %>% layer_dropout(pm$dropout)
  
  sae <- keras_get_out_layer(sae, dt, pi, pm)
  
  return(sae)
}

keras_get_layers_cnn <- function(dt, sae, pi, pm, x_train, y_train, x_validation, y_validation){
  
  #sae <- keras_model_sequential()
  
  filters <- pm$cnn_filtres
  kernel_size <- pm$cnn_kernel_size
  ea_count_neurons <- pi$ea_count_neurons
  name_index <- 1
  
  #input_shape <- ifelse(length(dt$best) > 1, c(pi$PrevBar,length(dt$best)), c(pi$PrevBar))
  input_shape <- c(pi$PrevBar, length(dt$best))
  sae %>% layer_conv_1d(filters=filters, kernel_size=kernel_size, activation=pm$n_act, input_shape = input_shape, activity_regularizer = regularizer_l2(0.001), name = paste0("Conv1D_", name_index) )
  
  kernel_size <- floor(kernel_size / 2)
  #filters <- floor(filters / 2)
  
  #sae %>% layer_max_pooling_1d(pool_size = kernel_size)
  sae %>% layer_max_pooling_1d()
  
  sae %>% layer_conv_1d(filters=filters, kernel_size=kernel_size, activation=pm$n_act, activity_regularizer = regularizer_l2(0.001), name = paste0("Conv1D_", 2) )
  if(pi$use_batch_normalization) sae %>% layer_batch_normalization()
  
  kernel_size <- floor(kernel_size / 2)
  #filters <- floor(filters / 2)
  
  sae %>% layer_max_pooling_1d()
  if(pi$use_batch_normalization) sae %>% layer_batch_normalization()
  
  #sae %>% layer_conv_1d(filters=filters, kernel_size=kernel_size, activation=pm$n_act, activity_regularizer = regularizer_l2(0.001), name = paste0("Conv1D_", 3) )
  
  #kernel_size <- floor(kernel_size / 2)
  #filters <- floor(filters / 2)
  
  #sae %>% layer_max_pooling_1d()
  
  sae %>% layer_flatten()
  if(pi$use_batch_normalization) sae %>% layer_batch_normalization()
  if(pi$use_dropout) sae %>% layer_dropout(pm$dropout)
  
  ea_count_neurons <- (kernel_size / 2) * pm$cnn_filtres
  
  ea_count_neurons <- pi$ea_count_neurons
  
  if(pi$ea_count_hidden_layers > 0){
    for(i in 1:pi$ea_count_hidden_layers){
      sae %>% layer_dense(units=ea_count_neurons, activation=pm$n_act, activity_regularizer = regularizer_l2(0.001))
      #sae %>% layer_dense(units=pi$ea_count_neurons, activation=pm$n_act, activity_regularizer = regularizer_l2(0.001))
      if(pi$use_batch_normalization) sae %>% layer_batch_normalization()
      if(pi$use_dropout) sae %>% layer_dropout(pm$dropout)
      ea_count_neurons <- ifelse(floor(ea_count_neurons / 2) < 2, 2, floor(ea_count_neurons / 2))
    }
  }
  
  sae <- keras_get_out_layer(sae, dt, pi, pm)
  
  return(sae)
}

}
