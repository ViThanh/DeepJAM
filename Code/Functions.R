# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Author:       Vi Thanh Pham
# GitHub:       https://github.com/ViThanh
# Description:  Define functions for DeepJAM algorithm
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# Load libraries ----------------------------------------------------------
library(abind)
library(keras)
library(magrittr)
library(reticulate)
library(tensorflow)
tfp <- tf_probability()

# -------------------------------------------------------------------------
# Warping -----------------------------------------------------------------
# -------------------------------------------------------------------------
# Warping function --------------------------------------------------------
Warp <- \(f, gamma){
  # Apply the warping action f(gamma(t))
  # f: functions to be warped
  # gamma: warping functions
  
  # Warp
  warp <- tfp$math$batch_interp_regular_1d_grid(
    x = gamma,
    x_ref_min = 0L,
    x_ref_max = 1L,
    y_ref = f,
    axis = length(dim(gamma)) - 1L)
  return(warp)
}

# Inverse -----------------------------------------------------------------
Inverse <- \(f, dtype = if("tensorflow.tensor" %in% class(f)) f$dtype else tf$float64){
  # Calculate an inverse of functions f
  # f: functions
  
  # To array
  if("numeric" %in% class(f)) f <- as.array(f)
  
  # Extract dimensions
  dims <- dim(f)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  margin <- axis + 1
  
  # Linear space
  lin <- tf$cast(tf$linspace(start = 0L, 
                             stop = 1L, 
                             num = dims[margin]), dtype = dtype)
  
  # Calculate inverse
  inverse <- tfp$math$batch_interp_rectilinear_nd_grid(
    x = tf$expand_dims(lin, axis = -1L), 
    x_grid_points = tuple(f), 
    y_ref = lin, 
    axis = 0L)
  return(inverse)
}

# Gradient ----------------------------------------------------------------
Gradient <- \(f){
  # Calculate the numerical derivative of f(t)
  # f: functions
  
  # To array
  if("numeric" %in% class(f)) f <- as.array(f)
  
  # Extract dimensions
  dims <- dim(f)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  margin <- axis + 1
  
  # Extract length and time
  n <- dims[margin]
  h <- 1 / (n - 1)
  
  # Forward differences on boundaries and interior points
  if(is.array(f)){
    g1 <- (asub(f, 2, dims = margin, drop = F) - asub(f, 1, dims = margin, drop = F)) / h
    gn <- (asub(f, n, dims = margin, drop = F) - asub(f, n - 1, dims = margin, drop = F)) / h
    g <- (asub(f, 3:n, dims = margin, drop = F) - asub(f, 1:(n - 2), dims = margin, drop = F)) / (2 * h)
  }else{
    g1 <- tf$expand_dims((tf$gather(params = f, indices = 1L, axis = axis) - tf$gather(params = f, indices = 0L, axis = axis)) / h, axis = axis)
    gn <- tf$expand_dims((tf$gather(params = f, indices = n - 1L, axis = axis) - tf$gather(params = f, indices = n - 2L, axis = axis)) / h, axis = axis)
    g <- (tf$gather(params = f, indices = 2:(n - 1), axis = axis) - tf$gather(params = f, indices = 0:(n - 3), axis = axis)) / (2 * h)
  }
  
  # Concatenate
  return(if(is.array(f)) abind(g1, g, gn, along = margin) else tf$concat(list(g1, g, gn), axis = axis))
}

# Square-root slope function representation -------------------------------
SRSF <- \(f){
  # Calculate the square-root slope function representation of f
  # f: functions
  
  # Calculate gradient
  f_dot <- Gradient(f)
  
  # Calculate SRSF representation
  return(sign(f_dot) * sqrt(abs(f_dot)))
}

# Warping function in SRSF space ------------------------------------------
WarpSRSF <- \(q, gamma){
  # Apply the warping action in SRSF space
  # q: function in SRSF space
  # gamma: warping function
  
  # To array
  if("numeric" %in% class(q)) q <- as.array(q)
  if("numeric" %in% class(gamma)) gamma <- as.array(gamma)
  
  # Extract dimensions
  l_dims_q <- length(dim(q))
  l_dims_gamma <- length(dim(gamma))
  
  # Calculate gradient of warping functions
  gamma_dot <- tf$keras$backend$clip(Gradient(gamma), min_value = tf$keras.backend$epsilon(), max_value = NULL)
  if(l_dims_q != l_dims_gamma){
    for(i in seq_len(l_dims_q - l_dims_gamma)){
      gamma_dot <- tf$expand_dims(gamma_dot, axis = -1L)
    }
  }
  
  # Calculate warping in SRSF space
  return(Warp(q, gamma) * sqrt(gamma_dot))
}

# Trapezoidal rule --------------------------------------------------------
Trapz <- \(f){
  # Calculate the increments per time interval
  # f: functions
  
  # To array
  if("numeric" %in% class(f)) f <- as.array(f)
  
  # Extract dimensions
  dims <- dim(f)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  margin <- axis + 1
  
  # Extract length and time
  n <- dims[margin]
  h <- 1 / (n - 1)
  
  # Calculate the trapezoidal summands
  trapz <- if(is.array(f)){
    (asub(f, 1:(n - 1), dim = margin, drop = F) + asub(f, 2:n, dim = margin, drop = F)) * h / 2
  }else{
    (tf$gather(params = f, indices = 0:(n - 2), axis = axis) + tf$gather(params = f, indices = 1:(n - 1), axis = axis)) * h / 2
  }
  return(trapz)
}

# Transform SRSF to function ----------------------------------------------
SRSFToFunction <- \(q){
  # Transform SRSF to function
  # q: SRSF
  
  # To array
  if("numeric" %in% class(q)) q <- as.array(q)
  
  # Extract dimensions
  dims <- dim(q)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  margin <- axis + 1
  
  # Calculate trapezoidal summands
  trapz <- Trapz(q * abs(q))
  
  # Extract function
  cumtrapz <- if(is.array(trapz)){
    if(axis == 0){
      as.array(c(0, cumsum(trapz)))
    }else{
      keepdims <- setdiff(seq_along(dims), margin)
      dims[margin] <- 1
      abind(array(0, dims), aperm(apply(trapz, keepdims, cumsum), order(c(margin, keepdims))), along = margin)
    }
  }else{
    tf$concat(list(tf$zeros_like(tf$expand_dims(tf$gather(params = trapz, indices = 0L, axis = axis), axis = axis)), tf$cumsum(trapz, axis = axis)), axis = axis)
  }
  return(cumtrapz)
}

# Transform element on functional unit sphere to warping function ---------
SRSFToWarp <- \(psi){
  # Transform an element on functional unit sphere to warping function
  # psi: function
  
  # To array
  if("numeric" %in% class(psi)) psi <- as.array(psi)
  
  # Extract dimensions
  dims <- dim(psi)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  margin <- axis + 1
  
  # Extract length
  n <- dims[margin]
  
  # Calculate cumulative trapezoidal summands
  cumtrapz <- SRSFToFunction(psi)
  
  # Normalize to [0, 1] interval (note that cumtrapz already starts at 0)
  warp <- if(is.array(cumtrapz)){
    cumtrapz_last <- asub(cumtrapz, n, dims = margin, drop = F)
    cumtrapz / abind(lapply(seq_len(n), \(x) cumtrapz_last), along = margin)
  }else{
    cumtrapz / tf$expand_dims(tf$gather(params = cumtrapz, indices = n - 1L, axis = axis), axis = axis)
  }
  return(warp)
}

# Inverse of the exponential map ------------------------------------------
ExpMapInverse <- \(psi, p = NULL){
  # Projects the functions psi on unit sphere to tangent space
  # psi: functions on unit sphere
  # p: point on unit sphere
  
  # To array
  if("numeric" %in% class(psi)) psi <- as.array(psi)
  if(!is.null(p)){
    if("numeric" %in% class(p)) p <- as.array(p)
  }
  
  # Extract dimensions
  dims <- dim(psi)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  
  # Define point on unit sphere
  if(is.null(p)){
    p <- if(is.array(psi)){
      array(1, dim = dims)
    }else{
      tf$ones_like(tf$expand_dims(tf$gather(psi, indices = 0L, axis = 0L), axis = 0L), dtype = psi$dtype)
    }
  }else{
    if(is.array(psi) & (axis == 1)){
      p <- abind(lapply(seq_len(dims[1]), \(x) p), along = 1)
    }
  }
  
  # Calculate theta
  theta <- if(is.array(p)){
    angle <- if(axis == 0) sum(Trapz(psi * p)) else apply(Trapz(psi * p), axis, sum)
    acos(ifelse(angle < 0, 0, ifelse(angle > 1, 1, angle)))
  }else{
    tf$acos(tf$clip_by_value(tf$reduce_sum(Trapz(psi * p), axis = axis, keepdims = T), clip_value_min = 0, clip_value_max = 1))
  }
  
  # Calculate exponential map inverse
  exp_map_inverse <- if(is.array(psi)){
    inverse <- theta / sin(theta) * (psi - p * cos(theta))
    inverse[is.nan(inverse)] <- 0
    inverse
  }else{
    tf$where(tf$equal(theta, 0),
             tf$zeros_like(psi),
             theta / sin(theta) * (psi - p * cos(theta)))
  }
  return(exp_map_inverse)
}

# Exponential map ---------------------------------------------------------
ExpMap <- \(v, p = NULL){
  # Projects the point v in the tangent space at identity warp to functional space
  # v: vectors in the tangent space
  # p: point on unit sphere
  
  # To array
  if("numeric" %in% class(v)) v <- as.array(v)
  
  # Extract dimensions
  dims <- dim(v)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  
  # Define point on unit sphere
  if(is.null(p)){
    p <- if(is.array(v)){
      array(1, dim = dims)
    }else{
      tf$ones_like(tf$expand_dims(tf$gather(v, indices = 0L, axis = 0L), axis = 0L))
    }
  }else{
    if(is.array(v) & (axis == 1)){
      p <- abind(lapply(seq_len(dims[1]), \(x) p), along = 1)
    }
  }
  
  # Calculate norm
  norm <- if(is.array(v)){
    if(axis == 0) sqrt(sum(Trapz(v ^ 2))) else sqrt(apply(Trapz(v ^ 2), axis, sum))
  }else{
    sqrt(tf$reduce_sum(Trapz(v ^ 2), axis = axis, keepdims = T))
  }
  
  # Exponential map
  exp_map <- if(is.array(v)){
    map <- cos(norm) * p + sin(norm) * v / norm
    map[is.nan(map)] <- rep(asub(p, 1, 1), each = sum(norm == 0))
    map
  }else{
    tf$where(tf$equal(norm, 0),
             p,
             cos(norm) * p + sin(norm) * v / norm)
  }
  return(exp_map)
}

# -------------------------------------------------------------------------
# Multiscale warping model ------------------------------------------------
# -------------------------------------------------------------------------
# Extend warp -------------------------------------------------------------
ExtendWarp <- \(gamma, K){
  # Extend warps gamma to K periods
  # gamma: warping functions
  # K: number of periods
  
  # To array
  if("numeric" %in% class(gamma)) gamma <- as.array(gamma)
  
  # Extract dimensions
  dims <- dim(gamma)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  margin <- axis + 1
  
  # Extract length of a period
  P <- dims[margin]
  
  # Extract extension
  extension <- if(is.array(gamma)){
    abind(lapply(seq_len(K + 1), \(x) if(x <= K) asub(gamma, 1:(P - 1), dims = margin, drop = F) + x - 1 else asub(gamma, P, dims = margin, drop = F) + K - 1), along = margin) / K
  }else{
    tf$concat(unlist(list(lapply(seq_len(K), \(x) tf$gather(gamma, indices = 0:(P - 2), axis = axis) + x - 1),
                          tf$expand_dims(tf$gather(gamma, indices = P - 1L, axis = axis), axis = axis) + K - 1)), axis = axis) / K
  }
  return(extension)
}

# Extend function ---------------------------------------------------------
ExtendFunction <- \(f, K){
  # Extend functions f for a specific number of periods K
  # f: functions
  # K: number of periods
  
  # To array
  if("numeric" %in% class(f)) f <- as.array(f)
  
  # Extract dimensions
  dims <- dim(f)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  margin <- axis + 1
  
  # Extract length of a period
  P <- dims[margin]
  
  # Extract extension
  extension <- if(is.array(f)){
    abind(lapply(seq_len(K + 1), \(x) if(x <= K) asub(f, 1:(P - 1), dims = margin, drop = F) else asub(f, P, dims = margin, drop = F)), along = margin)
  }else{
    tf$concat(unlist(list(lapply(seq_len(K), \(x) tf$gather(f, indices = 0:(P - 2), axis = axis)),
                          tf$expand_dims(tf$gather(f, indices = P - 1L, axis = axis), axis = axis))), axis = axis)
  }
  return(extension)
}

# Split periods -----------------------------------------------------------
SplitPeriod <- \(f, K){
  # Split functions f into K periods
  # f: functions
  # K: number of periods
  
  # To array
  if("numeric" %in% class(f)) f <- as.array(f)
  
  # Extract dimensions
  dims <- dim(f)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  margin <- axis + 1
  
  # Extract length of a period
  P <- as.integer((dims[margin] + K - 1) / K)
  
  # Extract split
  split <- if(is.array(f)){
    lapply(seq_len(K), \(x) asub(f, 1:P + (x - 1) * (P - 1), dims = margin, drop = F))
  }else{
    lapply(seq_len(K), \(x) tf$gather(f, indices = as.integer(1:P + (x - 1) * (P - 1) - 1L), axis = axis))
  }
  return(split)
}

# Split warps -------------------------------------------------------------
SplitWarp <- \(gamma, K){
  # Split warps gamma into K periods
  # gamma: warping functions
  # K: number of periods
  
  # To array
  if("numeric" %in% class(gamma)) gamma <- as.array(gamma)
  
  # Extract dimensions
  dims <- dim(gamma)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  margin <- axis + 1
  
  # Split periods
  warp_split <- SplitPeriod(gamma, K)
  
  # Normalize to [0, 1]
  split <- if(is.array(warp_split[[1]])){
    if(length(dims) == 1){
      lapply(warp_split, \(x) (x - min(x)) / (max(x) - min(x)))
    }else{
      dims_split <- dim(warp_split[[1]])
      keepdims <- setdiff(seq_along(dims_split), margin)
      lapply(warp_split, \(x){
        mins <- array(apply(x, keepdims, min), dim = c(dims_split[-margin], 1))
        mins <- aperm(abind(lapply(seq_len(dims_split[margin]), \(x) mins), along = length(dims_split)), order(c(keepdims, margin)))
        maxs <- array(apply(x, keepdims, max), dim = c(dims_split[-margin], 1))
        maxs <- aperm(abind(lapply(seq_len(dims_split[margin]), \(x) maxs), along = length(dims_split)), order(c(keepdims, margin)))
        (x - mins) / (maxs - mins)
      })
    }
  }else{
    lapply(warp_split, \(x) (x - tf$reduce_min(x, axis = axis, keepdims = T)) / (tf$reduce_max(x, axis = axis, keepdims = T) - tf$reduce_min(x, axis = axis, keepdims = T)))
  }
  return(split)
}

# Combine periods ---------------------------------------------------------
CombinePeriods <- \(f){
  # Combine periods from a list
  # f: periods of a function
  
  # Extract number of periods
  K <- length(f)
  
  # To array
  if("numeric" %in% class(f[[1]])) f <- lapply(f, \(x) as.array(x))
  
  # Extract dimensions
  dims <- dim(f[[1]])
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  margin <- axis + 1
  
  # Extract length of a period
  P <- dims[margin]
  
  # Combine periods
  combined <- if(is.array(f[[1]])){
    abind(lapply(seq_len(K + 1), \(x) if(x <= K) asub(f[[x]], 1:(P - 1), dims = margin, drop = F) else asub(f[[K]], P, dims = margin, drop = F)), along = margin)
  }else{
    tf$concat(unlist(list(lapply(seq_len(K), \(x) tf$gather(f[[x]], indices = 0:(P - 2), axis = axis)),
                          tf$expand_dims(tf$gather(f[[K]], indices = P - 1L, axis = axis), axis = axis))), axis = axis)
  }
  return(combined)
}

# Calculate mean orbit ----------------------------------------------------
MeanOrbitPeriod <- \(f, K, keepdims = length(dim(f)) > 1){
  # Calculate mean orbit of functions f
  # f: functions
  # K: number of periods
  
  # To array
  if("numeric" %in% class(f)) f <- as.array(f)
  
  # Extract dimensions
  dims <- dim(f)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  
  # Split periods
  f_split <- if(is.array(f)){
    abind(lapply(SplitPeriod(f, K), \(x) array(x, dim = c(dim(x), 1))), along = length(dims) + 1)
  }else{
    tf$concat(lapply(SplitPeriod(f, K), \(x) tf$expand_dims(x, axis = -1L)), axis = -1L)
  }
  
  # Calculate mean
  f_mean <- if(is.array(f)){
    keep_dims <- setdiff(seq_along(dim(f_split)), if(axis == 0) length(dim(f_split)) else c(1, length(dim(f_split))))
    as.array(apply(f_split, keep_dims, mean))
  }else{
    tf$reduce_mean(f_split, axis = if(axis == 0) -1L else c(0L, -1L))
  }
  if(keepdims) f_mean <- if(is.array(f)) array(f_mean, dim = c(1, dim(f_mean))) else tf$expand_dims(f_mean, axis = 0L)
  return(f_mean)
}

# -------------------------------------------------------------------------
# Karcher mean ------------------------------------------------------------
# -------------------------------------------------------------------------
KarcherMean <- \(gamma, keepdims = length(dim(gamma)) > 1, stopping_criterion = 10 ^ (-7), max_iter = 10 ^ 4, step_size = 0.5, ...){
  # Calculate the mean of warping functions
  # gamma: warping functions
  # keepdims: whether or not to keep first axis of the resulting warping function
  # stopping_criterion: stopping criterion for the norm of mean in the tangent space
  # max_iter: maximum number of iterations
  # step_size: step_size of update of the mean in the tangent space
  
  # Extract dimensions
  dims <- dim(gamma)
  
  # Initialize norm of mean of vectors in the tangent space
  mu_v <- if(is.array(gamma)){
    array(0, dim = c(1, dims[-1]))
  }else{
    tf$constant(0, shape = c(1L, dims[-1]), dtype = tf$float64)
  }
  
  # Calculate the SRSF of the warping functions
  psi <- SRSF(gamma)
  if(!is.array(gamma)) psi <- tf$cast(psi, tf$float64)
  
  # Initialize the mean to the normalized mean of the SRSFs
  if(is.array(gamma)){
    mu_psi <- matrix(colMeans(psi), nrow = 1)
    mu_psi <- mu_psi / sqrt(sum(Trapz(mu_psi ^ 2)))
  }else{
    mu_psi <- tf$reduce_mean(psi, axis = 0L, keepdims = T)
    mu_psi <- mu_psi / sqrt(tf$reduce_sum(Trapz(mu_psi ^ 2), axis = 1L, keepdims = T))
  }
  
  # Calculate Karcher mean of SRSF of warping functions
  iter <- 0
  if(is.array(gamma)){
    while(((sqrt(sum(Trapz(mu_v ^ 2))) >= stopping_criterion) | (iter == 0)) & (iter < max_iter)){
      iter <- iter + 1
      mu_psi <- ExpMap(step_size * mu_v, mu_psi)
      v <- ExpMapInverse(psi, mu_psi)
      mu_v <- matrix(colMeans(v), nrow = 1)
    }
  }else{
    while(((as.numeric(sqrt(sum(Trapz(mu_v ^ 2)))) >= stopping_criterion) | (iter == 0)) & (iter < max_iter)){
      iter <- iter + 1
      mu_psi <- ExpMap(step_size * mu_v, mu_psi)
      v <- ExpMapInverse(psi, mu_psi)
      mu_v <- tf$reduce_mean(v, axis = 0L, keepdims = T)
    }
  }
  
  # Calculate Karcher mean of warping functions
  warp_KM <- SRSFToWarp(mu_psi)
  if(!keepdims){
    warp_KM <- tf$squeeze(warp_KM, axis = 0L)
    if(is.array(gamma)){
      warp_KM <- as.array(warp_KM)
    }
  }
  if(!is.array(gamma)) warp_KM <- tf$cast(warp_KM, gamma$dtype)
  return(warp_KM)
}

# Extract overall Karcher mean --------------------------------------------
ExtractOverallKM <- \(gamma, K, keepdims = length(dim(gamma)) > 1, as.tensor = T, ...){
  # Extract Karcher mean of all observations and periods
  # gamma: warping functions
  # K: number of periods
  # keepdims: whether or not to keep first axis of the resulting warping function
  # as.tensor: whether to calculate the Karcher mean with tensorflow operations
  
  # To array
  if("numeric" %in% class(gamma)) gamma <- as.array(gamma)
  
  # Extract dimensions
  dims <- dim(gamma)
  
  # Extract period warps
  gamma_split <- SplitWarp(gamma, K)
  if(length(dims) == 1) gamma_split <- lapply(gamma_split, \(x) tf$expand_dims(x, axis = 0L))
  gamma_split <- tf$concat(gamma_split, axis = 0L)
  if(!as.tensor) gamma_split <- as.array(gamma_split)
  
  # Calculate Karcher mean
  return(KarcherMean(gamma_split, keepdims = keepdims, ...))
}

# Extract Karcher mean of warps per individual ----------------------------
ExtractIndividualKM <- \(gamma, K, keepdims = length(dim(gamma)) > 1, as.tensor = F, ...){
  # Extract Karcher mean per observation
  # gamma: warping functions
  # K: number of periods
  # keepdims: whether or not to keep first axis of the resulting warping function
  # as.tensor: whether to calculate the Karcher mean with tensorflow operations
  
  # To array
  if("numeric" %in% class(gamma)) gamma <- as.array(gamma)
  
  # Extract dimensions
  dims <- dim(gamma)
  
  # Extract axis
  axis <- if(length(dims) == 1) 0L else 1L
  
  # Extract period warps
  gamma_split <- tf$concat(lapply(SplitWarp(gamma, K), \(x) tf$expand_dims(x, axis = axis)), axis = axis)
  if(length(dims) == 1) gamma_split <- tf$expand_dims(gamma_split, axis = axis)
  if(!as.tensor) gamma_split <- as.array(gamma_split)
  
  # Calculate Karcher mean per observation
  KM <- tf$concat(lapply(seq_len(nrow(gamma_split)), \(x){
    spl <- gamma_split[x,,]
    if(is.null(dim(spl))) spl <- array(spl, dim = c(1, length(spl)))
    KarcherMean(spl, keepdims = T, ...)}), axis = 0L)
  if(!keepdims){
    KM <- tf$squeeze(KM, axis = 0L)
  }
  if(is.array(gamma)){
    KM <- as.array(KM)
  }
  return(KM)
}

# -------------------------------------------------------------------------
# Model -------------------------------------------------------------------
# -------------------------------------------------------------------------
# Convert vector to a warping function ------------------------------------
NNToWarp <- \(y){
  # Transform vector to unit simplex
  # y: vector to be transformed
  
  # Convert to proportions
  z <- tf$clip_by_value(tf$sigmoid(y - log(tf$range(start = ncol(y), limit = 0, delta = -1, dtype = y$dtype))),
                        clip_value_min = tf$keras$backend$epsilon(), clip_value_max = 1 - tf$keras$backend$epsilon())
  
  # Calculate unit simplex
  s <- tf$clip_by_value(tf$math$cumprod(1 - z, axis = 1L, exclusive = T) * z,
                        clip_value_min = tf$keras$backend$epsilon(), clip_value_max = 1 - tf$keras$backend$epsilon())
  
  # Calculate warping function
  gamma <- tf$cumsum(s, axis = 1L)
  max_val <- tf$reduce_max(gamma, axis = 1L, keepdims = T)
  gamma <- tf$where(max_val >= 1,
                    gamma / max_val * (1 - tf$keras$backend$epsilon()),
                    gamma)
  gamma <- tf$concat(list(tf$zeros_like(s[, 1, drop = F]), gamma, tf$ones_like(s[, 1, drop = F])), axis = 1L)
  
  # Extrapolate to have the same number of time points
  return(tfp$math$batch_interp_regular_1d_grid(tf$cast(tf$linspace(0L, 1L, ncol(y)), dtype = y$dtype), 0L, 1L, gamma))
}

# Create outcome ----------------------------------------------------------
CreateOutcome <- \(f, N, dtype = tf$float32){
  # Create outcome data
  # f: outcome
  # N: number of observation
  
  # To array
  if("numeric" %in% class(f)) f <- as.array(f)
  
  # Extract dimensions
  dims <- dim(f)
  
  # Dtype
  f <- tf$cast(f, dtype = dtype)
  
  # Add dimension
  if(length(dims) == 1) f <- tf$expand_dims(f, axis = 0L)
  if(length(dims) <= 2) f <- tf$expand_dims(f, axis = -1L)
  
  # Repeat outcome
  return(tf$`repeat`(f, N, axis = 0L))
}

# Warping model -----------------------------------------------------------
DeepJAM <- \(L, n, filters, kernel, warp_kernel, activation, dtype = tf$float32, ...){
  # Create neural network
  # x: input data
  # filters: number of filters in each hidden layer (vector)
  # kernel: kernel size
  # activation: activation in the hidden layers
  
  # Define input
  if(n == 1){
    input <- layer_input(shape = L, dtype = dtype)
    input_expand <- tf$expand_dims(input, axis = -1L)
  }else{
    input <- layer_input(shape = c(L, n), dtype = dtype)
  }
  
  # Define layers
  layers <- lapply(seq_along(filters), \(x) layer_conv_1d(filters = filters[x], kernel_size = kernel[x], padding = "same", activation = activation, ...))
  output <- layer_flatten(layer_conv_1d(freduce(if(n == 1) input_expand else input, layers), filters = 1, kernel_size = warp_kernel, padding = "same"))
  gamma <- NNToWarp(output)
  warped <- WarpSRSF(input, gamma)
  if(n == 1) warped <- tf$expand_dims(WarpSRSF(input, gamma), axis = -1L)
  
  # Define output
  output <- tf$concat(list(warped, tf$expand_dims(gamma, axis = -1L)), axis = -1L)
  
  # Define model
  model <- keras_model(inputs = input, outputs = output)
  class(model) <- c("DeepJAM", class(model))
  
  # Save configuration
  model$configuration <- list(L = L, n = n, filters = filters, kernel = kernel, warp_kernel = warp_kernel, activation = activation)
  return(model)
}

# Loss function -----------------------------------------------------------
LossWarping <- \(y_true, y_pred){
  # Loss function for warping
  # y_true: template function
  # y_pred: output from neural network
  
  # Separate output
  warped <- y_pred[,, seq_len(dim(y_pred)[3] - 1), drop = F]
  
  # Calculate loss
  return(tf$reduce_mean(tf$reduce_sum(Trapz((y_true - warped) ^ 2), axis = 1L), axis = -1L))
}

# Predict from model ------------------------------------------------------
predict.DeepJAM <- \(object, q, K, ...){
  # Predict from model
  # object: DeepJAM neural network
  # q: input functions
  # K: number of periods
  
  # Predict
  prediction <- object(q)
  
  # Extract
  aligned_SRSF <- prediction[,, seq_len(dim(prediction)[3] - 1)]
  gamma <- prediction[,, dim(prediction)[3]]
  
  # Extract warps
  gamma_KM <- ExtractOverallKM(gamma, K, ...)
  gamma_inv <- ExtendWarp(Inverse(gamma_KM), K)
  
  # Aligned SRSFs
  aligned_SRSF <- WarpSRSF(aligned_SRSF, gamma_inv)
  
  # Combine warps
  gamma <- Warp(gamma, gamma_inv)
  
  if(!("template_orbit" %in% names(object))){
    return(list(aligned_SRSF = aligned_SRSF, gamma = gamma, gamma_KM = gamma_KM))
  }else{
    return(list(aligned_SRSF = aligned_SRSF, template_orbit = object$template_orbit, gamma = gamma, gamma_KM = gamma_KM))
  }
}

# Fit ---------------------------------------------------------------------
fit.DeepJAM <- \(object, K, x, validation_data = NULL, validation_split = 0, epochs = 10, 
                 early_stopping = F, best_model = early_stopping, patience = if(early_stopping) 10 else Inf, early_stopping_criterion = 0.0001, ...){
  # Fit DeepJAM neural network
  # object: DeepJAM neural network
  # x: input functions
  # validation_data: validation data
  # validation_split: validation split
  # epochs: maximum number of epochs
  # early_stopping: whether or not to apply early stopping
  # best_model: whether to return the best weights of the early stopping
  # patience: patience for early stopping
  # early_stopping_criterion: improvement smaller than early_stopping_criterion is considered as no improvement
  
  # Initialize history
  history <- list()
  
  # Initialize warping functions
  prediction <- predict(object, x, K, ...)
  
  # Initialize epoch
  epoch <- 0
  
  # Initialize patience
  pat <- 0
  
  if(early_stopping){
    # Initialize convergence criterion
    criterion <- Inf
    
    # Initialize best loss
    best_loss <- .Machine$double.xmax
  }
  
  # To array
  if("numeric" %in% class(x)) x <- as.array(x)
  
  # Extract dimensions
  dims <- dim(x)
  
  # Add dimension
  if(length(dims) == 1) x <- tf$expand_dims(x, axis = 0L)
  
  # Drop dimension
  if((length(dims) == 3) & (dims[3] == 1)) x <- tf$squeeze(x, axis = 2L)
  
  # Validation data
  if(!is.null(validation_data)){
    # To array
    if("numeric" %in% class(validation_data)) validation_data <- as.array(validation_data)
    
    # Extract dimensions
    dims <- dim(validation_data)
    
    # Add dimension
    if(length(dims) == 1) validation_data <- tf$expand_dims(validation_data, axis = 0L)
    
    # Drop dimension
    if((length(dims) == 3) & (dims[3] == 1)) validation_data <- tf$squeeze(validation_data, axis = 2L)
  }
  
  # Train model
  epochs_print <- if(is.infinite(epochs)) NULL else paste0("/", epochs)
  while((epoch < epochs) & (pat < patience)){
    epoch <- epoch + 1
    # Extract extended template
    q_aligned <- prediction$aligned_SRSF
    
    # Calculate mean orbit
    mu_q_template <- ExtendFunction(MeanOrbitPeriod(q_aligned, K), K)
    
    # Fit 1 epoch
    cat("Epoch", " ", epoch, epochs_print, "\n", sep = "")
    class(object) <- setdiff(class(object), "DeepJAM")
    history[[epoch]] <- object %>% fit(x = x,
                                       y = CreateOutcome(mu_q_template, nrow(x), dtype = object$dtype),
                                       validation_data = if(is.null(validation_data)) NULL else list(validation_data, CreateOutcome(mu_q_template, nrow(validation_data))),
                                       validation_split = validation_split, 
                                       epochs = 1L,
                                       ...)
    class(object) <- c("DeepJAM", class(object))
    
    # Predict
    prediction <- predict(object, x, K, ...)
    
    # Early stooping
    if(early_stopping){
      # Extract validation loss
      val_loss <- history[[epoch]]$metrics$val_loss
      criterion <- (best_loss - val_loss) / best_loss
      if(val_loss < best_loss){
        best_loss <- val_loss
        if(best_model){
          best_weights <- object$get_weights()
          best_template <- mu_q_template
        }
      }
      if(criterion > early_stopping_criterion){
        pat <- 0
      }else{
        pat <- pat + 1
      }
    }
  }
  
  # Transform history
  metrics <- if(is.null(validation_data) & (validation_split == 0)){
    list(loss = unname(unlist(lapply(history, function(x) x$metrics))))
  }else{
    lapply(data.frame(t(sapply(history, function(x) x$metrics))), function(x) unlist(x))
  }
  history <- list(params = history[[1]]$params, metrics = metrics)
  history$params$epochs <- epoch
  class(history) <- "keras_training_history"
  
  # Return best weights
  if(best_model){
    object$set_weights(best_weights)
  }
  
  # Return best template
  object$template_orbit <- if(best_model) tf$squeeze(best_template, axis = 0L) else tf$squeeze(mu_q_template, axis = 0L) 
  
  # Return history
  return(history)
}

# Evaluate model ----------------------------------------------------------
evaluate.DeepJAM <- \(object, q, ...){
  # Evaluate DeepJAM model
  # object: DeepJAM neural network
  # q: functions to evaluate
  
  # Change class
  class(object) <- setdiff(class(object), "DeepJAM")
  
  # Evaluate
  return(evaluate(object, x = q, y = CreateOutcome(tf$expand_dims(object$template_orbit, axis = 0L), nrow(q)), ...))
}

# Extract total warp ------------------------------------------------------
ExtractTotalWarp <- \(object, q, K, ...){
  # Extract total warps
  # object: DeepJAM neural network
  # q: input functions
  # K: number of periods
  
  # Prediction
  prediction <- predict(object, q, K, ...)
  
  # Extract total warp
  return(prediction$gamma)
}

# Extract global and local warp ------------------------------------------
ExtractGlobalLocalWarp <- \(object, q, K, ...){
  # Extract global and local warps
  # object: DeepJAM neural network
  # q: input functions
  # K: number of periods
  
  # Prediction
  prediction <- predict(object, q, K, ...)
  
  # Extract individual KM
  individual_KM <- tf$cast(ExtractIndividualKM(prediction$gamma, K, ...), dtype = object$dtype)
  
  # Inverse of individual KM
  individual_KM_inverse <- ExtendWarp(Inverse(individual_KM), K, ...)
  
  # Global warp
  gamma_global <- Warp(prediction$gamma, individual_KM_inverse)
  return(list(gamma_global = gamma_global, gamma_local = individual_KM))
}

# Save model weights ------------------------------------------------------
SaveDeepJAM <- \(object, file, ...){
  # Save DeepJAM model
  # object: DeepJAM neural network
  # file: path to the file
  # overwrite: whether to overwrite the existing file
  
  # Create folder
  if(!dir.exists(file)) dir.create(file)
  
  # Save model weights
  save_model_weights_hdf5(object, filepath = paste0(file, "/model.h5"), ...)
  
  # Save template for evaluation
  saveRDS(as.array(object$template_orbit), paste0(file, "/template_orbit.RDS"))
  
  # Save configuration
  saveRDS(object$configuration, paste0(file, "/configuration.RDS"))
}

# Load model_weights ------------------------------------------------------
LoadDeepJAM <- \(file, dtype = tf$float32, ...){
  # Load DeepJAM model
  # file: path to the file
  
  # Load configuration
  configuration <- readRDS(paste0(file, "/configuration.RDS"))
  
  # Create model
  for(conf in names(configuration)){
    assign(conf, configuration[[conf]])
  }
  model <- DeepJAM(L = L, n = n, filters = as.numeric(filters), kernel = as.numeric(kernel), warp_kernel = warp_kernel, activation = activation, dtype = dtype, ...)
  load_model_weights_hdf5(model, paste0(file, "/model.h5"))
  
  # Load template
  model$template_orbit <- tf$cast(readRDS(paste0(file, "/template_orbit.RDS")), dtype)
  return(model)
}