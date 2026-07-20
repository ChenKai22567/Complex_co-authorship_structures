# ============================================================
# VALUED ERGM FOR M1 + M_core + M_prod + M_hole + M_full
# Standardized term names aligned with formula:
#
# Pr_theta(Y=y) = 1/kappa(theta) * exp{
#   theta_0  sum
# + theta_1  nonzero
# + theta_2  B(gwdegree | NZ)
# + theta_3  edgecov(highlow | NZ)
# + theta_4  nodecov(pubcount | NZ)
# + theta_5  nodecov(strhole | NZ)
# + theta_6  nodematch(field | NZ)
# + theta_7  nodecov(interdisc | NZ)
# + theta_8  nodecov(strongtie | NZ)
# + theta_9  nodecov(tieduration | SUM)
# + theta_10 nodecov(highcontrib | SUM)
# }
#
# Models kept:
#   M0      helper only: sum + nonzero
#   M1      structural baseline:
#           offset(sum) + offset(nonzero) + B(~gwdegree(...), form="nonzero")
#   M_core  M1 + highlow + field + interdisc
#   M_prod  M1 + highlow + pubcount + field + interdisc + strongtie
#              + tieduration(sum) + highcontrib(sum)
#   M_hole  M1 + highlow + strhole + field + interdisc + strongtie
#              + tieduration(sum) + highcontrib(sum)
#   M_full  M1 + highlow + pubcount + strhole + field + interdisc + strongtie
#              + tieduration(sum) + highcontrib(sum)
#
# Added / revised:
#   1) Standard term naming in code and output tables
#   2) Model-specific collinearity outputs for every model
#   3) Model-specific official nonidentifiability summaries
#   4) Adjustable MCMC diagnostic layout + export format (pdf/png/both)
#   5) Console printing of standardized coefficient tables
# ============================================================

suppressPackageStartupMessages({
  library(xml2)
  library(network)
  library(ergm)
  library(ergm.count)
  library(coda)
})

# ----------------------------
# 0) Global seed
# ----------------------------
SEED <- 42
RNGkind("L'Ecuyer-CMRG")
set.seed(SEED)

# ----------------------------
# 1) User settings
# ----------------------------
get_script_dir <- function(){
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", cmd_args, value = TRUE)
  if(length(file_arg) > 0){
    script_path <- sub("^--file=", "", file_arg[1])
    return(dirname(normalizePath(script_path, winslash = "/", mustWork = TRUE)))
  }

  frame_files <- vapply(
    sys.frames(),
    function(frame) if(is.null(frame$ofile)) NA_character_ else as.character(frame$ofile)[1],
    character(1)
  )
  frame_files <- frame_files[!is.na(frame_files) & nzchar(frame_files)]
  if(length(frame_files) > 0){
    return(dirname(normalizePath(tail(frame_files, 1), winslash = "/", mustWork = TRUE)))
  }

  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
}

script_dir <- get_script_dir()
algorithm_root <- normalizePath(file.path(script_dir, "..", ".."), winslash = "/", mustWork = TRUE)
cat("Algorithm root: ", algorithm_root, "\n", sep = "")

gexf_file <- file.path(
  algorithm_root, "data", "08_ergm", "active_coauthorship_newman_s5_s3_s2_2006_2025.gexf"
)
author_file <- file.path(algorithm_root, "data", "08_ergm", "ergm_author_attributes.csv")

missing_inputs <- c(gexf_file, author_file)[!file.exists(c(gexf_file, author_file))]
if(length(missing_inputs) > 0){
  stop(
    "Missing input file(s) in ", script_dir, ": ",
    paste(missing_inputs, collapse = ", ")
  )
}

out_dir <- file.path(algorithm_root, "results", "08_ergm", "valued_ergm_models")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

out_prefix <- "valued_ergm_standard_terms_seed42"

# parallel
n_cores <- 4
parallel_type <- "PSOCK"

# Newman weight -> integer count
delta <- 0.2
cap_q <- 0.99
disc_method_for_model <- "ceiling"

# endogenous structural term
deg_decay <- 3.0

# ----------------------------
# 1.1) Raw columns in author.csv
# ----------------------------
impact_raw_var      <- "impact_top20_citation_all"  # 0/1, used to build highlow dyadic covariate
pubcount_raw_var    <- "paper"
strhole_raw_var     <- "structure_hole"
field_raw_var       <- "community"
interdisc_raw_var   <- "inter"
strongtie_raw_var   <- "weight_6a"
tieduration_raw_var <- "time_6b"
highcontrib_raw_var <- "single"

# ----------------------------
# 1.2) Standard model term names
# ----------------------------
highlow_name      <- "highlow"       # dyadic matrix from impact_raw_var
pubcount_name     <- "pubcount"      # node covariate in model
strhole_name      <- "strhole"       # node covariate in model
field_name        <- "field"         # nodematch field
interdisc_name    <- "interdisc"     # node covariate in model
strongtie_name    <- "strongtie"     # node covariate in model
tieduration_name  <- "tieduration"   # node covariate in sum layer
highcontrib_name  <- "highcontrib"   # node covariate in sum layer

# ----------------------------
# 1.3) LOG SWITCHES
# ----------------------------
use_log_pubcount     <- TRUE
use_log_strhole      <- TRUE
use_log_interdisc    <- FALSE
use_log_strongtie    <- FALSE
use_log_tieduration  <- FALSE
use_log_highcontrib  <- TRUE

# diagnostics / GOF
save_diag_plots <- TRUE
run_gof         <- TRUE
nsim_gof        <- 200
save_gof_pdf    <- TRUE

# raw ergm summary printing switch
print_raw_ergm_summary <- FALSE

# ----------------------------
# 1.4) MCMC diagnostic plot layout / format
# ----------------------------
# output format: "pdf", "png", or "both"
diag_output_format <- "both"

# official mcmc.diagnostics() output
diag_vars_per_page <- 4
diag_compact       <- 16
diag_width         <- 13
diag_height        <- 10
diag_png_width_px  <- 2200
diag_png_height_px <- 1700
diag_png_res       <- 220

# custom compact trace+density output
save_diag_compact  <- TRUE
compact_ncol       <- 2
compact_row_height <- 2.8
compact_width      <- 12
compact_png_width_px  <- 2200
compact_png_height_px <- 1700
compact_png_res       <- 220

# GOF plot layout
gof_width        <- 12
gof_base_height  <- 6
gof_row_height   <- 0.5
gof_point_cex    <- 0.9
gof_axis_cex     <- 0.8

# ----------------------------
# 2) MCMC settings
# ----------------------------
burnin_base     <- 50000
interval_base   <- 400
samplesize_base <- 15000
ess_base        <- 200
maxit_base      <- 50

burnin_struct     <- 50000
interval_struct   <- 400
samplesize_struct <- 15000
ess_struct        <- 200
maxit_struct      <- 55

burnin_attr     <- 50000
interval_attr   <- 400
samplesize_attr <- 15000
ess_attr        <- 200
maxit_attr      <- 60

density_guard_mult <- 300
maxedges_mult      <- 35

# ----------------------------
# 2.1) Nonidentifiability settings
# ----------------------------
nonident_action <- "warning"  # one of "warning", "message", "error"
nonident_tol    <- 1e-10

# ----------------------------
# 3) Helpers
# ----------------------------
make_w_cnt <- function(weight, delta, method){
  if(method == "round_drop0"){
    as.integer(round(weight / delta))
  } else if(method == "ceiling"){
    as.integer(ceiling(weight / delta))
  } else if(method == "round_pmax1"){
    pmax(1L, as.integer(round(weight / delta)))
  } else {
    stop("Unknown discretization method.")
  }
}

z_std <- function(x){
  x <- as.numeric(x)
  s <- sd(x, na.rm = TRUE)
  if(is.na(s) || s == 0) return(rep(0, length(x)))
  as.numeric((x - mean(x, na.rm = TRUE)) / s)
}

transform_variable <- function(x, use_log){
  x_num <- as.numeric(x)
  trans <- if(use_log) log1p(x_num) else x_num
  list(raw = x_num, trans = trans, z = z_std(trans))
}

p_to_stars <- function(p){
  ifelse(
    is.na(p), "",
    ifelse(p < 0.001, "***",
           ifelse(p < 0.01, "**",
                  ifelse(p < 0.05, "*",
                         ifelse(p < 0.1, ".", ""))))
  )
}

summary_row <- function(x, var_name){
  x_num <- as.numeric(x)
  data.frame(
    variable = var_name,
    mean = mean(x_num, na.rm = TRUE),
    sd = sd(x_num, na.rm = TRUE),
    min = min(x_num, na.rm = TRUE),
    q25 = as.numeric(quantile(x_num, 0.25, na.rm = TRUE)),
    median = median(x_num, na.rm = TRUE),
    q75 = as.numeric(quantile(x_num, 0.75, na.rm = TRUE)),
    max = max(x_num, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

analytic_init_sum_nonzero <- function(M, P0_obs, mean_y_obs){
  f_logb <- function(logb){
    b <- exp(logb)
    num <- M * b * (1 + b)^(M - 1) * (1 - P0_obs)
    den <- (1 + b)^M - 1
    num / den - mean_y_obs
  }

  root <- tryCatch(
    uniroot(f_logb, lower = -12, upper = 1),
    error = function(e) uniroot(f_logb, lower = -20, upper = 2)
  )

  theta_sum <- root$root
  b <- exp(theta_sum)
  a <- (1 / P0_obs - 1) / ((1 + b)^M - 1)
  theta_nz <- log(a)

  c(sum = theta_sum, nonzero = theta_nz)
}

extract_coef_value <- function(fit, keyword, default = 0){
  if(is.null(fit)) return(default)
  nm <- names(coef(fit))
  hit <- grep(keyword, nm, value = TRUE)
  if(length(hit) < 1) return(default)
  unname(coef(fit)[hit[1]])
}

# ----------------------------
# 3.1) Standard term dictionary / labels
# ----------------------------
term_dictionary <- data.frame(
  theta = c("theta_0","theta_1","theta_2","theta_3","theta_4","theta_5","theta_6","theta_7","theta_8","theta_9","theta_10"),
  standard_term = c(
    "sum",
    "nonzero",
    "gwdegree",
    "highlow",
    "pubcount",
    "strhole",
    "field",
    "interdisc",
    "strongtie",
    "tieduration",
    "highcontrib"
  ),
  formula_term = c(
    "sum",
    "nonzero",
    "B(gwdegree | NZ)",
    "edgecov(highlow | NZ)",
    "nodecov(pubcount | NZ)",
    "nodecov(strhole | NZ)",
    "nodematch(field | NZ)",
    "nodecov(interdisc | NZ)",
    "nodecov(strongtie | NZ)",
    "nodecov(tieduration | SUM)",
    "nodecov(highcontrib | SUM)"
  ),
  source_column = c(
    NA, NA, NA,
    impact_raw_var,
    pubcount_raw_var,
    strhole_raw_var,
    field_raw_var,
    interdisc_raw_var,
    strongtie_raw_var,
    tieduration_raw_var,
    highcontrib_raw_var
  ),
  model_object_name = c(
    "sum", "nonzero", "gwdegree",
    highlow_name,
    pubcount_name,
    strhole_name,
    field_name,
    interdisc_name,
    strongtie_name,
    tieduration_name,
    highcontrib_name
  ),
  stringsAsFactors = FALSE
)

map_term_to_theta <- function(term){
  term <- trimws(term)

  if(term %in% c("sum", "offset(sum)")) return("theta_0")
  if(term %in% c("nonzero", "offset(nonzero)")) return("theta_1")

  if(grepl("B\\(nonzero\\)~gwdeg", term) || grepl("gwdeg.fixed", term) || grepl("gwdegree", term)){
    return("theta_2")
  }

  if(grepl("^edgecov", term) || grepl("edgecov\\.nonzero", term)){
    return("theta_3")
  }

  if(grepl("^nodecov", term) && grepl("pubcount", term)){
    return("theta_4")
  }

  if(grepl("^nodecov", term) && grepl("strhole", term)){
    return("theta_5")
  }

  if(grepl("^nodematch", term) && grepl("field", term)){
    return("theta_6")
  }

  if(grepl("^nodecov", term) && grepl("interdisc", term)){
    return("theta_7")
  }

  if(grepl("^nodecov", term) && grepl("strongtie", term)){
    return("theta_8")
  }

  if(grepl("^nodecov", term) && grepl("tieduration", term)){
    return("theta_9")
  }

  if(grepl("^nodecov", term) && grepl("highcontrib", term)){
    return("theta_10")
  }

  NA_character_
}

map_term_to_standard <- function(term){
  term <- trimws(term)

  if(term %in% c("sum", "offset(sum)")) return("sum")
  if(term %in% c("nonzero", "offset(nonzero)")) return("nonzero")

  if(grepl("B\\(nonzero\\)~gwdeg", term) || grepl("gwdeg.fixed", term) || grepl("gwdegree", term)){
    return("gwdegree")
  }

  if(grepl("^edgecov", term) || grepl("edgecov\\.nonzero", term)){
    return("highlow")
  }

  if(grepl("^nodecov", term) && grepl("pubcount", term)){
    return("pubcount")
  }

  if(grepl("^nodecov", term) && grepl("strhole", term)){
    return("strhole")
  }

  if(grepl("^nodematch", term) && grepl("field", term)){
    return("field")
  }

  if(grepl("^nodecov", term) && grepl("interdisc", term)){
    return("interdisc")
  }

  if(grepl("^nodecov", term) && grepl("strongtie", term)){
    return("strongtie")
  }

  if(grepl("^nodecov", term) && grepl("tieduration", term)){
    return("tieduration")
  }

  if(grepl("^nodecov", term) && grepl("highcontrib", term)){
    return("highcontrib")
  }

  term
}

make_coef_df <- function(fit){
  cm <- summary(fit)$coef
  est <- as.numeric(cm[, 1])
  se  <- as.numeric(cm[, 2])
  mcmc_pct <- as.numeric(cm[, 3])
  z   <- as.numeric(cm[, 4])
  p   <- as.numeric(cm[, 5])

  term_raw <- rownames(cm)
  term_std <- vapply(term_raw, map_term_to_standard, character(1))
  theta_id <- vapply(term_raw, map_term_to_theta, character(1))

  data.frame(
    term = term_std,          # 标准展示名称
    term_raw = term_raw,      # 保留原始 ergm 名称
    theta = theta_id,
    estimate = est,
    std_error = se,
    ci95_low = est - 1.96 * se,
    ci95_high = est + 1.96 * se,
    mcmc_percent = mcmc_pct,
    z_value = z,
    abs_z = abs(z),
    p_value = p,
    p_stars = p_to_stars(p),
    is_offset = grepl("^offset\\(", term_raw),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}

write_coef_table <- function(fit, path){
  out <- make_coef_df(fit)
  write.csv(out, file = path, row.names = FALSE)
  invisible(out)
}

print_coef_console <- function(fit, model_name){
  if(is.null(fit)) return(invisible(NULL))

  cat("\n====================================\n")
  cat("Coefficient table (standard names):", model_name, "\n")
  cat("====================================\n")

  df <- make_coef_df(fit)
  print(
    df[, c("theta","term","estimate","std_error","z_value","p_value","p_stars","term_raw")],
    row.names = FALSE
  )
}

safe_loglik_ic <- function(fit, D_dyads, model_name){
  if(is.null(fit)){
    return(data.frame(
      model = model_name,
      k_total = NA,
      k_free = NA,
      logLik = NA,
      AIC_manual = NA,
      BIC_dyads = NA,
      AIC_R = NA,
      BIC_R = NA,
      stringsAsFactors = FALSE
    ))
  }

  nm <- names(coef(fit))
  k_total <- length(nm)
  k_free  <- sum(!grepl("^offset\\(", nm))

  ll <- tryCatch(as.numeric(logLik(fit)), error = function(e) NA_real_)
  aic_manual <- if(is.finite(ll)) -2 * ll + 2 * k_free else NA_real_
  bic_dyads  <- if(is.finite(ll)) -2 * ll + log(D_dyads) * k_free else NA_real_

  aic_R <- tryCatch(as.numeric(AIC(fit)), error = function(e) NA_real_)
  bic_R <- tryCatch(as.numeric(BIC(fit)), error = function(e) NA_real_)

  data.frame(
    model = model_name,
    k_total = k_total,
    k_free = k_free,
    logLik = ll,
    AIC_manual = aic_manual,
    BIC_dyads = bic_dyads,
    AIC_R = aic_R,
    BIC_R = bic_R,
    stringsAsFactors = FALSE
  )
}

build_model_term_long <- function(fit, model_name, model_order, D_dyads){
  if(is.null(fit)) return(NULL)
  coef_df <- make_coef_df(fit)
  fit_df  <- safe_loglik_ic(fit, D_dyads, model_name)

  out <- coef_df
  out$model <- model_name
  out$model_order <- model_order
  out$k_total <- fit_df$k_total[1]
  out$k_free <- fit_df$k_free[1]
  out$logLik <- fit_df$logLik[1]
  out$AIC_manual <- fit_df$AIC_manual[1]
  out$BIC_dyads <- fit_df$BIC_dyads[1]
  out$AIC_R <- fit_df$AIC_R[1]
  out$BIC_R <- fit_df$BIC_R[1]

  out <- out[, c(
    "model","model_order","theta","term","term_raw","estimate","std_error",
    "ci95_low","ci95_high","mcmc_percent","z_value","abs_z","p_value","p_stars",
    "is_offset","k_total","k_free","logLik","AIC_manual","BIC_dyads","AIC_R","BIC_R"
  )]
  out
}

make_wide_table <- function(long_df, value_col, row_label_col = "term"){
  if(is.null(long_df) || nrow(long_df) == 0) return(NULL)
  rows <- unique(long_df[[row_label_col]])
  model_levels <- unique(long_df$model)
  out <- data.frame(row_label = rows, stringsAsFactors = FALSE)
  names(out)[1] <- row_label_col

  for(m in model_levels){
    sub <- long_df[long_df$model == m, c(row_label_col, value_col), drop = FALSE]
    vec <- setNames(sub[[value_col]], sub[[row_label_col]])
    out[[m]] <- unname(vec[out[[row_label_col]]])
  }
  out
}

summary_formula_on_network <- function(formula_obj, nw){
  f <- formula_obj
  environment(f) <- list2env(
    list(nw = nw),
    parent = environment(formula_obj)
  )
  summary(f, response = "w")
}

# ----------------------------
# 3.2) Approximate collinearity diagnostics
# ----------------------------
make_vif_table <- function(df, vars, model_name){
  if(length(vars) == 0){
    return(data.frame(
      model = model_name,
      variable = character(0),
      r2_on_others = numeric(0),
      vif = numeric(0),
      stringsAsFactors = FALSE
    ))
  }

  X <- df[, vars, drop = FALSE]
  X[] <- lapply(X, as.numeric)

  keep <- vapply(X, function(z) {
    s <- sd(z, na.rm = TRUE)
    !(is.na(s) || s == 0)
  }, logical(1))
  X <- X[, keep, drop = FALSE]

  if(ncol(X) == 0){
    return(data.frame(
      model = model_name,
      variable = character(0),
      r2_on_others = numeric(0),
      vif = numeric(0),
      stringsAsFactors = FALSE
    ))
  }

  out_list <- vector("list", ncol(X))
  for(i in seq_len(ncol(X))){
    y_name <- names(X)[i]
    y <- X[[i]]
    others <- X[, setdiff(names(X), y_name), drop = FALSE]

    if(ncol(others) == 0){
      r2 <- 0
      vif <- 1
    } else {
      dat <- data.frame(y = y, others, check.names = FALSE)
      fit <- lm(y ~ ., data = dat)
      r2 <- summary(fit)$r.squared
      if(is.na(r2) || r2 >= 1){
        vif <- Inf
      } else {
        vif <- 1 / (1 - r2)
      }
    }

    out_list[[i]] <- data.frame(
      model = model_name,
      variable = y_name,
      r2_on_others = r2,
      vif = vif,
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, out_list)
}

write_model_collinearity_outputs <- function(df, vars, model_name, out_dir, out_prefix){
  summary_path <- file.path(out_dir, paste0(out_prefix, "_table_collinearity_summary_", model_name, ".csv"))
  corr_path    <- file.path(out_dir, paste0(out_prefix, "_table_corr_", model_name, ".csv"))
  vif_path     <- file.path(out_dir, paste0(out_prefix, "_table_vif_", model_name, ".csv"))

  if(length(vars) == 0){
    summary_tab <- data.frame(
      model = model_name,
      n_continuous_covariates = 0,
      max_abs_pairwise_corr = NA_real_,
      any_abs_corr_ge_0_70 = NA,
      max_vif = NA_real_,
      any_vif_ge_5 = NA,
      scope = "No continuous node covariates in this model; see official nonident summary for ergm-level diagnostics.",
      stringsAsFactors = FALSE
    )
    write.csv(summary_tab, summary_path, row.names = FALSE)
    return(invisible(list(summary = summary_tab, cor = NULL, vif = NULL)))
  }

  X <- df[, vars, drop = FALSE]
  X[] <- lapply(X, as.numeric)

  cor_mat <- cor(as.matrix(X), use = "pairwise.complete.obs")
  if(is.null(dim(cor_mat))){
    cor_mat <- matrix(cor_mat, nrow = 1, ncol = 1,
                      dimnames = list(vars[1], vars[1]))
  }
  write.csv(cor_mat, corr_path, row.names = TRUE)

  vif_tab <- make_vif_table(df, vars, model_name)
  write.csv(vif_tab, vif_path, row.names = FALSE)

  pair_vals <- if(ncol(cor_mat) >= 2) abs(cor_mat[upper.tri(cor_mat)]) else numeric(0)
  max_abs_corr <- if(length(pair_vals) > 0) max(pair_vals, na.rm = TRUE) else NA_real_
  max_vif <- if(nrow(vif_tab) > 0) {
    finite_vif <- vif_tab$vif[is.finite(vif_tab$vif)]
    if(length(finite_vif) > 0) max(finite_vif, na.rm = TRUE) else Inf
  } else NA_real_

  summary_tab <- data.frame(
    model = model_name,
    n_continuous_covariates = ncol(X),
    max_abs_pairwise_corr = max_abs_corr,
    any_abs_corr_ge_0_70 = if(is.na(max_abs_corr)) NA else max_abs_corr >= 0.70,
    max_vif = max_vif,
    any_vif_ge_5 = if(is.na(max_vif) || is.infinite(max_vif)) NA else max_vif >= 5,
    scope = "Continuous node covariates only; categorical nodematch(field) and dyadic edgecov(highlow) are not included here.",
    stringsAsFactors = FALSE
  )

  if(ncol(X) == 1){
    summary_tab$scope <- "Only one continuous node covariate; pairwise correlation is not applicable. VIF=1 by construction."
  }

  write.csv(summary_tab, summary_path, row.names = FALSE)
  invisible(list(summary = summary_tab, cor = cor_mat, vif = vif_tab))
}

print_collinearity_console <- function(model_name, coll_obj){
  cat("\n====================================\n")
  cat("Collinearity diagnostics:", model_name, "\n")
  cat("====================================\n")

  if(!is.null(coll_obj$summary)){
    cat("\n[Summary]\n")
    print(coll_obj$summary, row.names = FALSE)
  }

  if(!is.null(coll_obj$cor)){
    cat("\n[Correlation matrix]\n")
    print(round(coll_obj$cor, 4))
  } else {
    cat("\n[Correlation matrix]\nNot applicable.\n")
  }

  if(!is.null(coll_obj$vif) && nrow(coll_obj$vif) > 0){
    cat("\n[VIF table]\n")
    print(coll_obj$vif, row.names = FALSE)
  } else {
    cat("\n[VIF table]\nNot applicable.\n")
  }
}

# ----------------------------
# 3.3) Warning / nonident summary
# ----------------------------
write_warning_log <- function(model_name, warnings, out_dir, out_prefix){
  path <- file.path(out_dir, paste0(out_prefix, "_warnings_", model_name, ".txt"))
  if(length(warnings) == 0){
    writeLines("No warnings captured.", con = path)
  } else {
    writeLines(unique(warnings), con = path)
  }
}

write_nonident_summary <- function(model_name, warnings, out_dir, out_prefix){
  txt <- if(length(warnings) == 0) "No warnings captured." else paste(unique(warnings), collapse = " || ")
  summary_tab <- data.frame(
    model = model_name,
    warning_count = length(unique(warnings)),
    has_any_warning = length(warnings) > 0,
    has_nonident_warning = any(grepl("nonident|linearly dependent|redundant|multicol", warnings, ignore.case = TRUE)),
    has_nonvar_warning = any(grepl("nonvar|constant", warnings, ignore.case = TRUE)),
    warning_text = txt,
    stringsAsFactors = FALSE
  )
  write.csv(
    summary_tab,
    file = file.path(out_dir, paste0(out_prefix, "_table_nonident_", model_name, ".csv")),
    row.names = FALSE
  )
  invisible(summary_tab)
}

print_nonident_console <- function(model_name, warnings){
  cat("\n------------------------------------\n")
  cat("Official ergm warning / nonident summary:", model_name, "\n")
  cat("------------------------------------\n")

  if(length(warnings) == 0){
    cat("No warnings captured.\n")
  } else {
    cat(paste(unique(warnings), collapse = "\n"), "\n")
  }
}

run_ergm_safe <- function(model_name, formula_obj, response_name, reference_obj,
                          offset_vec = NULL, control_obj, out_dir, out_prefix,
                          eval.loglik = TRUE, verbose = TRUE){
  warn <- character(0)

  fit <- withCallingHandlers(
    tryCatch(
      ergm(
        formula_obj,
        response = response_name,
        reference = reference_obj,
        offset.coef = offset_vec,
        control = control_obj,
        eval.loglik = eval.loglik,
        verbose = verbose
      ),
      error = function(e){
        warn <<- c(warn, paste0("ERROR: ", conditionMessage(e)))
        NULL
      }
    ),
    warning = function(w){
      warn <<- c(warn, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )

  write_warning_log(model_name, warn, out_dir, out_prefix)
  write_nonident_summary(model_name, warn, out_dir, out_prefix)
  print_nonident_console(model_name, warn)

  list(fit = fit, warnings = warn)
}

# ----------------------------
# 3.4) MCMC diagnostics export
# ----------------------------
open_device <- function(path_no_ext, format = c("pdf","png"), width = 12, height = 8,
                        png_width_px = 2000, png_height_px = 1500, png_res = 220){
  format <- match.arg(format)
  if(format == "pdf"){
    pdf(paste0(path_no_ext, ".pdf"), width = width, height = height)
  } else if(format == "png"){
    png(paste0(path_no_ext, ".png"), width = png_width_px, height = png_height_px, res = png_res)
  }
}

close_plot_device <- function(device_id){
  devices <- dev.list()
  if(!is.na(device_id) && !is.null(devices) && device_id %in% devices){
    dev.off(device_id)
  }
}

close_new_plot_devices <- function(devices_before){
  current_devices <- unname(dev.list())
  extra_devices <- setdiff(current_devices, unname(devices_before))
  for(device_id in rev(extra_devices)) close_plot_device(device_id)
}

save_mcmc_diag_official <- function(fit, file_stem){
  if(!save_diag_plots || is.null(fit)) return(invisible(NULL))
  devices_before <- dev.list()
  on.exit(close_new_plot_devices(devices_before), add = TRUE)

  formats <- switch(
    diag_output_format,
    pdf  = "pdf",
    png  = "png",
    both = c("pdf", "png"),
    "pdf"
  )

  for(fmt in formats){
    try({
      open_device(
        file_stem,
        format = fmt,
        width = diag_width,
        height = diag_height,
        png_width_px = diag_png_width_px,
        png_height_px = diag_png_height_px,
        png_res = diag_png_res
      )
      mcmc.diagnostics(
        fit,
        vars.per.page = diag_vars_per_page,
        compact = diag_compact,
        which = c("plots", "summary", "autocorrelation", "crosscorrelation", "burnin")
      )
      dev.off()
    }, silent = TRUE)
  }
}

save_mcmc_diag_compact <- function(fit, file_stem){
  if(!save_diag_compact || is.null(fit)) return(invisible(NULL))
  devices_before <- dev.list()
  on.exit(close_new_plot_devices(devices_before), add = TRUE)

  samp <- NULL
  if(!is.null(fit$sample)){
    samp <- fit$sample
  } else if(!is.null(fit$sample.obs)){
    samp <- fit$sample.obs
  }
  if(is.null(samp)) return(invisible(NULL))

  samp_mat <- as.matrix(samp)
  if(is.null(samp_mat) || ncol(samp_mat) == 0) return(invisible(NULL))

  n_var <- ncol(samp_mat)
  nrow_page <- ceiling(n_var / compact_ncol)
  page_height <- max(7, nrow_page * compact_row_height)

  formats <- switch(
    diag_output_format,
    pdf  = "pdf",
    png  = "png",
    both = c("pdf", "png"),
    "pdf"
  )

  for(fmt in formats){
    try({
      open_device(
        file_stem,
        format = fmt,
        width = compact_width,
        height = page_height,
        png_width_px = compact_png_width_px,
        png_height_px = max(compact_png_height_px, as.integer(page_height * compact_png_res)),
        png_res = compact_png_res
      )
      par(
        mfrow = c(nrow_page, compact_ncol),
        mar = c(3.2, 3.2, 2.2, 1.2),
        oma = c(0.5, 0.5, 1, 0.5),
        mgp = c(1.8, 0.5, 0),
        tcl = -0.25
      )

      for(j in seq_len(n_var)){
        x <- samp_mat[, j]
        ts.plot(
          x,
          xlab = "Iteration",
          ylab = "Value",
          main = paste0("Trace: ", colnames(samp_mat)[j])
        )
      }

      missing_panels <- nrow_page * compact_ncol - n_var
      if(missing_panels > 0){
        for(k in seq_len(missing_panels)){
          plot.new()
        }
      }

      dev.off()
    }, silent = TRUE)
  }

  for(fmt in formats){
    try({
      open_device(
        paste0(file_stem, "_density"),
        format = fmt,
        width = compact_width,
        height = page_height,
        png_width_px = compact_png_width_px,
        png_height_px = max(compact_png_height_px, as.integer(page_height * compact_png_res)),
        png_res = compact_png_res
      )
      par(
        mfrow = c(nrow_page, compact_ncol),
        mar = c(3.2, 3.2, 2.2, 1.2),
        oma = c(0.5, 0.5, 1, 0.5),
        mgp = c(1.8, 0.5, 0),
        tcl = -0.25
      )

      for(j in seq_len(n_var)){
        x <- samp_mat[, j]
        d <- try(density(x, na.rm = TRUE), silent = TRUE)
        if(inherits(d, "try-error")){
          plot.new()
          title(main = paste0("Density: ", colnames(samp_mat)[j]))
          text(0.5, 0.5, "Density unavailable")
        } else {
          plot(d,
               main = paste0("Density: ", colnames(samp_mat)[j]),
               xlab = "Value",
               ylab = "Density")
        }
      }

      missing_panels <- nrow_page * compact_ncol - n_var
      if(missing_panels > 0){
        for(k in seq_len(missing_panels)){
          plot.new()
        }
      }

      dev.off()
    }, silent = TRUE)
  }
}

save_mcmc_outputs <- function(fit, model_name, out_dir, out_prefix){
  if(is.null(fit)) return(invisible(NULL))
  save_mcmc_diag_official(
    fit,
    file_stem = file.path(out_dir, paste0(out_prefix, "_diag_official_", model_name))
  )
  save_mcmc_diag_compact(
    fit,
    file_stem = file.path(out_dir, paste0(out_prefix, "_diag_compact_", model_name))
  )
}

# ----------------------------
# 3.5) GOF helpers
# ----------------------------
make_gof_table <- function(fit, obs_network, stat_fun, nsim, seed){
  set.seed(seed)

  obs <- stat_fun(obs_network)
  sims <- simulate(fit, nsim = nsim, output = "network", response = "w")

  sim_stats <- t(vapply(
    sims,
    function(nw) as.numeric(stat_fun(nw)),
    FUN.VALUE = as.numeric(obs)
  ))
  colnames(sim_stats) <- names(obs)

  ci <- apply(sim_stats, 2, quantile, probs = c(0.025, 0.5, 0.975), na.rm = TRUE)
  inside <- (obs >= ci[1, ]) & (obs <= ci[3, ])

  gof_tab <- data.frame(
    term = names(obs),
    obs = as.numeric(obs),
    sim_q025 = as.numeric(ci[1, ]),
    sim_median = as.numeric(ci[2, ]),
    sim_q975 = as.numeric(ci[3, ]),
    inside_95 = as.integer(inside),
    row.names = NULL,
    stringsAsFactors = FALSE
  )

  list(table = gof_tab, sim_stats = sim_stats)
}

save_gof_plot_pdf <- function(gof_table, filename, plot_title){
  if(!save_gof_pdf || is.null(gof_table) || nrow(gof_table) == 0) return(invisible(NULL))
  devices_before <- dev.list()
  on.exit(close_new_plot_devices(devices_before), add = TRUE)

  try({
    n_term <- nrow(gof_table)
    left_margin <- max(10, min(24, 8 + max(nchar(gof_table$term)) * 0.18))

    pdf(filename, width = gof_width, height = max(gof_base_height, gof_row_height * n_term + 2))
    par(mar = c(5, left_margin, 4, 2))

    idx <- seq_len(n_term)
    xlim <- range(c(gof_table$sim_q025, gof_table$sim_q975, gof_table$obs), na.rm = TRUE)

    plot(
      gof_table$sim_median, idx,
      xlim = xlim,
      ylim = c(0.5, n_term + 0.5),
      yaxt = "n",
      ylab = "",
      xlab = "Statistic value",
      main = plot_title,
      pch = 1,
      cex = gof_point_cex
    )
    segments(gof_table$sim_q025, idx, gof_table$sim_q975, idx, lwd = 2)
    points(gof_table$obs, idx, pch = 16, cex = gof_point_cex)
    axis(2, at = idx, labels = gof_table$term, las = 2, cex.axis = gof_axis_cex)
    legend(
      "topright",
      legend = c("Observed", "Sim median", "95% sim interval"),
      pch = c(16, 1, NA),
      lty = c(NA, NA, 1),
      bty = "n"
    )

    dev.off()
  }, silent = TRUE)
}

save_gof_outputs <- function(model_name, fit, obs_network, stat_fun, nsim, seed, out_dir, out_prefix){
  if(is.null(fit)) return(NULL)

  cat("\n===== GOF:", model_name, "=====\n")
  gof_obj <- make_gof_table(
    fit = fit,
    obs_network = obs_network,
    stat_fun = stat_fun,
    nsim = nsim,
    seed = seed
  )

  write.csv(
    gof_obj$table,
    file = file.path(out_dir, paste0(out_prefix, "_table_gof_", model_name, ".csv")),
    row.names = FALSE
  )
  write.csv(
    gof_obj$sim_stats,
    file = file.path(out_dir, paste0(out_prefix, "_gof_", model_name, "_sim_stats_raw.csv")),
    row.names = FALSE
  )
  save_gof_plot_pdf(
    gof_table = gof_obj$table,
    filename = file.path(out_dir, paste0(out_prefix, "_gofplot_", model_name, ".pdf")),
    plot_title = paste("GOF -", model_name)
  )

  gof_long <- gof_obj$table
  gof_long$model <- model_name
  gof_long <- gof_long[, c("model", "term", "obs", "sim_q025", "sim_median", "sim_q975", "inside_95")]

  list(
    table = gof_obj$table,
    sim_stats = gof_obj$sim_stats,
    long = gof_long
  )
}

# ----------------------------
# 3.6) Formula builder
# ----------------------------
build_formula <- function(lhs = "net",
                          include_highlow = FALSE,
                          include_pubcount = FALSE,
                          include_strhole = FALSE,
                          include_field = FALSE,
                          include_interdisc = FALSE,
                          include_strongtie = FALSE,
                          include_tieduration = FALSE,
                          include_highcontrib = FALSE){

  terms <- c(
    "offset(sum)",
    "offset(nonzero)",
    sprintf('B(~gwdegree(decay = %s, fixed = TRUE), form = "nonzero")',
            format(deg_decay, scientific = FALSE, trim = TRUE))
  )

  if(include_highlow){
    terms <- c(terms, sprintf('edgecov(%s, form = "nonzero")', highlow_name))
  }
  if(include_pubcount){
    terms <- c(terms, sprintf('nodecov("%s", form = "nonzero")', pubcount_name))
  }
  if(include_strhole){
    terms <- c(terms, sprintf('nodecov("%s", form = "nonzero")', strhole_name))
  }
  if(include_field){
    terms <- c(terms, sprintf('nodematch("%s", diff = FALSE, form = "nonzero")', field_name))
  }
  if(include_interdisc){
    terms <- c(terms, sprintf('nodecov("%s", form = "nonzero")', interdisc_name))
  }
  if(include_strongtie){
    terms <- c(terms, sprintf('nodecov("%s", form = "nonzero")', strongtie_name))
  }
  if(include_tieduration){
    terms <- c(terms, sprintf('nodecov("%s", form = "sum")', tieduration_name))
  }
  if(include_highcontrib){
    terms <- c(terms, sprintf('nodecov("%s", form = "sum")', highcontrib_name))
  }

  as.formula(paste(lhs, "~", paste(terms, collapse = " + ")))
}

make_control <- function(init_vec, burnin, interval, samplesize, ess, maxit, m_obs, maxedges_cap){
  control.ergm(
    init = unname(init_vec),
    init.method = "zeros",
    main.method = "MCMLE",
    MCMC.burnin = burnin,
    MCMC.interval = interval,
    MCMC.samplesize = samplesize,
    MCMC.effectiveSize = ess,
    MCMLE.maxit = maxit,
    parallel = n_cores,
    parallel.type = parallel_type,
    seed = SEED,
    MCMLE.density.guard = density_guard_mult * m_obs,
    MCMC.maxedges = maxedges_cap,
    MPLE.nonident = nonident_action,
    MPLE.nonident.tol = nonident_tol,
    MCMLE.nonident = nonident_action,
    MCMLE.nonident.tol = nonident_tol,
    MPLE.nonvar = "warning",
    MCMLE.nonvar = "warning"
  )
}

ctrl_base <- function(init_vec, m_obs, maxedges_cap){
  make_control(init_vec, burnin_base, interval_base, samplesize_base, ess_base, maxit_base, m_obs, maxedges_cap)
}
ctrl_struct <- function(init_vec, m_obs, maxedges_cap){
  make_control(init_vec, burnin_struct, interval_struct, samplesize_struct, ess_struct, maxit_struct, m_obs, maxedges_cap)
}
ctrl_attr <- function(init_vec, m_obs, maxedges_cap){
  make_control(init_vec, burnin_attr, interval_attr, samplesize_attr, ess_attr, maxit_attr, m_obs, maxedges_cap)
}

# ----------------------------
# 4) Read GEXF -> undirected collapse
# ----------------------------
doc <- read_xml(gexf_file)
doc <- xml_ns_strip(doc)

edges_xml <- xml_find_all(doc, "//edge")
nodes_xml <- xml_find_all(doc, "//node")

el <- data.frame(
  source = xml_attr(edges_xml, "source"),
  target = xml_attr(edges_xml, "target"),
  weight = as.numeric(xml_attr(edges_xml, "weight")),
  stringsAsFactors = FALSE
)

nd <- data.frame(
  id    = xml_attr(nodes_xml, "id"),
  label = xml_attr(nodes_xml, "label"),
  stringsAsFactors = FALSE
)
nd$label[is.na(nd$label) | nd$label == ""] <- nd$id[is.na(nd$label) | nd$label == ""]

if(any(is.na(nd$id) | nd$id == "")) stop("GEXF contains a node with a missing id.")
if(anyDuplicated(nd$id)) stop("GEXF contains duplicate node ids.")
if(anyDuplicated(nd$label)) stop("GEXF contains duplicate node labels; author matching would be ambiguous.")

cat("Parsed:", nrow(nd), "nodes,", nrow(el), "edges\n")

el$source <- as.character(el$source)
el$target <- as.character(el$target)
el <- el[!is.na(el$weight) & el$source != el$target, , drop = FALSE]

# GEXF edges refer to node ids, while author.csv is matched to node labels.
# Convert endpoints explicitly so the script remains correct when id != label.
label_by_id <- setNames(as.character(nd$label), as.character(nd$id))
missing_endpoint_ids <- setdiff(unique(c(el$source, el$target)), names(label_by_id))
if(length(missing_endpoint_ids) > 0){
  stop("GEXF edge endpoint id(s) not found in node list: ", paste(head(missing_endpoint_ids, 10), collapse = ", "))
}
el$source <- unname(label_by_id[el$source])
el$target <- unname(label_by_id[el$target])

from <- pmin(el$source, el$target)
to   <- pmax(el$source, el$target)
el_u <- data.frame(from = from, to = to, weight = el$weight, stringsAsFactors = FALSE)
el_u <- aggregate(weight ~ from + to, data = el_u, FUN = sum)

vnames <- as.character(nd$label)
n <- length(vnames)

write.csv(
  el_u,
  file = file.path(out_dir, paste0(out_prefix, "_edgelist_raw_undirected.csv")),
  row.names = FALSE
)

# ----------------------------
# 5) Discretize + cap => M
# ----------------------------
el_u$w_cnt <- make_w_cnt(el_u$weight, delta, disc_method_for_model)
el_m <- el_u[el_u$w_cnt > 0, , drop = FALSE]
if(nrow(el_m) == 0) stop("No w_cnt > 0 edges after discretization.")

M <- as.integer(quantile(el_m$w_cnt, cap_q, na.rm = TRUE))
if(M < 1L) M <- max(el_m$w_cnt, na.rm = TRUE)
el_m$w_cnt <- pmin(el_m$w_cnt, M)

cat("Modeling discretization:\n")
cat("  method =", disc_method_for_model, "\n")
cat("  delta  =", delta, " cap_q =", cap_q, " trials(M) =", M, "\n")
print(summary(el_m$w_cnt))

write.csv(
  el_m[, c("from","to","weight","w_cnt")],
  file = file.path(out_dir, paste0(out_prefix, "_edgelist_model.csv")),
  row.names = FALSE
)

# ----------------------------
# 6) Read author.csv and create standard model attributes
# ----------------------------
author_df <- read.csv(author_file, stringsAsFactors = FALSE, check.names = FALSE)

required_cols <- c(
  "Id",
  impact_raw_var,
  pubcount_raw_var,
  strhole_raw_var,
  field_raw_var,
  interdisc_raw_var,
  strongtie_raw_var,
  tieduration_raw_var,
  highcontrib_raw_var
)
miss_cols <- setdiff(required_cols, names(author_df))
if(length(miss_cols) > 0){
  stop("author.csv 缺少字段: ", paste(miss_cols, collapse = ", "))
}

if(anyDuplicated(author_df$Id)){
  dup_ids <- unique(author_df$Id[duplicated(author_df$Id)])
  stop("author.csv 存在重复 Id，例如: ", paste(head(dup_ids, 10), collapse = ", "))
}

author_df$Id <- as.character(author_df$Id)

node_attr <- data.frame(Id = vnames, stringsAsFactors = FALSE)
node_attr <- merge(node_attr, author_df, by = "Id", all.x = TRUE, sort = FALSE)
node_attr <- node_attr[match(vnames, node_attr$Id), ]

if(any(is.na(node_attr$Id))) stop("author attributes merge failed.")

numeric_raw_vars <- c(
  impact_raw_var,
  pubcount_raw_var,
  strhole_raw_var,
  interdisc_raw_var,
  strongtie_raw_var,
  tieduration_raw_var,
  highcontrib_raw_var
)
for(var_name in numeric_raw_vars){
  converted <- suppressWarnings(as.numeric(node_attr[[var_name]]))
  invalid <- is.na(converted) | !is.finite(converted)
  if(any(invalid)){
    bad_values <- unique(node_attr[[var_name]][invalid])
    stop(
      var_name, " contains non-numeric or non-finite values, e.g. ",
      paste(head(bad_values, 10), collapse = ", ")
    )
  }
  node_attr[[var_name]] <- converted
}

logged_raw_vars <- c(
  if(use_log_pubcount) pubcount_raw_var,
  if(use_log_strhole) strhole_raw_var,
  if(use_log_interdisc) interdisc_raw_var,
  if(use_log_strongtie) strongtie_raw_var,
  if(use_log_tieduration) tieduration_raw_var,
  if(use_log_highcontrib) highcontrib_raw_var
)
for(var_name in logged_raw_vars){
  if(any(node_attr[[var_name]] < 0)){
    stop(var_name, " contains negative values and cannot be transformed safely with log1p().")
  }
}

# impact 0/1
impact_raw <- node_attr[[impact_raw_var]]
if(any(is.na(impact_raw))) stop(impact_raw_var, " 存在 NA。")
impact_bin <- as.integer(impact_raw)
bad_vals_impact <- setdiff(unique(impact_bin), c(0L, 1L))
if(length(bad_vals_impact) > 0){
  stop(impact_raw_var, " 必须是 0/1 变量。发现: ", paste(bad_vals_impact, collapse = ", "))
}
node_attr$impact_bin <- impact_bin

# pubcount
if(any(is.na(node_attr[[pubcount_raw_var]]))) stop(pubcount_raw_var, " 存在 NA。")
pubcount_obj <- transform_variable(node_attr[[pubcount_raw_var]], use_log_pubcount)
node_attr[[paste0(pubcount_name, "_raw")]] <- pubcount_obj$raw
if(use_log_pubcount) node_attr[[paste0(pubcount_name, "_log")]] <- pubcount_obj$trans
node_attr[[pubcount_name]] <- pubcount_obj$z

# strhole
if(any(is.na(node_attr[[strhole_raw_var]]))) stop(strhole_raw_var, " 存在 NA。")
strhole_obj <- transform_variable(node_attr[[strhole_raw_var]], use_log_strhole)
node_attr[[paste0(strhole_name, "_raw")]] <- strhole_obj$raw
if(use_log_strhole) node_attr[[paste0(strhole_name, "_log")]] <- strhole_obj$trans
node_attr[[strhole_name]] <- strhole_obj$z

# interdisc
if(any(is.na(node_attr[[interdisc_raw_var]]))) stop(interdisc_raw_var, " 存在 NA。")
interdisc_obj <- transform_variable(node_attr[[interdisc_raw_var]], use_log_interdisc)
node_attr[[paste0(interdisc_name, "_raw")]] <- interdisc_obj$raw
if(use_log_interdisc) node_attr[[paste0(interdisc_name, "_log")]] <- interdisc_obj$trans
node_attr[[interdisc_name]] <- interdisc_obj$z

# field
field_raw <- node_attr[[field_raw_var]]
if(any(is.na(field_raw) | field_raw == "")){
  stop(field_raw_var, " 存在 NA 或空字符串。")
}
node_attr[[field_name]] <- as.character(field_raw)

# strongtie
if(any(is.na(node_attr[[strongtie_raw_var]]))) stop(strongtie_raw_var, " 存在 NA。")
strongtie_obj <- transform_variable(node_attr[[strongtie_raw_var]], use_log_strongtie)
node_attr[[paste0(strongtie_name, "_raw")]] <- strongtie_obj$raw
if(use_log_strongtie) node_attr[[paste0(strongtie_name, "_log")]] <- strongtie_obj$trans
node_attr[[strongtie_name]] <- strongtie_obj$z

# tieduration
if(any(is.na(node_attr[[tieduration_raw_var]]))) stop(tieduration_raw_var, " 存在 NA。")
tieduration_obj <- transform_variable(node_attr[[tieduration_raw_var]], use_log_tieduration)
node_attr[[paste0(tieduration_name, "_raw")]] <- tieduration_obj$raw
if(use_log_tieduration) node_attr[[paste0(tieduration_name, "_log")]] <- tieduration_obj$trans
node_attr[[tieduration_name]] <- tieduration_obj$z

# highcontrib
if(any(is.na(node_attr[[highcontrib_raw_var]]))) stop(highcontrib_raw_var, " 存在 NA。")
highcontrib_obj <- transform_variable(node_attr[[highcontrib_raw_var]], use_log_highcontrib)
node_attr[[paste0(highcontrib_name, "_raw")]] <- highcontrib_obj$raw
if(use_log_highcontrib) node_attr[[paste0(highcontrib_name, "_log")]] <- highcontrib_obj$trans
node_attr[[highcontrib_name]] <- highcontrib_obj$z

cat("\nTransform setting:\n")
cat("  ", pubcount_raw_var, " -> ", pubcount_name, "\n", sep = "")
cat("  ", strhole_raw_var, " -> ", strhole_name, "\n", sep = "")
cat("  ", interdisc_raw_var, " -> ", interdisc_name, "\n", sep = "")
cat("  ", field_raw_var, " -> ", field_name, "\n", sep = "")
cat("  ", strongtie_raw_var, " -> ", strongtie_name, "\n", sep = "")
cat("  ", tieduration_raw_var, " -> ", tieduration_name, " | sum layer\n", sep = "")
cat("  ", highcontrib_raw_var, " -> ", highcontrib_name, " | sum layer\n", sep = "")

# outputs
write.csv(
  term_dictionary,
  file = file.path(out_dir, paste0(out_prefix, "_table_term_dictionary.csv")),
  row.names = FALSE
)

attr_out <- data.frame(
  Id = node_attr$Id,
  impact_bin = node_attr$impact_bin,
  field = node_attr[[field_name]],
  pubcount_raw = node_attr[[paste0(pubcount_name, "_raw")]],
  strhole_raw = node_attr[[paste0(strhole_name, "_raw")]],
  interdisc_raw = node_attr[[paste0(interdisc_name, "_raw")]],
  strongtie_raw = node_attr[[paste0(strongtie_name, "_raw")]],
  tieduration_raw = node_attr[[paste0(tieduration_name, "_raw")]],
  highcontrib_raw = node_attr[[paste0(highcontrib_name, "_raw")]],
  pubcount = node_attr[[pubcount_name]],
  strhole = node_attr[[strhole_name]],
  interdisc = node_attr[[interdisc_name]],
  strongtie = node_attr[[strongtie_name]],
  tieduration = node_attr[[tieduration_name]],
  highcontrib = node_attr[[highcontrib_name]],
  stringsAsFactors = FALSE,
  check.names = FALSE
)

if(use_log_pubcount)    attr_out[[paste0(pubcount_name, "_log")]]    <- node_attr[[paste0(pubcount_name, "_log")]]
if(use_log_strhole)     attr_out[[paste0(strhole_name, "_log")]]     <- node_attr[[paste0(strhole_name, "_log")]]
if(use_log_interdisc)   attr_out[[paste0(interdisc_name, "_log")]]   <- node_attr[[paste0(interdisc_name, "_log")]]
if(use_log_strongtie)   attr_out[[paste0(strongtie_name, "_log")]]   <- node_attr[[paste0(strongtie_name, "_log")]]
if(use_log_tieduration) attr_out[[paste0(tieduration_name, "_log")]] <- node_attr[[paste0(tieduration_name, "_log")]]
if(use_log_highcontrib) attr_out[[paste0(highcontrib_name, "_log")]] <- node_attr[[paste0(highcontrib_name, "_log")]]

write.csv(
  attr_out,
  file = file.path(out_dir, paste0(out_prefix, "_table_author_attributes_used.csv")),
  row.names = FALSE
)

field_tab <- as.data.frame(table(node_attr[[field_name]]), stringsAsFactors = FALSE)
names(field_tab) <- c("field", "n_authors")
write.csv(
  field_tab,
  file = file.path(out_dir, paste0(out_prefix, "_table_field_counts.csv")),
  row.names = FALSE
)

summary_entries <- list(
  summary_row(node_attr[[paste0(pubcount_name, "_raw")]], paste0(pubcount_name, "_raw")),
  if(use_log_pubcount) summary_row(node_attr[[paste0(pubcount_name, "_log")]], paste0(pubcount_name, "_log")) else NULL,
  summary_row(node_attr[[pubcount_name]], pubcount_name),

  summary_row(node_attr[[paste0(strhole_name, "_raw")]], paste0(strhole_name, "_raw")),
  if(use_log_strhole) summary_row(node_attr[[paste0(strhole_name, "_log")]], paste0(strhole_name, "_log")) else NULL,
  summary_row(node_attr[[strhole_name]], strhole_name),

  summary_row(node_attr[[paste0(interdisc_name, "_raw")]], paste0(interdisc_name, "_raw")),
  if(use_log_interdisc) summary_row(node_attr[[paste0(interdisc_name, "_log")]], paste0(interdisc_name, "_log")) else NULL,
  summary_row(node_attr[[interdisc_name]], interdisc_name),

  summary_row(node_attr[[paste0(strongtie_name, "_raw")]], paste0(strongtie_name, "_raw")),
  if(use_log_strongtie) summary_row(node_attr[[paste0(strongtie_name, "_log")]], paste0(strongtie_name, "_log")) else NULL,
  summary_row(node_attr[[strongtie_name]], strongtie_name),

  summary_row(node_attr[[paste0(tieduration_name, "_raw")]], paste0(tieduration_name, "_raw")),
  if(use_log_tieduration) summary_row(node_attr[[paste0(tieduration_name, "_log")]], paste0(tieduration_name, "_log")) else NULL,
  summary_row(node_attr[[tieduration_name]], tieduration_name),

  summary_row(node_attr[[paste0(highcontrib_name, "_raw")]], paste0(highcontrib_name, "_raw")),
  if(use_log_highcontrib) summary_row(node_attr[[paste0(highcontrib_name, "_log")]], paste0(highcontrib_name, "_log")) else NULL,
  summary_row(node_attr[[highcontrib_name]], highcontrib_name)
)
summary_entries <- Filter(Negate(is.null), summary_entries)

attr_summary <- do.call(rbind, summary_entries)
write.csv(
  attr_summary,
  file = file.path(out_dir, paste0(out_prefix, "_table_attribute_summary.csv")),
  row.names = FALSE
)

attr_cor_df <- data.frame(
  pubcount = node_attr[[pubcount_name]],
  strhole = node_attr[[strhole_name]],
  interdisc = node_attr[[interdisc_name]],
  strongtie = node_attr[[strongtie_name]],
  tieduration = node_attr[[tieduration_name]],
  highcontrib = node_attr[[highcontrib_name]],
  check.names = FALSE
)
attr_cor <- cor(attr_cor_df, use = "pairwise.complete.obs")
write.csv(
  attr_cor,
  file = file.path(out_dir, paste0(out_prefix, "_table_attribute_correlation_all.csv")),
  row.names = TRUE
)

transform_config <- data.frame(
  standard_name = c(pubcount_name, strhole_name, interdisc_name, field_name, strongtie_name, tieduration_name, highcontrib_name),
  source_column = c(pubcount_raw_var, strhole_raw_var, interdisc_raw_var, field_raw_var, strongtie_raw_var, tieduration_raw_var, highcontrib_raw_var),
  use_log = c(use_log_pubcount, use_log_strhole, use_log_interdisc, NA, use_log_strongtie, use_log_tieduration, use_log_highcontrib),
  model_layer = c("nonzero","nonzero","nonzero","nonzero","nonzero","sum","sum"),
  stringsAsFactors = FALSE
)
write.csv(
  transform_config,
  file = file.path(out_dir, paste0(out_prefix, "_table_transform_config.csv")),
  row.names = FALSE
)

model_spec_table <- data.frame(
  model = c("M0","M1","M_core","M_prod","M_hole","M_full"),
  description = c(
    "helper only",
    "H1 structural baseline",
    "M1 + highlow + field + interdisc",
    "M1 + highlow + pubcount + field + interdisc + strongtie + tieduration(sum) + highcontrib(sum)",
    "M1 + highlow + strhole + field + interdisc + strongtie + tieduration(sum) + highcontrib(sum)",
    "M1 + highlow + pubcount + strhole + field + interdisc + strongtie + tieduration(sum) + highcontrib(sum)"
  ),
  stringsAsFactors = FALSE
)
write.csv(
  model_spec_table,
  file = file.path(out_dir, paste0(out_prefix, "_table_model_specification.csv")),
  row.names = FALSE
)

# ----------------------------
# 6.1) Model-specific approximate collinearity tables + console output
# ----------------------------
model_covariate_map <- list(
  M0 = character(0),
  M1 = character(0),
  M_core = c(interdisc_name),
  M_prod = c(pubcount_name, interdisc_name, strongtie_name, tieduration_name, highcontrib_name),
  M_hole = c(strhole_name, interdisc_name, strongtie_name, tieduration_name, highcontrib_name),
  M_full = c(pubcount_name, strhole_name, interdisc_name, strongtie_name, tieduration_name, highcontrib_name)
)

collinearity_results <- list()

for(mn in names(model_covariate_map)){
  coll_obj <- write_model_collinearity_outputs(
    df = node_attr,
    vars = model_covariate_map[[mn]],
    model_name = mn,
    out_dir = out_dir,
    out_prefix = out_prefix
  )
  collinearity_results[[mn]] <- coll_obj
  print_collinearity_console(mn, coll_obj)
}

collinearity_summary_all <- do.call(
  rbind,
  lapply(collinearity_results, function(x) x$summary)
)
write.csv(
  collinearity_summary_all,
  file = file.path(out_dir, paste0(out_prefix, "_table_collinearity_summary_all_models.csv")),
  row.names = FALSE
)

# ----------------------------
# 7) Build statnet network
# ----------------------------
idmap <- setNames(seq_len(n), vnames)

miss_from <- setdiff(unique(el_m$from), vnames)
miss_to   <- setdiff(unique(el_m$to), vnames)
if(length(miss_from) > 0 || length(miss_to) > 0){
  stop("Edge endpoints not in node list.")
}

net <- network::network.initialize(n, directed = FALSE)
network::network.vertex.names(net) <- vnames
network::add.edges(net, tail = idmap[el_m$from], head = idmap[el_m$to])
network::set.edge.attribute(net, "w", el_m$w_cnt)

network::set.vertex.attribute(net, pubcount_name, node_attr[[pubcount_name]])
network::set.vertex.attribute(net, strhole_name, node_attr[[strhole_name]])
network::set.vertex.attribute(net, interdisc_name, node_attr[[interdisc_name]])
network::set.vertex.attribute(net, field_name, node_attr[[field_name]])
network::set.vertex.attribute(net, strongtie_name, node_attr[[strongtie_name]])
network::set.vertex.attribute(net, tieduration_name, node_attr[[tieduration_name]])
network::set.vertex.attribute(net, highcontrib_name, node_attr[[highcontrib_name]])

m_obs <- network::network.edgecount(net)
D_dyads <- n * (n - 1) / 2
sum_obs <- as.numeric(summary(net ~ sum, response = "w"))
P0_obs <- 1 - m_obs / D_dyads
mean_y_obs <- sum_obs / D_dyads

cat("\nNetwork built: nodes=", network::network.size(net),
    " edges(nonzero)=", m_obs, "\n")

net_table <- data.frame(
  n_nodes = n,
  n_edges_nonzero = m_obs,
  dyads_D = D_dyads,
  density_nonzero = m_obs / D_dyads,
  P0_zero_share = P0_obs,
  disc_method = disc_method_for_model,
  delta = delta,
  cap_q = cap_q,
  trials_M = M,
  sum_w_cnt = sum_obs,
  mean_y_over_all_dyads = mean_y_obs,
  stringsAsFactors = FALSE
)
write.csv(
  net_table,
  file = file.path(out_dir, paste0(out_prefix, "_table_network_summary.csv")),
  row.names = FALSE
)

# ----------------------------
# 8) Build highlow dyad matrix
# ----------------------------
imp <- node_attr$impact_bin

highlow <- outer(imp, imp, function(a, b) as.integer((a + b) == 1L))
diag(highlow) <- 0
mode(highlow) <- "numeric"

HH <- outer(imp, imp, function(a, b) as.integer(a == 1L & b == 1L))
LL <- outer(imp, imp, function(a, b) as.integer(a == 0L & b == 0L))
diag(HH) <- 0
diag(LL) <- 0
mode(HH) <- "numeric"
mode(LL) <- "numeric"

dyad_HL <- sum(highlow) / 2
dyad_HH <- sum(HH) / 2
dyad_LL <- sum(LL) / 2

edge_type <- ifelse(
  imp[idmap[el_m$from]] + imp[idmap[el_m$to]] == 1L, "HL",
  ifelse(imp[idmap[el_m$from]] + imp[idmap[el_m$to]] == 2L, "HH", "LL")
)

edge_mix_tab <- aggregate(
  cbind(n_edges = rep(1, nrow(el_m)), sum_w = el_m$w_cnt),
  by = list(type = edge_type),
  FUN = sum
)

dyad_mix_tab <- data.frame(
  type = c("LL", "HL", "HH"),
  dyads_all_possible = c(dyad_LL, dyad_HL, dyad_HH),
  stringsAsFactors = FALSE
)

mix_table <- merge(dyad_mix_tab, edge_mix_tab, by = "type", all.x = TRUE, sort = FALSE)
mix_table$n_edges[is.na(mix_table$n_edges)] <- 0
mix_table$sum_w[is.na(mix_table$sum_w)] <- 0
mix_table$edge_density_nonzero <- mix_table$n_edges / mix_table$dyads_all_possible

write.csv(
  mix_table,
  file = file.path(out_dir, paste0(out_prefix, "_table_observed_highlow_mix.csv")),
  row.names = FALSE
)

# same-field vs different-field
fld <- node_attr[[field_name]]
same_field <- outer(fld, fld, function(a, b) as.integer(a == b))
diag(same_field) <- 0
mode(same_field) <- "numeric"

dyad_same_field <- sum(same_field) / 2
dyad_diff_field <- D_dyads - dyad_same_field

edge_same_field <- fld[idmap[el_m$from]] == fld[idmap[el_m$to]]
field_mix_tab <- aggregate(
  cbind(n_edges = rep(1, nrow(el_m)), sum_w = el_m$w_cnt),
  by = list(type = ifelse(edge_same_field, "same_field", "different_field")),
  FUN = sum
)

field_dyad_tab <- data.frame(
  type = c("same_field", "different_field"),
  dyads_all_possible = c(dyad_same_field, dyad_diff_field),
  stringsAsFactors = FALSE
)

field_obs_tab <- merge(
  field_dyad_tab, field_mix_tab,
  by = "type", all.x = TRUE, sort = FALSE
)
field_obs_tab$n_edges[is.na(field_obs_tab$n_edges)] <- 0
field_obs_tab$sum_w[is.na(field_obs_tab$sum_w)] <- 0
field_obs_tab$edge_density_nonzero <- field_obs_tab$n_edges / field_obs_tab$dyads_all_possible

write.csv(
  field_obs_tab,
  file = file.path(out_dir, paste0(out_prefix, "_table_observed_field_mix.csv")),
  row.names = FALSE
)

# ----------------------------
# 9) maxedges cap + nonident config
# ----------------------------
maxedges_cap <- max(5 * m_obs, maxedges_mult * m_obs)

nonident_cfg <- data.frame(
  parameter = c("MPLE.nonident","MPLE.nonident.tol","MCMLE.nonident","MCMLE.nonident.tol","MPLE.nonvar","MCMLE.nonvar"),
  value = c(nonident_action, nonident_tol, nonident_action, nonident_tol, "warning", "warning"),
  stringsAsFactors = FALSE
)
write.csv(
  nonident_cfg,
  file = file.path(out_dir, paste0(out_prefix, "_table_nonident_config.csv")),
  row.names = FALSE
)

preflight_only <- tolower(trimws(Sys.getenv("MODEL5_PREFLIGHT_ONLY", "false"))) %in%
  c("1", "true", "yes", "y")
if(preflight_only){
  cat("\nPREFLIGHT_OK\n")
  cat("Inputs, packages, transformations, network construction, and output directory are valid.\n")
  cat("MCMC model estimation was skipped because MODEL5_PREFLIGHT_ONLY is enabled.\n")
  quit(save = "no", status = 0, runLast = FALSE)
}

# ----------------------------
# 10) M0 helper
# ----------------------------
cat("\n===== M0 helper =====\n")
set.seed(SEED)

base_init <- analytic_init_sum_nonzero(M, P0_obs, mean_y_obs)
init_base <- c(
  sum = unname(base_init["sum"]),
  nonzero = unname(base_init["nonzero"]) - 1.0
)

res_m0 <- run_ergm_safe(
  model_name = "M0",
  formula_obj = net ~ sum + nonzero,
  response_name = "w",
  reference_obj = ~Binomial(trials = M),
  offset_vec = NULL,
  control_obj = ctrl_base(init_base, m_obs, maxedges_cap),
  out_dir = out_dir,
  out_prefix = out_prefix,
  eval.loglik = TRUE,
  verbose = TRUE
)
fit_m0 <- res_m0$fit

if(is.null(fit_m0)) stop("M0 helper failed; cannot continue.")

saveRDS(fit_m0, file = file.path(out_dir, paste0(out_prefix, "_fit_M0.rds")))
write_coef_table(fit_m0, file.path(out_dir, paste0(out_prefix, "_table_coef_M0.csv")))
save_mcmc_outputs(fit_m0, "M0", out_dir, out_prefix)
print_coef_console(fit_m0, "M0")

offset_vec <- unname(coef(fit_m0)[c("sum", "nonzero")])

# ----------------------------
# 11) M1: structural baseline
# ----------------------------
cat("\n===== M1: structural baseline =====\n")
set.seed(SEED)

form_m1 <- build_formula(lhs = "net")
init_m1 <- c(offset_vec, 0)

res_m1 <- run_ergm_safe(
  model_name = "M1",
  formula_obj = form_m1,
  response_name = "w",
  reference_obj = ~Binomial(trials = M),
  offset_vec = offset_vec,
  control_obj = ctrl_struct(init_m1, m_obs, maxedges_cap),
  out_dir = out_dir,
  out_prefix = out_prefix,
  eval.loglik = TRUE,
  verbose = TRUE
)
fit_m1 <- res_m1$fit

if(is.null(fit_m1)) stop("M1 failed; cannot continue.")

saveRDS(fit_m1, file = file.path(out_dir, paste0(out_prefix, "_fit_M1.rds")))
write_coef_table(fit_m1, file.path(out_dir, paste0(out_prefix, "_table_coef_M1.csv")))
save_mcmc_outputs(fit_m1, "M1", out_dir, out_prefix)
print_coef_console(fit_m1, "M1")

gwdeg_start <- extract_coef_value(fit_m1, "gwdeg", 0)

# ----------------------------
# 12) M_core
# ----------------------------
cat("\n===== M_core =====\n")
set.seed(SEED)

form_mcore <- build_formula(
  lhs = "net",
  include_highlow = TRUE,
  include_field = TRUE,
  include_interdisc = TRUE
)

init_mcore <- c(
  offset_vec,
  gwdeg_start,
  0,  # highlow
  0,  # field
  0   # interdisc
)

res_mcore <- run_ergm_safe(
  model_name = "M_core",
  formula_obj = form_mcore,
  response_name = "w",
  reference_obj = ~Binomial(trials = M),
  offset_vec = offset_vec,
  control_obj = ctrl_attr(init_mcore, m_obs, maxedges_cap),
  out_dir = out_dir,
  out_prefix = out_prefix,
  eval.loglik = TRUE,
  verbose = TRUE
)
fit_mcore <- res_mcore$fit

if(!is.null(fit_mcore)){
  saveRDS(fit_mcore, file = file.path(out_dir, paste0(out_prefix, "_fit_M_core.rds")))
  write_coef_table(fit_mcore, file.path(out_dir, paste0(out_prefix, "_table_coef_M_core.csv")))
  save_mcmc_outputs(fit_mcore, "M_core", out_dir, out_prefix)
  print_coef_console(fit_mcore, "M_core")
}

gwdeg_core_start     <- extract_coef_value(fit_mcore, "gwdeg", gwdeg_start)
highlow_core_start   <- extract_coef_value(fit_mcore, "edgecov", 0)
field_core_start     <- extract_coef_value(fit_mcore, "nodematch.*field", 0)
interdisc_core_start <- extract_coef_value(fit_mcore, "nodecov.*interdisc", 0)

# ----------------------------
# 13) M_prod
# ----------------------------
cat("\n===== M_prod =====\n")
set.seed(SEED)

form_mprod <- build_formula(
  lhs = "net",
  include_highlow = TRUE,
  include_pubcount = TRUE,
  include_field = TRUE,
  include_interdisc = TRUE,
  include_strongtie = TRUE,
  include_tieduration = TRUE,
  include_highcontrib = TRUE
)

init_mprod <- c(
  offset_vec,
  gwdeg_core_start,
  highlow_core_start,
  0,                    # pubcount
  field_core_start,
  interdisc_core_start,
  0,                    # strongtie
  0,                    # tieduration
  0                     # highcontrib
)

res_mprod <- run_ergm_safe(
  model_name = "M_prod",
  formula_obj = form_mprod,
  response_name = "w",
  reference_obj = ~Binomial(trials = M),
  offset_vec = offset_vec,
  control_obj = ctrl_attr(init_mprod, m_obs, maxedges_cap),
  out_dir = out_dir,
  out_prefix = out_prefix,
  eval.loglik = TRUE,
  verbose = TRUE
)
fit_mprod <- res_mprod$fit

if(!is.null(fit_mprod)){
  saveRDS(fit_mprod, file = file.path(out_dir, paste0(out_prefix, "_fit_M_prod.rds")))
  write_coef_table(fit_mprod, file.path(out_dir, paste0(out_prefix, "_table_coef_M_prod.csv")))
  save_mcmc_outputs(fit_mprod, "M_prod", out_dir, out_prefix)
  print_coef_console(fit_mprod, "M_prod")
}

# ----------------------------
# 14) M_hole
# ----------------------------
cat("\n===== M_hole =====\n")
set.seed(SEED)

form_mhole <- build_formula(
  lhs = "net",
  include_highlow = TRUE,
  include_strhole = TRUE,
  include_field = TRUE,
  include_interdisc = TRUE,
  include_strongtie = TRUE,
  include_tieduration = TRUE,
  include_highcontrib = TRUE
)

init_mhole <- c(
  offset_vec,
  gwdeg_core_start,
  highlow_core_start,
  0,                    # strhole
  field_core_start,
  interdisc_core_start,
  0,                    # strongtie
  0,                    # tieduration
  0                     # highcontrib
)

res_mhole <- run_ergm_safe(
  model_name = "M_hole",
  formula_obj = form_mhole,
  response_name = "w",
  reference_obj = ~Binomial(trials = M),
  offset_vec = offset_vec,
  control_obj = ctrl_attr(init_mhole, m_obs, maxedges_cap),
  out_dir = out_dir,
  out_prefix = out_prefix,
  eval.loglik = TRUE,
  verbose = TRUE
)
fit_mhole <- res_mhole$fit

if(!is.null(fit_mhole)){
  saveRDS(fit_mhole, file = file.path(out_dir, paste0(out_prefix, "_fit_M_hole.rds")))
  write_coef_table(fit_mhole, file.path(out_dir, paste0(out_prefix, "_table_coef_M_hole.csv")))
  save_mcmc_outputs(fit_mhole, "M_hole", out_dir, out_prefix)
  print_coef_console(fit_mhole, "M_hole")
}

# starts for full model
gwdeg_full_start       <- if(!is.null(fit_mprod)) extract_coef_value(fit_mprod, "gwdeg", gwdeg_core_start) else if(!is.null(fit_mhole)) extract_coef_value(fit_mhole, "gwdeg", gwdeg_core_start) else gwdeg_core_start
highlow_full_start     <- if(!is.null(fit_mprod)) extract_coef_value(fit_mprod, "edgecov", highlow_core_start) else if(!is.null(fit_mhole)) extract_coef_value(fit_mhole, "edgecov", highlow_core_start) else highlow_core_start
pubcount_full_start    <- if(!is.null(fit_mprod)) extract_coef_value(fit_mprod, "nodecov.*pubcount", 0) else 0
strhole_full_start     <- if(!is.null(fit_mhole)) extract_coef_value(fit_mhole, "nodecov.*strhole", 0) else 0
field_full_start       <- if(!is.null(fit_mprod)) extract_coef_value(fit_mprod, "nodematch.*field", field_core_start) else if(!is.null(fit_mhole)) extract_coef_value(fit_mhole, "nodematch.*field", field_core_start) else field_core_start
interdisc_full_start   <- if(!is.null(fit_mprod)) extract_coef_value(fit_mprod, "nodecov.*interdisc", interdisc_core_start) else if(!is.null(fit_mhole)) extract_coef_value(fit_mhole, "nodecov.*interdisc", interdisc_core_start) else interdisc_core_start
strongtie_full_start   <- if(!is.null(fit_mprod)) extract_coef_value(fit_mprod, "nodecov.*strongtie", 0) else if(!is.null(fit_mhole)) extract_coef_value(fit_mhole, "nodecov.*strongtie", 0) else 0
tieduration_full_start <- if(!is.null(fit_mprod)) extract_coef_value(fit_mprod, "nodecov.*tieduration", 0) else if(!is.null(fit_mhole)) extract_coef_value(fit_mhole, "nodecov.*tieduration", 0) else 0
highcontrib_full_start <- if(!is.null(fit_mprod)) extract_coef_value(fit_mprod, "nodecov.*highcontrib", 0) else if(!is.null(fit_mhole)) extract_coef_value(fit_mhole, "nodecov.*highcontrib", 0) else 0

# ----------------------------
# 15) M_full
# ----------------------------
cat("\n===== M_full =====\n")
set.seed(SEED)

form_mfull <- build_formula(
  lhs = "net",
  include_highlow = TRUE,
  include_pubcount = TRUE,
  include_strhole = TRUE,
  include_field = TRUE,
  include_interdisc = TRUE,
  include_strongtie = TRUE,
  include_tieduration = TRUE,
  include_highcontrib = TRUE
)

init_mfull <- c(
  offset_vec,
  gwdeg_full_start,
  highlow_full_start,
  pubcount_full_start,
  strhole_full_start,
  field_full_start,
  interdisc_full_start,
  strongtie_full_start,
  tieduration_full_start,
  highcontrib_full_start
)

res_mfull <- run_ergm_safe(
  model_name = "M_full",
  formula_obj = form_mfull,
  response_name = "w",
  reference_obj = ~Binomial(trials = M),
  offset_vec = offset_vec,
  control_obj = ctrl_attr(init_mfull, m_obs, maxedges_cap),
  out_dir = out_dir,
  out_prefix = out_prefix,
  eval.loglik = TRUE,
  verbose = TRUE
)
fit_mfull <- res_mfull$fit

if(!is.null(fit_mfull)){
  saveRDS(fit_mfull, file = file.path(out_dir, paste0(out_prefix, "_fit_M_full.rds")))
  write_coef_table(fit_mfull, file.path(out_dir, paste0(out_prefix, "_table_coef_M_full.csv")))
  save_mcmc_outputs(fit_mfull, "M_full", out_dir, out_prefix)
  print_coef_console(fit_mfull, "M_full")
}

# ----------------------------
# 16) IC table
# ----------------------------
ic_tab <- do.call(rbind, list(
  safe_loglik_ic(fit_m0,    D_dyads, "M0_helper"),
  safe_loglik_ic(fit_m1,    D_dyads, "M1_baseline"),
  safe_loglik_ic(fit_mcore, D_dyads, "M_core"),
  safe_loglik_ic(fit_mprod, D_dyads, "M_prod"),
  safe_loglik_ic(fit_mhole, D_dyads, "M_hole"),
  safe_loglik_ic(fit_mfull, D_dyads, "M_full")
))

write.csv(
  ic_tab,
  file = file.path(out_dir, paste0(out_prefix, "_table_ic_models.csv")),
  row.names = FALSE
)

# ----------------------------
# 17) Detailed result tables
# ----------------------------
model_term_long_list <- list(
  build_model_term_long(fit_m0,    "M0_helper",   0, D_dyads),
  build_model_term_long(fit_m1,    "M1_baseline", 1, D_dyads),
  build_model_term_long(fit_mcore, "M_core",      2, D_dyads),
  build_model_term_long(fit_mprod, "M_prod",      3, D_dyads),
  build_model_term_long(fit_mhole, "M_hole",      4, D_dyads),
  build_model_term_long(fit_mfull, "M_full",      5, D_dyads)
)
model_term_long_list <- Filter(Negate(is.null), model_term_long_list)
model_term_long <- do.call(rbind, model_term_long_list)

write.csv(
  model_term_long,
  file = file.path(out_dir, paste0(out_prefix, "_table_model_term_results_long.csv")),
  row.names = FALSE
)

write.csv(
  model_term_long[!model_term_long$is_offset, ],
  file = file.path(out_dir, paste0(out_prefix, "_table_model_term_results_nonoffset.csv")),
  row.names = FALSE
)

wide_estimate <- make_wide_table(model_term_long, "estimate", "term")
wide_se       <- make_wide_table(model_term_long, "std_error", "term")
wide_p        <- make_wide_table(model_term_long, "p_value", "term")
wide_stars    <- make_wide_table(model_term_long, "p_stars", "term")
wide_ci_low   <- make_wide_table(model_term_long, "ci95_low", "term")
wide_ci_high  <- make_wide_table(model_term_long, "ci95_high", "term")

if(!is.null(wide_estimate)){
  write.csv(wide_estimate, file = file.path(out_dir, paste0(out_prefix, "_table_model_term_estimate_wide.csv")), row.names = FALSE)
  write.csv(wide_se,       file = file.path(out_dir, paste0(out_prefix, "_table_model_term_se_wide.csv")), row.names = FALSE)
  write.csv(wide_p,        file = file.path(out_dir, paste0(out_prefix, "_table_model_term_pvalue_wide.csv")), row.names = FALSE)
  write.csv(wide_stars,    file = file.path(out_dir, paste0(out_prefix, "_table_model_term_stars_wide.csv")), row.names = FALSE)
  write.csv(wide_ci_low,   file = file.path(out_dir, paste0(out_prefix, "_table_model_term_ci95low_wide.csv")), row.names = FALSE)
  write.csv(wide_ci_high,  file = file.path(out_dir, paste0(out_prefix, "_table_model_term_ci95high_wide.csv")), row.names = FALSE)
}

fit_summary_table <- ic_tab
fit_summary_table$model_order <- seq_len(nrow(fit_summary_table))
fit_summary_table <- fit_summary_table[, c("model_order", "model", "k_total", "k_free", "logLik", "AIC_manual", "BIC_dyads", "AIC_R", "BIC_R")]
write.csv(
  fit_summary_table,
  file = file.path(out_dir, paste0(out_prefix, "_table_model_fit_summary.csv")),
  row.names = FALSE
)

nonident_summary_all <- do.call(rbind, list(
  read.csv(file.path(out_dir, paste0(out_prefix, "_table_nonident_M0.csv")), stringsAsFactors = FALSE),
  read.csv(file.path(out_dir, paste0(out_prefix, "_table_nonident_M1.csv")), stringsAsFactors = FALSE),
  read.csv(file.path(out_dir, paste0(out_prefix, "_table_nonident_M_core.csv")), stringsAsFactors = FALSE),
  read.csv(file.path(out_dir, paste0(out_prefix, "_table_nonident_M_prod.csv")), stringsAsFactors = FALSE),
  read.csv(file.path(out_dir, paste0(out_prefix, "_table_nonident_M_hole.csv")), stringsAsFactors = FALSE),
  read.csv(file.path(out_dir, paste0(out_prefix, "_table_nonident_M_full.csv")), stringsAsFactors = FALSE)
))
write.csv(
  nonident_summary_all,
  file = file.path(out_dir, paste0(out_prefix, "_table_nonident_all_models.csv")),
  row.names = FALSE
)

# ----------------------------
# 18) GOF outputs
# ----------------------------
if(run_gof){

  get_stats_m0 <- function(nw){
    summary(nw ~ sum + nonzero, response = "w")
  }

  form_gof_m1 <- build_formula(lhs = "nw")
  form_gof_mcore <- build_formula(
    lhs = "nw",
    include_highlow = TRUE,
    include_field = TRUE,
    include_interdisc = TRUE
  )
  form_gof_mprod <- build_formula(
    lhs = "nw",
    include_highlow = TRUE,
    include_pubcount = TRUE,
    include_field = TRUE,
    include_interdisc = TRUE,
    include_strongtie = TRUE,
    include_tieduration = TRUE,
    include_highcontrib = TRUE
  )
  form_gof_mhole <- build_formula(
    lhs = "nw",
    include_highlow = TRUE,
    include_strhole = TRUE,
    include_field = TRUE,
    include_interdisc = TRUE,
    include_strongtie = TRUE,
    include_tieduration = TRUE,
    include_highcontrib = TRUE
  )
  form_gof_mfull <- build_formula(
    lhs = "nw",
    include_highlow = TRUE,
    include_pubcount = TRUE,
    include_strhole = TRUE,
    include_field = TRUE,
    include_interdisc = TRUE,
    include_strongtie = TRUE,
    include_tieduration = TRUE,
    include_highcontrib = TRUE
  )

  get_stats_m1    <- function(nw) summary_formula_on_network(form_gof_m1, nw)
  get_stats_mcore <- function(nw) summary_formula_on_network(form_gof_mcore, nw)
  get_stats_mprod <- function(nw) summary_formula_on_network(form_gof_mprod, nw)
  get_stats_mhole <- function(nw) summary_formula_on_network(form_gof_mhole, nw)
  get_stats_mfull <- function(nw) summary_formula_on_network(form_gof_mfull, nw)

  gof_list <- list(
    save_gof_outputs("M0",     fit_m0,    net, get_stats_m0,    nsim_gof, SEED, out_dir, out_prefix),
    save_gof_outputs("M1",     fit_m1,    net, get_stats_m1,    nsim_gof, SEED, out_dir, out_prefix),
    save_gof_outputs("M_core", fit_mcore, net, get_stats_mcore, nsim_gof, SEED, out_dir, out_prefix),
    save_gof_outputs("M_prod", fit_mprod, net, get_stats_mprod, nsim_gof, SEED, out_dir, out_prefix),
    save_gof_outputs("M_hole", fit_mhole, net, get_stats_mhole, nsim_gof, SEED, out_dir, out_prefix),
    save_gof_outputs("M_full", fit_mfull, net, get_stats_mfull, nsim_gof, SEED, out_dir, out_prefix)
  )
  gof_list <- Filter(Negate(is.null), gof_list)

  if(length(gof_list) > 0){
    gof_long <- do.call(rbind, lapply(gof_list, function(x) x$long))
    write.csv(
      gof_long,
      file = file.path(out_dir, paste0(out_prefix, "_table_gof_results_long.csv")),
      row.names = FALSE
    )
  }
}

# ----------------------------
# 19) Console summaries
# ----------------------------
cat("\n===== IC table =====\n")
print(ic_tab)

if(print_raw_ergm_summary){
  cat("\n===== RAW ergm summary: M1 =====\n")
  print(summary(fit_m1))

  if(!is.null(fit_mcore)){
    cat("\n===== RAW ergm summary: M_core =====\n")
    print(summary(fit_mcore))
  }

  if(!is.null(fit_mprod)){
    cat("\n===== RAW ergm summary: M_prod =====\n")
    print(summary(fit_mprod))
  }

  if(!is.null(fit_mhole)){
    cat("\n===== RAW ergm summary: M_hole =====\n")
    print(summary(fit_mhole))
  }

  if(!is.null(fit_mfull)){
    cat("\n===== RAW ergm summary: M_full =====\n")
    print(summary(fit_mfull))
  }
}

cat("\nDONE. Outputs in: ", normalizePath(out_dir), "\n", sep = "")
cat("Key files:\n")
cat(" - ", paste0(out_prefix, "_table_term_dictionary.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_transform_config.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_model_specification.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_author_attributes_used.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_attribute_summary.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_attribute_correlation_all.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_collinearity_summary_all_models.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_nonident_all_models.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_model_fit_summary.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_model_term_results_long.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_model_term_results_nonoffset.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_model_term_estimate_wide.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_model_term_se_wide.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_model_term_pvalue_wide.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_model_term_stars_wide.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_model_term_ci95low_wide.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_model_term_ci95high_wide.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_ic_models.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_coef_M0.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_coef_M1.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_coef_M_core.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_coef_M_prod.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_coef_M_hole.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_coef_M_full.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_collinearity_summary_M0.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_collinearity_summary_M1.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_collinearity_summary_M_core.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_collinearity_summary_M_prod.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_collinearity_summary_M_hole.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_collinearity_summary_M_full.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_nonident_M0.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_nonident_M1.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_nonident_M_core.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_nonident_M_prod.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_nonident_M_hole.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_table_nonident_M_full.csv"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_warnings_M0.txt"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_warnings_M1.txt"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_warnings_M_core.txt"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_warnings_M_prod.txt"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_warnings_M_hole.txt"), "\n", sep = "")
cat(" - ", paste0(out_prefix, "_warnings_M_full.txt"), "\n", sep = "")

if(save_diag_plots){
  cat(" - ", paste0(out_prefix, "_diag_official_M0.[pdf/png]"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_diag_official_M1.[pdf/png]"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_diag_official_M_core.[pdf/png]"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_diag_official_M_prod.[pdf/png]"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_diag_official_M_hole.[pdf/png]"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_diag_official_M_full.[pdf/png]"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_diag_compact_M0.[pdf/png]"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_diag_compact_M1.[pdf/png]"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_diag_compact_M_core.[pdf/png]"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_diag_compact_M_prod.[pdf/png]"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_diag_compact_M_hole.[pdf/png]"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_diag_compact_M_full.[pdf/png]"), "\n", sep = "")
}

if(run_gof){
  cat(" - ", paste0(out_prefix, "_table_gof_results_long.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_table_gof_M0.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_table_gof_M1.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_table_gof_M_core.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_table_gof_M_prod.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_table_gof_M_hole.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_table_gof_M_full.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gof_M0_sim_stats_raw.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gof_M1_sim_stats_raw.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gof_M_core_sim_stats_raw.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gof_M_prod_sim_stats_raw.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gof_M_hole_sim_stats_raw.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gof_M_full_sim_stats_raw.csv"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gofplot_M0.pdf"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gofplot_M1.pdf"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gofplot_M_core.pdf"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gofplot_M_prod.pdf"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gofplot_M_hole.pdf"), "\n", sep = "")
  cat(" - ", paste0(out_prefix, "_gofplot_M_full.pdf"), "\n", sep = "")
}
