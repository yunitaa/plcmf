# =============================================================================
# 0. INSTALASI & LOAD PACKAGE
# =============================================================================

packages <- c("MTS", "vars", "ggplot2", "reshape2", "gridExtra",
              "lmtest", "tseries", "corrplot", "RColorBrewer",
              "dplyr", "tidyr", "patchwork", "ggcorrplot",
              "urca")          # [TAMBAHAN] untuk ADF dengan seleksi lag AIC

installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed], dependencies = TRUE)
}

invisible(lapply(packages, library, character.only = TRUE))

cat("Semua package berhasil dimuat.\n")


# =============================================================================
# KONFIGURASI GLOBAL
# =============================================================================

REGIONS    <- c("region_1", "region_2", "region_3", "region_4",
                "region_5", "region_6")
PERIODS    <- c("period_1", "period_2", "period_3")
N_LAGS     <- 16       # lag maksimal PLCMF dan CCF
ALPHA      <- 0.05     # tingkat signifikansi seragam
MAX_LAG_GC <- 10       # batas atas orde lag Granger (= 5 jam @ 30 menit)
MAX_LAG_ADF <- 20      # batas atas lag augmentasi ADF
OUTPUT_DIR <- "./output_analysis"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

REGION_MAP <- c(
  "region_1" = "region_1",
  "region_2" = "region_2",
  "region_3" = "region_3",
  "region_4" = "region_4",
  "region_5" = "region_5",
  "region_6" = "region_6"
)

DATE_FORMAT <- "%Y-%m-%d"


# =============================================================================
# BAGIAN 0: LOAD / GENERATE DATA
# =============================================================================

generate_synthetic_data <- function(n = 500, seed = 42) {
  set.seed(seed)
  data_list <- list()

  for (period in PERIODS) {
    common <- cumsum(rnorm(n)) * 0.3
    mat    <- matrix(0, nrow = n, ncol = length(REGIONS))
    colnames(mat) <- REGIONS

    for (i in seq_along(REGIONS)) {
      ar1   <- 0.6 + runif(1, -0.1, 0.1)
      ar2   <- 0.2 + runif(1, -0.05, 0.05)
      noise <- rnorm(n, 0, 0.5 + i * 0.1)

      for (t in 3:n) {
        mat[t, i] <- ar1 * mat[t-1, i] + ar2 * mat[t-2, i] +
                     0.4 * common[t] + noise[t]
        if (i > 1) {
          mat[t, i] <- mat[t, i] + 0.2 * mat[t-1, i-1]
        }
      }
    }

    df <- as.data.frame(scale(mat))
    data_list[[period]] <- df
    cat(sprintf("Data %s (sintetis): %d observasi, %d region\n",
                period, nrow(df), ncol(df)))
  }
  return(data_list)
}


load_one_period <- function(filepath, period) {
  df <- read.csv(filepath, stringsAsFactors = FALSE,
                 fileEncoding = "UTF-8-BOM")
  colnames(df) <- trimws(colnames(df))

  expected_cols <- c("date", "time", names(REGION_MAP))
  missing_cols  <- setdiff(expected_cols, colnames(df))
  if (length(missing_cols) > 0) {
    stop(sprintf(
      "[!] Kolom tidak ditemukan di file '%s' (periode %s):\n    Missing : %s",
      basename(filepath), period, paste(missing_cols, collapse = ", ")
    ))
  }

  df$datetime <- as.POSIXct(paste(df$date, df$time),
                             format = paste(DATE_FORMAT, "%H:%M"),
                             tz     = "Asia/Jakarta")

  df_clean <- na.omit(df[, REGIONS, drop = FALSE])
  df_clean <- as.data.frame(lapply(df_clean, as.numeric))

  cat(sprintf("Data %-8s loaded: %d baris x %d region  |  %s\n",
              period, nrow(df_clean), ncol(df_clean), basename(filepath)))
  return(df_clean)
}


load_data <- function(path_periode1, path_periode2, path_periode3) {
  paths <- list(period_1 = path_periode1,
                period_2 = path_periode2,
                period_3 = path_periode3)
  data_list <- list()
  for (period in PERIODS) {
    data_list[[period]] <- load_one_period(paths[[period]], period)
  }
  cat("\nSemua data berhasil dimuat.\n")
  return(data_list)
}


# =============================================================================
# [TAMBAHAN 1] BAGIAN 0B: UJI STASIONERITAS — AUGMENTED DICKEY-FULLER (ADF)
# =============================================================================
# Referensi metodologi:
#   Uji ADF menguji H0: deret mengandung unit root (tidak stasioner) vs
#   H1: deret stasioner. Orde lag augmentasi q dipilih oleh AIC dalam
#   rentang q = 1, ..., MAX_LAG_ADF, sesuai rekomendasi untuk data
#   frekuensi tinggi (Wei, 2006).
#
# Fungsi ini memenuhi permintaan Reviewer 1 Poin 3 (verifikasi stasioneritas).
# =============================================================================

#' Pilih orde lag optimal untuk regresi ADF menggunakan AIC
#'
#' Regresi ADF:
#'   Delta(Z_t) = alpha + delta*Z_{t-1} + sum_{j=1}^{q} gamma_j Delta(Z_{t-j}) + eps
#'
#' AIC dihitung pada model unrestricted (termasuk Z_{t-1}) untuk setiap q.
#' q* = argmin_{q in 1..max_q} AIC(q)
#'
#' @param series  numeric vector, deret waktu
#' @param max_q   integer, batas atas orde lag
#' @return integer, orde lag optimal q*
select_adf_lag_aic <- function(series, max_q = MAX_LAG_ADF) {
  n    <- length(series)
  dy   <- diff(series)
  n_dy <- length(dy)

  # Batasi max_q agar tidak melebihi batas praktis
  max_q <- min(max_q, floor(n_dy / 5), n_dy - 3)
  if (max_q < 1) return(1L)

  aic_vals <- rep(Inf, max_q)

  for (q in 1:max_q) {
    n_eff <- n_dy - q
    if (n_eff < q + 3) next

    # Variabel dependen: Delta(Z_t), t = q+2, ..., n
    y_dep  <- dy[(q + 1):n_dy]
    # Lag level: Z_{t-1}
    y_lev  <- series[(q + 1):(n - 1)]
    # Lag differences: Delta(Z_{t-1}), ..., Delta(Z_{t-q})
    lag_dy <- matrix(NA_real_, nrow = n_eff, ncol = q)
    for (j in 1:q) {
      lag_dy[, j] <- dy[(q + 1 - j):(n_dy - j)]
    }

    X <- cbind(1, y_lev, lag_dy)           # intercept + level + augmentation

    fit <- tryCatch(
      .lm.fit(X, y_dep),
      error = function(e) NULL
    )
    if (is.null(fit)) next

    rss         <- sum(fit$residuals^2)
    k_params    <- ncol(X)                  # jumlah parameter
    aic_vals[q] <- n_eff * log(rss / n_eff) + 2 * k_params
  }

  return(which.min(aic_vals))
}


#' Hitung uji ADF untuk satu deret, mengembalikan statistik dan p-value
#'
#' Menggunakan tseries::adf.test() dengan orde lag q* yang dipilih oleh AIC.
#' H0: deret tidak stasioner (mengandung unit root)
#' H1: deret stasioner
#'
#' @param series  numeric vector
#' @param max_q   integer, batas atas lag augmentasi
#' @return list: statistic, p.value, lag_q, reject_h0
run_adf_one_series <- function(series, max_q = MAX_LAG_ADF) {
  q_star <- select_adf_lag_aic(series, max_q)

  result <- tryCatch(
    tseries::adf.test(series, k = q_star),
    error = function(e) NULL
  )

  if (is.null(result)) {
    return(list(statistic = NA_real_, p.value = NA_real_,
                lag_q = q_star, reject_h0 = NA))
  }

  list(
    statistic = as.numeric(result$statistic),
    p.value   = result$p.value,
    lag_q     = q_star,
    reject_h0 = result$p.value < ALPHA    # TRUE = stasioner
  )
}


#' Hitung uji ADF untuk semua 6 region pada satu periode
#'
#' @param df_period  data.frame, data satu periode (data mentah, sebelum standardisasi)
#' @param period     string, nama periode
#' @return data.frame: Region, ADF_Statistic, P_Value, Lag_q, Stasioner
compute_adf_test <- function(df_period, period) {
  cat(sprintf("  Menghitung uji ADF untuk periode %s...\n", toupper(period)))

  rows <- lapply(REGIONS, function(reg) {
    res <- run_adf_one_series(df_period[[reg]])
    data.frame(
      Periode       = toupper(period),
      Region        = reg,
      ADF_Statistic = round(res$statistic, 4),
      P_Value       = round(res$p.value, 6),
      Lag_q_AIC     = res$lag_q,
      Stasioner     = res$reject_h0,            # TRUE = tolak H0 = stasioner
      Kesimpulan    = ifelse(isTRUE(res$reject_h0), "Stasioner", "Tidak Stasioner"),
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, rows)
}


#' Cetak ringkasan hasil ADF ke console
#'
#' @param adf_df   data.frame, output compute_adf_test()
#' @param period   string, nama periode
summarize_adf <- function(adf_df, period) {
  cat(sprintf("\n%s\n", strrep("=", 70)))
  cat(sprintf("  HASIL UJI ADF — PERIODE %s\n", toupper(period)))
  cat(sprintf("  H0: deret mengandung unit root (tidak stasioner)\n"))
  cat(sprintf("  Tingkat signifikansi: alpha = %.2f\n", ALPHA))
  cat(sprintf("  Pemilihan lag augmentasi: AIC (batas atas = %d)\n", MAX_LAG_ADF))
  cat(sprintf("%s\n", strrep("=", 70)))
  cat(sprintf("  %-12s %12s %10s %6s %s\n",
              "Region", "ADF Stat", "p-value", "Lag q*", "Kesimpulan"))
  cat(sprintf("  %s\n", strrep("-", 60)))

  for (k in seq_len(nrow(adf_df))) {
    cat(sprintf("  %-12s %12.4f %10.6f %6d  %s\n",
                adf_df$Region[k],
                adf_df$ADF_Statistic[k],
                adf_df$P_Value[k],
                adf_df$Lag_q_AIC[k],
                adf_df$Kesimpulan[k]))
  }

  n_stasioner <- sum(adf_df$Stasioner, na.rm = TRUE)
  cat(sprintf("  %s\n", strrep("-", 60)))
  cat(sprintf("  Stasioner: %d / %d region (p < %.2f)\n",
              n_stasioner, nrow(adf_df), ALPHA))
  cat(sprintf("%s\n\n", strrep("=", 70)))
}


#' Export hasil ADF ke CSV
#'
#' @param adf_df     data.frame, output compute_adf_test()
#' @param period     string, nama periode
#' @param output_dir string, direktori output
export_adf_to_csv <- function(adf_df, period, output_dir = OUTPUT_DIR) {
  out_path <- file.path(output_dir, sprintf("adf_test_%s.csv", tolower(period)))
  write.csv(adf_df, out_path, row.names = FALSE)
  cat(sprintf("  Hasil ADF tersimpan : %s\n", out_path))
  invisible(adf_df)
}


#' Buat tabel ringkasan ADF lintas semua periode (untuk Tabel 2a naskah)
#'
#' @param adf_results named list — key: period, value: output compute_adf_test()
#' @param output_dir  string
export_adf_summary_table <- function(adf_results, output_dir = OUTPUT_DIR) {
  all_df <- do.call(rbind, adf_results)

  # Pivot: baris = Region, kolom = Periode (statistik + p-value)
  summary_tbl <- data.frame(Region = REGIONS)
  for (period in PERIODS) {
    sub <- adf_results[[period]]
    summary_tbl[[paste0(toupper(period), "_ADF")]]  <- sub$ADF_Statistic
    summary_tbl[[paste0(toupper(period), "_pval")]] <- sub$P_Value
    summary_tbl[[paste0(toupper(period), "_lag")]]  <- sub$Lag_q_AIC
    summary_tbl[[paste0(toupper(period), "_kes")]]  <- sub$Kesimpulan
  }

  out_path <- file.path(output_dir, "adf_summary_table2a.csv")
  write.csv(summary_tbl, out_path, row.names = FALSE)
  cat(sprintf("\n  Tabel Ringkasan ADF (Tabel 2a) tersimpan: %s\n", out_path))

  # Cetak ke console
  cat("\n  TABEL RINGKASAN ADF LINTAS SEMUA PERIODE (untuk naskah Tabel 2a):\n")
  cat(sprintf("  %-12s", "Region"))
  for (p in PERIODS) cat(sprintf("  %-28s", toupper(p)))
  cat("\n")
  cat(sprintf("  %-12s", ""))
  for (p in PERIODS) cat(sprintf("  %-14s %-14s", "ADF Stat (p)", "Lag q*"))
  cat("\n  ", strrep("-", 100), "\n", sep = "")

  for (i in seq_along(REGIONS)) {
    cat(sprintf("  %-12s", REGIONS[i]))
    for (p in PERIODS) {
      sub <- adf_results[[p]]
      row <- sub[sub$Region == REGIONS[i], ]
      cat(sprintf("  %8.4f (%7.5f) %4d          ",
                  row$ADF_Statistic, row$P_Value, row$Lag_q_AIC))
    }
    cat("\n")
  }

  invisible(summary_tbl)
}


# =============================================================================
# BAGIAN 1: CCF MULTI-LAG  (tidak berubah)
# =============================================================================

get_significance_bound <- function(n, alpha = ALPHA) {
  z <- qnorm(1 - alpha / 2)
  return(z / sqrt(n))
}


compute_ccf_matrix <- function(df_period, n_lags = N_LAGS) {
  df_std   <- as.data.frame(scale(df_period))
  n        <- nrow(df_std)
  ccf_list <- list()

  for (reg_i in REGIONS) {
    for (reg_j in REGIONS) {
      key      <- paste0(reg_i, "__", reg_j)
      ccf_vals <- numeric(n_lags + 1)
      x <- df_std[[reg_i]]
      y <- df_std[[reg_j]]

      for (k in 0:n_lags) {
        if (k == 0) {
          ccf_vals[k + 1] <- cor(x, y)
        } else {
          ccf_vals[k + 1] <- cor(x[(k+1):n], y[1:(n-k)])
        }
      }
      ccf_list[[key]] <- ccf_vals
    }
  }
  return(ccf_list)
}


plot_ccf_matrix <- function(ccf_list, n_obs, period, n_lags = N_LAGS,
                            save_path = NULL) {
  bound <- get_significance_bound(n_obs)
  lags  <- 0:n_lags
  m     <- length(REGIONS)
  plot_list <- list()

  for (i in seq_along(REGIONS)) {
    for (j in seq_along(REGIONS)) {
      reg_i <- REGIONS[i]; reg_j <- REGIONS[j]
      key   <- paste0(reg_i, "__", reg_j)
      vals  <- ccf_list[[key]]
      df_plot <- data.frame(lag = lags, ccf = vals, sig = abs(vals) > bound)
      is_diag <- (reg_i == reg_j)
      title   <- if (is_diag) paste0(reg_i, "\n(ACF)") else
                              paste0(reg_i, " \u2190 ", reg_j)

      p <- ggplot(df_plot, aes(x = lag, y = ccf, fill = sig)) +
        geom_bar(stat = "identity", width = 0.7, alpha = 0.85) +
        geom_hline(yintercept =  bound, linetype="dashed",
                   color="navy", linewidth=0.6) +
        geom_hline(yintercept = -bound, linetype="dashed",
                   color="navy", linewidth=0.6) +
        geom_hline(yintercept = 0, color="black", linewidth=0.3) +
        scale_fill_manual(values=c("FALSE"="#90A4AE","TRUE"="#D32F2F")) +
        scale_y_continuous(limits=c(-1,1)) +
        labs(title=title, x=NULL, y=NULL) +
        theme_minimal(base_size=7) +
        theme(legend.position="none",
              plot.title=element_text(size=6.5, hjust=0.5,
                face=if(is_diag)"bold" else "plain",
                color=if(is_diag)"#1565C0" else "#37474F"),
              plot.background=element_rect(
                fill=if(is_diag)"#E3F2FD" else "white", color=NA),
              panel.grid.minor=element_blank(),
              axis.text=element_text(size=5.5))

      plot_list[[(i-1)*m+j]] <- p
    }
  }

  combined <- wrap_plots(plot_list, nrow=m, ncol=m) +
    plot_annotation(
      title    = sprintf("Cross-Correlation Matrix Function (CCF) — %s", toupper(period)),
      subtitle = sprintf("Batas signifikansi: \u00b1%.3f  (\u03b1=%.2f)", bound, ALPHA),
      theme    = theme(plot.title=element_text(size=14, face="bold", hjust=0.5),
                       plot.subtitle=element_text(size=10, hjust=0.5, color="gray40"))
    )

  if (!is.null(save_path)) {
    ggsave(save_path, combined, width=22, height=20, dpi=150)
    cat(sprintf("  CCF plot disimpan: %s\n", save_path))
  }
  print(combined)
  invisible(combined)
}


extract_significant_ccf <- function(ccf_list, n_obs, n_lags = N_LAGS) {
  bound   <- get_significance_bound(n_obs)
  results <- list()
  for (reg_i in REGIONS) {
    for (reg_j in REGIONS) {
      if (reg_i == reg_j) next
      key      <- paste0(reg_i, "__", reg_j)
      vals     <- ccf_list[[key]]
      lags_sig <- which(abs(vals[2:(n_lags+1)]) > bound)
      if (length(lags_sig) > 0) {
        results[[length(results)+1]] <- data.frame(
          effect = reg_i, cause = reg_j,
          lag_signifikan = paste(lags_sig, collapse=", "),
          stringsAsFactors = FALSE)
      }
    }
  }
  if (length(results)==0) return(data.frame())
  do.call(rbind, results)
}


summarize_ccf <- function(sig_df, period) {
  cat(sprintf("\n%s\n  RINGKASAN CCF SIGNIFIKAN — %s\n%s\n",
              strrep("=",65), toupper(period), strrep("=",65)))
  if (nrow(sig_df)==0) {
    cat("  Tidak ada cross-regional CCF yang signifikan.\n")
  } else {
    cat(sprintf("  %-18s %-18s %s\n","Effect","Cause","Lag Signifikan"))
    cat(sprintf("  %s\n", strrep("-",60)))
    for (k in 1:nrow(sig_df))
      cat(sprintf("  %-18s %-18s [%s]\n",
                  sig_df$effect[k], sig_df$cause[k], sig_df$lag_signifikan[k]))
  }
  cat(sprintf("%s\n\n", strrep("=",65)))
}


# =============================================================================
# BAGIAN 2: PLCMF  (tidak berubah)
# =============================================================================

compute_autocovariance <- function(df_period, max_lag) {
  Z     <- as.matrix(scale(df_period))
  n     <- nrow(Z); m <- ncol(Z)
  Gamma <- vector("list", max_lag + 1)
  for (k in 0:max_lag) {
    G <- matrix(0, m, m)
    for (t in (k+1):n) G <- G + outer(Z[t,], Z[t-k,])
    Gamma[[k+1]] <- G / n
  }
  Gamma
}


compute_plcmf <- function(df_period, max_lag = N_LAGS) {
  Gamma <- compute_autocovariance(df_period, max_lag)
  m     <- ncol(df_period)
  P_kk_list <- list(); Sigma_list <- list()

  G0 <- Gamma[[1]]; G1 <- Gamma[[2]]; G0_inv <- solve(G0)
  Phi_fwd <- list(); Phi_bwd <- list()
  Phi_fwd[[1]] <- G1 %*% G0_inv
  Phi_bwd[[1]] <- t(G1) %*% G0_inv
  Sigma  <- G0 - Phi_fwd[[1]] %*% G0 %*% t(Phi_fwd[[1]])
  Sigma_ <- G0 - Phi_bwd[[1]] %*% G0 %*% t(Phi_bwd[[1]])
  d_fwd  <- sqrt(pmax(diag(Sigma), 1e-12)); d_G0 <- sqrt(diag(G0))
  P_kk_list[[1]]  <- diag(1/d_fwd) %*% Phi_fwd[[1]] %*% diag(d_G0)
  Sigma_list[[1]] <- Sigma

  for (k in 2:max_lag) {
    Sigma_inv  <- tryCatch(solve(Sigma),  error=function(e) NULL)
    Sigma_inv_ <- tryCatch(solve(Sigma_), error=function(e) NULL)
    if (is.null(Sigma_inv)||is.null(Sigma_inv_)) {
      cat(sprintf("  [!] Matriks singular pada lag %d.\n", k)); break
    }
    sum_fwd <- matrix(0,m,m); sum_bwd <- matrix(0,m,m)
    for (j in 1:(k-1)) {
      sum_fwd <- sum_fwd + Phi_fwd[[j]] %*% Gamma[[k-j+1]]
      sum_bwd <- sum_bwd + Phi_bwd[[j]] %*% t(Gamma[[k-j+1]])
    }
    new_Phi_kk  <- (Gamma[[k+1]]    - sum_fwd) %*% Sigma_inv_
    new_Phi_kk_ <- (t(Gamma[[k+1]]) - sum_bwd) %*% Sigma_inv
    new_Phi_fwd <- list(); new_Phi_bwd <- list()
    for (j in 1:(k-1)) {
      new_Phi_fwd[[j]] <- Phi_fwd[[j]] - new_Phi_kk  %*% Phi_bwd[[k-j]]
      new_Phi_bwd[[j]] <- Phi_bwd[[j]] - new_Phi_kk_ %*% Phi_fwd[[k-j]]
    }
    new_Phi_fwd[[k]] <- new_Phi_kk; new_Phi_bwd[[k]] <- new_Phi_kk_
    Phi_fwd <- new_Phi_fwd; Phi_bwd <- new_Phi_bwd
    Sigma_new  <- Sigma  - new_Phi_kk  %*% Sigma_ %*% t(new_Phi_kk)
    Sigma_new_ <- Sigma_ - new_Phi_kk_ %*% Sigma  %*% t(new_Phi_kk_)
    d_f <- sqrt(pmax(diag(Sigma_new), 1e-12))
    d_b <- sqrt(pmax(diag(Sigma_new_),1e-12))
    P_kk_list[[k]]  <- diag(1/d_f) %*% new_Phi_kk %*% diag(d_b)
    Sigma_list[[k]] <- Sigma_new
    Sigma  <- Sigma_new; Sigma_ <- Sigma_new_
  }
  list(P_kk_list=P_kk_list, Sigma_list=Sigma_list)
}


test_plcmf_significance <- function(P_kk_list, n_obs) {
  bound <- get_significance_bound(n_obs)
  sig_list <- list(); pval_list <- list()
  for (k in seq_along(P_kk_list)) {
    P_kk   <- P_kk_list[[k]]
    z_stat <- P_kk * sqrt(n_obs)
    pvals  <- 2 * (1 - pnorm(abs(z_stat)))
    sig    <- abs(P_kk) > bound
    rownames(sig) <- colnames(sig) <- REGIONS
    rownames(pvals) <- colnames(pvals) <- REGIONS
    sig_list[[k]]  <- sig
    pval_list[[k]] <- pvals
  }
  list(sig_list=sig_list, pval_list=pval_list)
}


plot_plcmf_heatmap <- function(P_kk_list, sig_list, period, n_obs,
                               max_display_lag=12, save_path=NULL) {
  bound     <- get_significance_bound(n_obs)
  lags_show <- min(max_display_lag, length(P_kk_list))
  plot_list <- list()
  for (k in 1:lags_show) {
    P_kk <- P_kk_list[[k]]; sig <- sig_list[[k]]
    rownames(P_kk) <- colnames(P_kk) <- REGIONS
    df_melt       <- melt(P_kk); df_sig <- melt(sig)
    df_melt$sig   <- df_sig$value
    df_melt$label <- ifelse(df_melt$sig,
                            paste0(round(df_melt$value,2),"*"),
                            round(df_melt$value,2))
    df_melt$Var1  <- factor(df_melt$Var1, levels=rev(REGIONS))
    df_melt$Var2  <- factor(df_melt$Var2, levels=REGIONS)

    p <- ggplot(df_melt, aes(x=Var2, y=Var1, fill=value)) +
      geom_tile(color="white", linewidth=0.4) +
      geom_text(aes(label=label,
                    color=abs(value)>0.5,
                    fontface=ifelse(sig,"bold","plain")), size=2.2) +
      scale_fill_gradient2(low="#1565C0",mid="white",high="#C62828",
                           midpoint=0,limits=c(-1,1),name="P_kk") +
      scale_color_manual(values=c("TRUE"="white","FALSE"="black"),guide="none") +
      labs(title=sprintf("Lag %d",k), x=NULL, y=NULL) +
      theme_minimal(base_size=7) +
      theme(axis.text.x=element_text(angle=45,hjust=1,size=6),
            axis.text.y=element_text(size=6),
            plot.title=element_text(size=8,face="bold",hjust=0.5),
            legend.position="none")
    plot_list[[k]] <- p
  }
  n_col <- 4; n_row <- ceiling(lags_show/n_col)
  combined <- wrap_plots(plot_list, nrow=n_row, ncol=n_col) +
    plot_annotation(
      title    = sprintf("PLCMF Heatmap — %s", toupper(period)),
      subtitle = sprintf("* = signifikan (\u03b1=%.2f, bound=\u00b1%.3f)", ALPHA, bound),
      theme    = theme(plot.title=element_text(size=13,face="bold",hjust=0.5),
                       plot.subtitle=element_text(size=9,hjust=0.5,color="gray40"))
    )
  if (!is.null(save_path)) {
    ggsave(save_path, combined, width=20, height=n_row*4, dpi=150)
    cat(sprintf("  PLCMF heatmap disimpan: %s\n", save_path))
  }
  print(combined); invisible(combined)
}


plot_plcmf_lines <- function(P_kk_list, sig_list, period, n_obs,
                             save_path=NULL) {
  bound <- get_significance_bound(n_obs)
  lags_show <- length(P_kk_list); m <- length(REGIONS)
  plot_list <- list()
  for (i in seq_along(REGIONS)) {
    for (j in seq_along(REGIONS)) {
      reg_i <- REGIONS[i]; reg_j <- REGIONS[j]
      vals  <- sapply(1:lags_show, function(k) P_kk_list[[k]][i,j])
      sigs  <- sapply(1:lags_show, function(k) sig_list[[k]][i,j])
      df_p  <- data.frame(lag=1:lags_show, val=vals, sig=sigs)
      is_diag <- (i==j)
      title   <- if(is_diag) paste0(reg_i,"\n(own)") else
                             paste0(reg_i,"\u2190",reg_j)

      p <- ggplot(df_p, aes(x=lag, y=val, fill=sig)) +
        geom_bar(stat="identity", width=0.65, alpha=0.85) +
        geom_hline(yintercept= bound, linetype="dashed",
                   color="navy", linewidth=0.5) +
        geom_hline(yintercept=-bound, linetype="dashed",
                   color="navy", linewidth=0.5) +
        geom_hline(yintercept=0, color="black", linewidth=0.3) +
        scale_fill_manual(values=c("FALSE"="#B0BEC5","TRUE"="#C62828")) +
        scale_y_continuous(limits=c(-1,1)) +
        labs(title=title, x=NULL, y=NULL) +
        theme_minimal(base_size=6.5) +
        theme(legend.position="none",
              plot.title=element_text(size=6,hjust=0.5,
                face=if(is_diag)"bold" else "plain",
                color=if(is_diag)"#2E7D32" else "#37474F"),
              plot.background=element_rect(
                fill=if(is_diag)"#E8F5E9" else "white",color=NA),
              panel.grid.minor=element_blank(),
              axis.text=element_text(size=5))
      plot_list[[(i-1)*m+j]] <- p
    }
  }
  combined <- wrap_plots(plot_list, nrow=m, ncol=m) +
    plot_annotation(
      title    = sprintf("PLCMF per Elemen vs Lag — %s", toupper(period)),
      subtitle = sprintf("Garis putus-putus = batas signifikansi \u00b1%.3f", bound),
      theme    = theme(plot.title=element_text(size=13,face="bold",hjust=0.5),
                       plot.subtitle=element_text(size=9,hjust=0.5,color="gray40"))
    )
  if (!is.null(save_path)) {
    ggsave(save_path, combined, width=22, height=20, dpi=150)
    cat(sprintf("  PLCMF line plot disimpan: %s\n", save_path))
  }
  print(combined); invisible(combined)
}


extract_significant_plcmf <- function(P_kk_list, sig_list) {
  results <- list()
  for (k in seq_along(P_kk_list)) {
    sig <- sig_list[[k]]
    for (i in seq_along(REGIONS)) {
      for (j in seq_along(REGIONS)) {
        if (i!=j && sig[i,j]) {
          results[[length(results)+1]] <- data.frame(
            effect=REGIONS[i], cause=REGIONS[j], lag=k,
            stringsAsFactors=FALSE)
        }
      }
    }
  }
  if (length(results)==0) return(data.frame())
  df <- do.call(rbind, results)
  as.data.frame(df %>% group_by(effect,cause) %>%
    summarise(lag_signifikan=paste(lag,collapse=", "), .groups="drop"))
}


summarize_plcmf <- function(sig_df, period) {
  cat(sprintf("\n%s\n  RINGKASAN PLCMF SIGNIFIKAN — %s\n%s\n",
              strrep("=",65), toupper(period), strrep("=",65)))
  if (nrow(sig_df)==0) {
    cat("  Tidak ada cross-regional partial dependency yang signifikan.\n")
  } else {
    cat(sprintf("  %-18s %-18s %s\n","Effect","Cause","Lag Signifikan (Parsial)"))
    cat(sprintf("  %s\n", strrep("-",62)))
    for (k in 1:nrow(sig_df))
      cat(sprintf("  %-18s %-18s [%s]\n",
                  sig_df$effect[k], sig_df$cause[k], sig_df$lag_signifikan[k]))
  }
  cat(sprintf("%s\n\n", strrep("=",65)))
}


export_plcmf_to_csv <- function(P_kk_list, sig_list, n_obs, period,
                                output_dir=OUTPUT_DIR) {
  bound <- get_significance_bound(n_obs); rows <- list()
  for (k in seq_along(P_kk_list)) {
    P_mat <- P_kk_list[[k]]; S_mat <- sig_list[[k]]
    for (i in seq_along(REGIONS)) {
      for (j in seq_along(REGIONS)) {
        rows[[length(rows)+1]] <- data.frame(
          Periode=toupper(period), Lag=k,
          Effect=REGIONS[i], Cause=REGIONS[j],
          PLCMF_Value=round(P_mat[i,j],6),
          Sig_PLCMF=S_mat[i,j], Bound=round(bound,6),
          stringsAsFactors=FALSE)
      }
    }
  }
  df_out   <- do.call(rbind, rows)
  out_path <- file.path(output_dir, sprintf("plcmf_values_%s.csv", tolower(period)))
  write.csv(df_out, out_path, row.names=FALSE)
  cat(sprintf("  Nilai PLCMF tersimpan : %s  (%d baris)\n", out_path, nrow(df_out)))
  invisible(df_out)
}


# =============================================================================
# [TAMBAHAN 2] BAGIAN 2B: KOREKSI BH-FDR PADA ELEMEN PLCMF
# =============================================================================
# Referensi metodologi:
#   Benjamini & Hochberg (1995) FDR dipilih atas Bonferroni karena elemen
#   PLCMF pada lag berdekatan berbagi conditioning set melalui rekursi
#   Whittle, menginduksi dependensi positif antar statistik uji.
#   Prosedur BH mengontrol FDR pada tingkat nominal alpha di bawah
#   dependensi positif (properti PRDS, Benjamini & Yekutieli, 2001).
#
#   Koreksi diterapkan pada seluruh elemen OFF-DIAGONAL matriks PLCMF
#   (30 pasang × N_LAGS lag) per periode secara terpisah.
# =============================================================================

#' Terapkan koreksi BH-FDR pada semua p-value elemen PLCMF off-diagonal
#'
#' Mengumpulkan semua p-value dari 30 pasang off-diagonal × N_LAGS lag,
#' mengurutkannya, dan menerapkan prosedur Benjamini-Hochberg.
#'
#' @param pval_list  list of matrix (m×m), p-value per lag, output test_plcmf_significance()
#' @param n_obs      integer, jumlah observasi
#' @param period     string, nama periode
#' @return list:
#'   - sig_bh_list : list of logical matrix (m×m) — signifikansi setelah koreksi BH
#'   - summary_df  : data.frame ringkasan per lag
#'   - overall     : list ringkasan keseluruhan
apply_bh_fdr_plcmf <- function(pval_list, n_obs, period) {
  bound  <- get_significance_bound(n_obs)
  m      <- length(REGIONS)
  K      <- length(pval_list)

  # --- Kumpulkan semua p-value off-diagonal + identitasnya ---
  all_pvals <- c()
  all_idx   <- list()    # untuk memetakan kembali ke (k, i, j)

  for (k in seq_len(K)) {
    P_mat <- pval_list[[k]]
    for (i in seq_len(m)) {
      for (j in seq_len(m)) {
        if (i == j) next           # lewati diagonal
        all_pvals           <- c(all_pvals, P_mat[i, j])
        all_idx[[length(all_idx) + 1]] <- c(k = k, i = i, j = j)
      }
    }
  }

  n_tests <- length(all_pvals)

  # --- Terapkan BH-FDR ---
  # p.adjust dengan method="BH" mengimplementasikan prosedur Benjamini-Hochberg
  pvals_adj   <- p.adjust(all_pvals, method = "BH")
  reject_raw  <- all_pvals  < ALPHA
  reject_bh   <- pvals_adj  < ALPHA

  # --- Rekonstruksi matriks signifikansi yang sudah dikoreksi ---
  sig_bh_list <- lapply(seq_len(K), function(k) {
    mat <- matrix(FALSE, m, m, dimnames = list(REGIONS, REGIONS))
    mat
  })

  for (idx_flat in seq_along(all_idx)) {
    k <- all_idx[[idx_flat]]["k"]
    i <- all_idx[[idx_flat]]["i"]
    j <- all_idx[[idx_flat]]["j"]
    sig_bh_list[[k]][i, j] <- reject_bh[idx_flat]
  }

  # --- Ringkasan per lag ---
  summary_rows <- lapply(seq_len(K), function(k) {
    start <- (k - 1) * (m * m - m) + 1
    end   <- k       * (m * m - m)
    start <- (k-1)*30 + 1; end <- k*30   # 30 = m*(m-1) off-diagonal
    idx_k <- which(sapply(all_idx, function(x) x["k"]) == k)
    n_sig_raw <- sum(reject_raw[idx_k])
    n_sig_bh  <- sum(reject_bh[idx_k])
    data.frame(
      Periode         = toupper(period),
      Lag             = k,
      N_Uji           = length(idx_k),       # = 30 (off-diagonal)
      Sig_Tanpa_Koreks = n_sig_raw,
      Sig_Setelah_BH  = n_sig_bh,
      Pct_Dipertahan  = ifelse(n_sig_raw > 0,
                               round(100 * n_sig_bh / n_sig_raw, 1), NA_real_),
      stringsAsFactors = FALSE
    )
  })
  summary_df <- do.call(rbind, summary_rows)

  # --- Ringkasan keseluruhan ---
  overall <- list(
    n_total_tests    = n_tests,
    n_sig_tanpa_kor  = sum(reject_raw),
    n_sig_setelah_bh = sum(reject_bh),
    pct_dipertahan   = round(100 * sum(reject_bh) / max(sum(reject_raw), 1), 1)
  )

  list(sig_bh_list   = sig_bh_list,
       pvals_adjusted = pvals_adj,
       reject_raw    = reject_raw,
       reject_bh     = reject_bh,
       summary_df    = summary_df,
       overall       = overall,
       n_tests       = n_tests)
}


#' Cetak ringkasan hasil koreksi BH-FDR ke console
#'
#' @param bh_result  list, output apply_bh_fdr_plcmf()
#' @param period     string, nama periode
summarize_bh_fdr <- function(bh_result, period) {
  ov <- bh_result$overall
  cat(sprintf("\n%s\n", strrep("=", 72)))
  cat(sprintf("  KOREKSI BH-FDR PADA ELEMEN PLCMF — PERIODE %s\n", toupper(period)))
  cat(sprintf("  Metode: Benjamini-Hochberg FDR  |  alpha = %.2f\n", ALPHA))
  cat(sprintf("  Jumlah uji: %d (30 off-diagonal × %d lag)\n",
              ov$n_total_tests, N_LAGS))
  cat(sprintf("%s\n", strrep("=", 72)))
  cat(sprintf("  Total signifikan TANPA koreksi : %d\n", ov$n_sig_tanpa_kor))
  cat(sprintf("  Total signifikan SETELAH BH    : %d\n", ov$n_sig_setelah_bh))
  cat(sprintf("  Persentase dipertahankan       : %.1f%%\n", ov$pct_dipertahan))
  cat(sprintf("\n  %-6s %8s %12s %12s %12s\n",
              "Lag", "N Uji", "Tanpa Kor.", "Setelah BH", "% Pertahan"))
  cat(sprintf("  %s\n", strrep("-", 58)))
  for (k in seq_len(nrow(bh_result$summary_df))) {
    row <- bh_result$summary_df[k, ]
    cat(sprintf("  %-6d %8d %12d %12d %11s%%\n",
                row$Lag, row$N_Uji,
                row$Sig_Tanpa_Koreks, row$Sig_Setelah_BH,
                ifelse(is.na(row$Pct_Dipertahan), "  —",
                       sprintf("%.1f", row$Pct_Dipertahan))))
  }
  cat(sprintf("%s\n\n", strrep("=", 72)))
}


#' Export hasil BH-FDR ke CSV (dua file: per-lag summary + detail per elemen)
#'
#' @param bh_result   list, output apply_bh_fdr_plcmf()
#' @param pval_list   list of matrix, p-value asli (tanpa koreksi)
#' @param P_kk_list   list of matrix, nilai PLCMF
#' @param period      string
#' @param output_dir  string
export_bh_fdr_to_csv <- function(bh_result, pval_list, P_kk_list, period,
                                 output_dir = OUTPUT_DIR) {
  # --- File 1: ringkasan per lag ---
  path1 <- file.path(output_dir, sprintf("bh_fdr_summary_%s.csv", tolower(period)))
  write.csv(bh_result$summary_df, path1, row.names = FALSE)
  cat(sprintf("  Ringkasan BH-FDR tersimpan : %s\n", path1))

  # --- File 2: detail per elemen (untuk Tabel 3 naskah) ---
  m    <- length(REGIONS)
  rows <- list()
  pval_adj_vec <- bh_result$pvals_adjusted
  idx_flat     <- 0L

  for (k in seq_along(P_kk_list)) {
    P_mat  <- P_kk_list[[k]]
    pv_mat <- pval_list[[k]]
    bh_mat <- bh_result$sig_bh_list[[k]]

    for (i in seq_len(m)) {
      for (j in seq_len(m)) {
        if (i == j) next
        idx_flat <- idx_flat + 1L
        rows[[length(rows) + 1]] <- data.frame(
          Periode      = toupper(period),
          Lag          = k,
          Effect       = REGIONS[i],
          Cause        = REGIONS[j],
          PLCMF_Value  = round(P_mat[i, j], 6),
          PVal_Raw     = round(pv_mat[i, j], 6),
          PVal_BH_Adj  = round(pval_adj_vec[idx_flat], 6),
          Sig_Raw      = pv_mat[i, j] < ALPHA,
          Sig_BH       = bh_mat[i, j],
          stringsAsFactors = FALSE
        )
      }
    }
  }

  df_detail <- do.call(rbind, rows)
  path2 <- file.path(output_dir, sprintf("bh_fdr_detail_%s.csv", tolower(period)))
  write.csv(df_detail, path2, row.names = FALSE)
  cat(sprintf("  Detail BH-FDR tersimpan    : %s  (%d baris)\n",
              path2, nrow(df_detail)))
  invisible(list(summary = bh_result$summary_df, detail = df_detail))
}


#' Buat plot perbandingan signifikansi sebelum vs sesudah BH-FDR per lag
#'
#' @param bh_result  list, output apply_bh_fdr_plcmf()
#' @param period     string
#' @param save_path  string atau NULL
plot_bh_comparison <- function(bh_result, period, save_path = NULL) {
  df_plot <- bh_result$summary_df %>%
    tidyr::pivot_longer(cols = c("Sig_Tanpa_Koreks", "Sig_Setelah_BH"),
                        names_to = "Metode", values_to = "N_Signifikan") %>%
    dplyr::mutate(
      Metode = dplyr::recode(Metode,
        "Sig_Tanpa_Koreks" = "Tanpa Koreksi",
        "Sig_Setelah_BH"   = "Setelah BH-FDR")
    )

  p <- ggplot(df_plot, aes(x = Lag, y = N_Signifikan, fill = Metode)) +
    geom_bar(stat = "identity", position = "dodge", width = 0.7, alpha = 0.85) +
    geom_hline(yintercept = 30, linetype = "dashed",
               color = "gray40", linewidth = 0.5) +
    annotate("text", x = N_LAGS - 0.5, y = 31, label = "Maks. 30 pasang",
             size = 3, color = "gray40") +
    scale_fill_manual(values = c("Tanpa Koreksi" = "#EF5350",
                                 "Setelah BH-FDR" = "#1565C0")) +
    scale_x_continuous(breaks = 1:N_LAGS) +
    labs(
      title    = sprintf("Perbandingan Signifikansi PLCMF: Tanpa vs Setelah BH-FDR — %s",
                         toupper(period)),
      subtitle = sprintf("Total uji: %d (30 off-diagonal × %d lag)  |  alpha = %.2f",
                         bh_result$n_tests, N_LAGS, ALPHA),
      x = "Lag", y = "Jumlah Elemen Signifikan (off-diagonal)",
      fill = "Metode"
    ) +
    theme_minimal(base_size = 11) +
    theme(plot.title    = element_text(face = "bold", hjust = 0.5, size = 12),
          plot.subtitle = element_text(hjust = 0.5, color = "gray40", size = 9),
          legend.position = "bottom")

  if (!is.null(save_path)) {
    ggsave(save_path, p, width = 14, height = 6, dpi = 150)
    cat(sprintf("  Plot BH-FDR disimpan: %s\n", save_path))
  }
  print(p)
  invisible(p)
}


# =============================================================================
# BAGIAN 3: GRANGER CAUSALITY — [DIREVISI] BERBASIS AIC
# =============================================================================
# Perubahan dari versi sebelumnya:
#   SEBELUM : Menguji p=1,...,10 dan mengambil minimum p-value. Pendekatan ini
#             menggelembungkan kesalahan Tipe I karena secara efektif melakukan
#             10 pengujian per pasang dan memilih hasil paling ekstrem.
#             P(setidaknya 1 false positive | 10 uji independen) = 1-(0.95)^10 ≈ 0.40
#
#   SEKARANG: Pilih orde lag optimal p* menggunakan AIC pada model unrestricted.
#             Lakukan SATU uji F tunggal pada p*. Pendekatan ini mengikuti
#             rekomendasi standar econometric (Lütkepohl, 2005) dan menghilangkan
#             inflasi Tipe I dari prosedur sebelumnya.
# =============================================================================

#' Pilih orde lag optimal untuk Granger Causality menggunakan AIC
#'
#' Model unrestricted (Persamaan 27):
#'   Z_it = alpha_0 + sum_{k=1}^{p} alpha_k Z_{i,t-k}
#'                 + sum_{k=1}^{p} beta_k  Z_{j,t-k} + eps
#'
#' AIC(p) = n_eff * log(RSS_U / n_eff) + 2*(2p + 1)
#' p*     = argmin_{p in 1..max_lag} AIC(p)
#'
#' @param y        numeric vector, deret target (effect / Z_it)
#' @param x        numeric vector, deret penyebab (cause / Z_jt)
#' @param max_lag  integer, batas atas orde lag
#' @return integer, p* optimal
select_granger_lag_aic <- function(y, x, max_lag = MAX_LAG_GC) {
  n        <- length(y)
  aic_vals <- rep(Inf, max_lag)

  for (p in 1:max_lag) {
    n_eff <- n - p
    if (n_eff < 2 * p + 2) next

    # Bangun matriks prediktor model unrestricted
    Y_dep <- y[(p + 1):n]

    own_lags  <- sapply(1:p, function(k) y[(p + 1 - k):(n - k)])
    cross_lags <- sapply(1:p, function(k) x[(p + 1 - k):(n - k)])
    if (p == 1) {
      own_lags   <- matrix(own_lags,   ncol = 1)
      cross_lags <- matrix(cross_lags, ncol = 1)
    }

    X_unrestr <- cbind(1, own_lags, cross_lags)

    fit <- tryCatch(
      .lm.fit(X_unrestr, Y_dep),
      error = function(e) NULL
    )
    if (is.null(fit)) next

    rss        <- sum(fit$residuals^2)
    k_params   <- 2 * p + 1             # intercept + p own + p cross
    aic_vals[p] <- n_eff * log(rss / n_eff) + 2 * k_params
  }

  return(which.min(aic_vals))
}


#' Uji Granger Causality berbasis AIC untuk semua pasangan region
#'
#' Untuk setiap pasang (i, j):
#'   1. Pilih p* = argmin_{p=1..MAX_LAG_GC} AIC pada model unrestricted
#'   2. Jalankan SATU uji F di p* menggunakan grangertest()
#'   3. Keputusan: tolak H0 jika p-value < ALPHA
#'
#' @param df_period  data.frame, data satu periode
#' @param max_lag    integer, batas atas orde lag
#' @return list:
#'   - pval_matrix  : data.frame m×m, p-value uji F di p*
#'   - pstar_matrix : data.frame m×m, orde lag optimal p*
#'   - result       : data.frame m×m, logical (TRUE = Granger-causes)
compute_granger_aic <- function(df_period, max_lag = MAX_LAG_GC) {
  m         <- length(REGIONS)
  pval_mat  <- matrix(NA_real_, m, m, dimnames = list(REGIONS, REGIONS))
  pstar_mat <- matrix(NA_integer_, m, m, dimnames = list(REGIONS, REGIONS))

  cat("  Menghitung Granger Causality (seleksi lag berbasis AIC)...\n")
  cat(sprintf("  Batas atas orde lag: p_max = %d  |  alpha = %.2f\n",
              max_lag, ALPHA))

  for (i in seq_along(REGIONS)) {
    for (j in seq_along(REGIONS)) {
      if (i == j) next

      y <- df_period[[REGIONS[i]]]   # deret yang DIPENGARUHI (effect)
      x <- df_period[[REGIONS[j]]]   # deret yang MEMPENGARUHI (cause)

      # 1. Pilih p* dengan AIC
      p_star <- select_granger_lag_aic(y, x, max_lag)
      pstar_mat[i, j] <- p_star

      # 2. Satu uji F tunggal di p*
      tryCatch({
        gt <- lmtest::grangertest(y ~ x, order = p_star)
        pval_mat[i, j] <- gt$`Pr(>F)`[2]
      }, error = function(e) {
        pval_mat[i, j] <<- NA_real_
        cat(sprintf("  [!] Error Granger %s→%s pada p*=%d: %s\n",
                    REGIONS[j], REGIONS[i], p_star, e$message))
      })
    }
  }

  pval_df  <- as.data.frame(pval_mat)
  pstar_df <- as.data.frame(pstar_mat)
  result_df <- as.data.frame(pval_mat < ALPHA)

  list(pval_matrix  = pval_df,
       pstar_matrix = pstar_df,
       result       = result_df)
}


#' Cetak ringkasan Granger Causality berbasis AIC
#'
#' @param granger_out  list, output compute_granger_aic()
#' @param period       string, nama periode
summarize_granger_aic <- function(granger_out, period) {
  pstar <- as.matrix(granger_out$pstar_matrix)
  pval  <- as.matrix(granger_out$pval_matrix)
  res   <- as.matrix(granger_out$result)
  m     <- length(REGIONS)

  cat(sprintf("\n%s\n", strrep("=", 72)))
  cat(sprintf("  GRANGER CAUSALITY (AIC-based) — PERIODE %s\n", toupper(period)))
  cat(sprintf("  Prosedur: seleksi orde lag AIC + satu uji F di p*\n"))
  cat(sprintf("%s\n", strrep("=", 72)))
  cat(sprintf("  %-14s %-14s %6s %10s %s\n",
              "Effect", "Cause", "p*", "p-value", "Sig."))
  cat(sprintf("  %s\n", strrep("-", 56)))

  for (i in seq_len(m)) {
    for (j in seq_len(m)) {
      if (i == j) next
      sig_mark <- if (!is.na(res[i,j]) && res[i,j]) "***" else ""
      cat(sprintf("  %-14s %-14s %6d %10.5f  %s\n",
                  REGIONS[i], REGIONS[j],
                  pstar[i, j], pval[i, j], sig_mark))
    }
  }

  n_sig <- sum(res, na.rm = TRUE)
  cat(sprintf("  %s\n", strrep("-", 56)))
  cat(sprintf("  Pasang signifikan: %d / %d  (*** = p < %.2f)\n",
              n_sig, m*(m-1), ALPHA))
  cat(sprintf("%s\n\n", strrep("=", 72)))
}


#' Buat tabel distribusi p* yang dipilih AIC (untuk Tabel 2b naskah)
#'
#' @param granger_results  named list — key: period, value: output compute_granger_aic()
#' @param output_dir       string
export_pstar_distribution_table <- function(granger_results,
                                            output_dir = OUTPUT_DIR) {
  rows <- list()
  for (period in names(granger_results)) {
    pstar_mat <- as.matrix(granger_results[[period]]$pstar_matrix)
    m <- nrow(pstar_mat)
    for (i in 1:m) {
      for (j in 1:m) {
        if (i == j) next
        rows[[length(rows)+1]] <- data.frame(
          Periode = toupper(period),
          Effect  = REGIONS[i],
          Cause   = REGIONS[j],
          p_star  = pstar_mat[i, j],
          stringsAsFactors = FALSE
        )
      }
    }
  }

  df_all <- do.call(rbind, rows)

  # Tabel distribusi frekuensi p* (untuk Tabel 2b di naskah)
  bins   <- c(1, 2, 3, "4-6", "7-10")
  tbl    <- data.frame(Lag_Order_p_star = bins)

  for (period in names(granger_results)) {
    sub <- df_all[df_all$Periode == toupper(period), "p_star"]
    tbl[[toupper(period)]] <- c(
      sum(sub == 1), sum(sub == 2), sum(sub == 3),
      sum(sub >= 4 & sub <= 6), sum(sub >= 7 & sub <= 10)
    )
  }
  tbl$Total <- rowSums(tbl[, -1])

  out_path <- file.path(output_dir, "pstar_distribution_table2b.csv")
  write.csv(tbl, out_path, row.names = FALSE)
  cat(sprintf("\n  Tabel Distribusi p* (Tabel 2b) tersimpan: %s\n", out_path))

  cat("\n  DISTRIBUSI ORDE LAG OPTIMAL p* (untuk naskah Tabel 2b):\n")
  print(tbl, row.names = FALSE)

  # Detail per pasang
  out_detail <- file.path(output_dir, "pstar_detail.csv")
  write.csv(df_all, out_detail, row.names = FALSE)
  cat(sprintf("  Detail p* per pasang tersimpan: %s\n", out_detail))

  invisible(tbl)
}


plot_granger <- function(granger_out, period, save_path = NULL) {
  pval_mat <- as.matrix(granger_out$pval_matrix)
  res_mat  <- as.matrix(granger_out$result)
  diag(pval_mat) <- NA; diag(res_mat) <- NA

  df_p <- melt(pval_mat, na.rm=FALSE)
  df_p$Var1  <- factor(df_p$Var1, levels=rev(REGIONS))
  df_p$Var2  <- factor(df_p$Var2, levels=REGIONS)
  df_p$label <- ifelse(is.na(df_p$value), "",
                       paste0(round(df_p$value,3),
                              ifelse(!is.na(df_p$value)&df_p$value<ALPHA,"*","")))

  p1 <- ggplot(df_p, aes(x=Var2,y=Var1,fill=value)) +
    geom_tile(color="white",linewidth=0.5) +
    geom_text(aes(label=label),size=2.8,
              color=ifelse(!is.na(df_p$value)&df_p$value<0.05,"white","black")) +
    scale_fill_gradientn(
      colors =c("#B71C1C","#EF5350","#FFCDD2","#E3F2FD","lightgray"),
      values =scales::rescale(c(0,0.01,0.05,0.1,NA)),
      limits=c(0,0.1), na.value="lightgray", name="p-value") +
    labs(title="P-Value Granger Causality (AIC-based)",
         subtitle=paste0("Merah = lebih signifikan, * = p < ",ALPHA),
         x="CAUSE", y="EFFECT") +
    theme_minimal(base_size=9) +
    theme(axis.text.x=element_text(angle=45,hjust=1,size=8),
          axis.text.y=element_text(size=8),
          plot.title=element_text(face="bold",hjust=0.5),
          plot.subtitle=element_text(hjust=0.5,color="gray40"))

  df_r <- melt(res_mat, na.rm=FALSE)
  df_r$Var1 <- factor(df_r$Var1, levels=rev(REGIONS))
  df_r$Var2 <- factor(df_r$Var2, levels=REGIONS)
  df_r$value <- as.numeric(df_r$value)
  df_r$icon  <- ifelse(is.na(df_r$value),"",
                       ifelse(df_r$value==1,"\u2713","\u2717"))

  p2 <- ggplot(df_r, aes(x=Var2,y=Var1,fill=factor(value))) +
    geom_tile(color="white",linewidth=0.5) +
    geom_text(aes(label=icon,color=factor(value)),size=5) +
    scale_fill_manual(values=c("0"="#ECEFF1","1"="#1B5E20"),
                      na.value="lightgray",name="",
                      labels=c("Tidak","Signifikan")) +
    scale_color_manual(values=c("0"="#607D8B","1"="white"),guide="none") +
    labs(title=sprintf("Peta Kausalitas Granger (\u03b1=%.2f)",ALPHA),
         subtitle="[p* dipilih AIC] \u2713=signifikan, \u2717=tidak",
         x="CAUSE", y="EFFECT") +
    theme_minimal(base_size=9) +
    theme(axis.text.x=element_text(angle=45,hjust=1,size=8),
          axis.text.y=element_text(size=8),
          plot.title=element_text(face="bold",hjust=0.5),
          plot.subtitle=element_text(hjust=0.5,color="gray40"),
          legend.position="bottom")

  combined <- (p1|p2) +
    plot_annotation(
      title=sprintf("Granger Causality (AIC-based) — %s", toupper(period)),
      theme=theme(plot.title=element_text(size=14,face="bold",hjust=0.5))
    )

  if (!is.null(save_path)) {
    ggsave(save_path, combined, width=16, height=7, dpi=150)
    cat(sprintf("  Granger plot disimpan: %s\n", save_path))
  }
  print(combined); invisible(combined)
}


export_granger_to_csv <- function(granger_out, period, output_dir=OUTPUT_DIR) {
  pval_mat  <- as.matrix(granger_out$pval_matrix)
  pstar_mat <- as.matrix(granger_out$pstar_matrix)
  rows <- list()
  for (i in seq_along(REGIONS)) {
    for (j in seq_along(REGIONS)) {
      if (i==j) next
      rows[[length(rows)+1]] <- data.frame(
        Periode        = toupper(period),
        Effect         = REGIONS[i],
        Cause          = REGIONS[j],
        Lag_pstar_AIC  = pstar_mat[i,j],
        Granger_PValue = round(pval_mat[i,j],6),
        Sig_Granger    = pval_mat[i,j] < ALPHA,
        stringsAsFactors=FALSE)
    }
  }
  df_out   <- do.call(rbind, rows)
  out_path <- file.path(output_dir,
                        sprintf("granger_aic_%s.csv", tolower(period)))
  write.csv(df_out, out_path, row.names=FALSE)
  cat(sprintf("  Nilai Granger (AIC) tersimpan: %s  (%d baris)\n",
              out_path, nrow(df_out)))
  invisible(df_out)
}


# =============================================================================
# BAGIAN 4: INTEGRASI FITUR GRU  (tidak berubah)
# =============================================================================

build_gru_features <- function(pacf_lags, plcmf_sig_df, granger_res, period) {
  results <- list()
  for (reg_i in REGIONS) {
    own_lag_str <- paste(pacf_lags[[reg_i]], collapse=", ")
    cross_info  <- c()
    if (nrow(plcmf_sig_df)>0) {
      plcmf_sub <- plcmf_sig_df[plcmf_sig_df$effect==reg_i,]
      for (k in seq_len(nrow(plcmf_sub))) {
        cause <- plcmf_sub$cause[k]; lags <- plcmf_sub$lag_signifikan[k]
        granger_sig <- if(cause %in% colnames(granger_res))
                         granger_res[reg_i, cause] else FALSE
        if (!is.na(granger_sig) && granger_sig)
          cross_info <- c(cross_info, sprintf("%s [lag: %s]", cause, lags))
      }
    }
    cross_lag_str <- if(length(cross_info)>0)
                       paste(cross_info,collapse=" | ") else "\u2014"
    results[[reg_i]] <- data.frame(
      Periode=toupper(period), Target=reg_i,
      Own_Lag_PACF=own_lag_str,
      Cross_Lag_PLCMF_Granger=cross_lag_str,
      stringsAsFactors=FALSE)
  }
  do.call(rbind, results)
}


print_gru_summary <- function(feature_df, period) {
  cat(sprintf("\n%s\n  REKOMENDASI FITUR INPUT GRU — %s\n  (PACF + PLCMF + Granger AIC)\n%s\n",
              strrep("=",80), toupper(period), strrep("=",80)))
  print(feature_df[,c("Target","Own_Lag_PACF","Cross_Lag_PLCMF_Granger")],
        row.names=FALSE)
  cat(sprintf("%s\n\n", strrep("=",80)))
}


# =============================================================================
# BAGIAN 5: PIPELINE UTAMA — [DIREVISI] dengan ADF, Granger-AIC, BH-FDR
# =============================================================================

run_full_analysis <- function(data_list,
                              pacf_lags_all = NULL,
                              output_dir    = OUTPUT_DIR) {
  dir.create(output_dir, showWarnings = FALSE)
  all_results  <- list()
  adf_all      <- list()     # kumpulkan semua hasil ADF lintas periode
  granger_all  <- list()     # kumpulkan semua hasil Granger untuk Tabel 2b

  for (period in PERIODS) {
    cat(sprintf("\n%s\n### ANALISIS PERIODE: %s\n%s\n",
                strrep("#",70), toupper(period), strrep("#",70)))

    df <- data_list[[period]]
    n  <- nrow(df)

    # ─── [BARU] 0. UJI ADF STASIONERITAS ─────────────────────────────────────
    cat("\n[0] Uji Stasioneritas ADF...\n")
    adf_result <- compute_adf_test(df, period)
    summarize_adf(adf_result, period)
    export_adf_to_csv(adf_result, period, output_dir)
    adf_all[[period]] <- adf_result

    n_stasioner <- sum(adf_result$Stasioner, na.rm = TRUE)
    if (n_stasioner < nrow(adf_result)) {
      warning(sprintf(
        "[!] %d/%d deret TIDAK stasioner pada periode %s. Periksa hasil ADF!",
        nrow(adf_result) - n_stasioner, nrow(adf_result), toupper(period)
      ))
    } else {
      cat(sprintf("  Semua %d deret stasioner (p < %.2f). Lanjut ke CCF.\n",
                  n_stasioner, ALPHA))
    }

    # ─── 1. CCF MULTI-LAG ─────────────────────────────────────────────────────
    cat("\n[1] Menghitung CCF Multi-Lag...\n")
    ccf_list <- compute_ccf_matrix(df, n_lags = N_LAGS)
    sig_ccf  <- extract_significant_ccf(ccf_list, n)
    summarize_ccf(sig_ccf, period)
    plot_ccf_matrix(ccf_list, n, period,
                    save_path = file.path(output_dir,
                                          sprintf("ccf_%s.png", period)))

    # ─── 2. PLCMF ─────────────────────────────────────────────────────────────
    cat("\n[2] Menghitung PLCMF...\n")
    plcmf_out <- compute_plcmf(df, max_lag = N_LAGS)
    sig_out   <- test_plcmf_significance(plcmf_out$P_kk_list, n)
    sig_plcmf <- extract_significant_plcmf(plcmf_out$P_kk_list,
                                            sig_out$sig_list)
    summarize_plcmf(sig_plcmf, period)

    plot_plcmf_heatmap(plcmf_out$P_kk_list, sig_out$sig_list, period, n,
                       max_display_lag = 12,
                       save_path = file.path(output_dir,
                                             sprintf("plcmf_heatmap_%s.png", period)))
    plot_plcmf_lines(plcmf_out$P_kk_list, sig_out$sig_list, period, n,
                     save_path = file.path(output_dir,
                                           sprintf("plcmf_lines_%s.png", period)))

    export_plcmf_to_csv(plcmf_out$P_kk_list, sig_out$sig_list,
                        n, period, output_dir)

    # ─── [BARU] 2B. KOREKSI BH-FDR PADA PLCMF ────────────────────────────────
    cat("\n[2B] Menerapkan koreksi BH-FDR pada elemen PLCMF...\n")
    bh_result <- apply_bh_fdr_plcmf(sig_out$pval_list, n, period)
    summarize_bh_fdr(bh_result, period)
    export_bh_fdr_to_csv(bh_result, sig_out$pval_list,
                         plcmf_out$P_kk_list, period, output_dir)
    plot_bh_comparison(bh_result, period,
                       save_path = file.path(output_dir,
                                             sprintf("bh_fdr_%s.png", period)))

    # Gunakan signifikansi PLCMF yang SUDAH dikoreksi BH untuk seleksi fitur
    sig_plcmf_bh <- extract_significant_plcmf(plcmf_out$P_kk_list,
                                               bh_result$sig_bh_list)
    cat("\n  Lag PLCMF signifikan SETELAH koreksi BH-FDR:\n")
    summarize_plcmf(sig_plcmf_bh, paste0(period, " [BH-FDR]"))

    # ─── [DIREVISI] 3. GRANGER CAUSALITY (AIC-based) ─────────────────────────
    cat("\n[3] Uji Granger Causality (seleksi lag berbasis AIC)...\n")
    granger_out <- compute_granger_aic(df, max_lag = MAX_LAG_GC)
    summarize_granger_aic(granger_out, period)
    export_granger_to_csv(granger_out, period, output_dir)
    plot_granger(granger_out, period,
                 save_path = file.path(output_dir,
                                       sprintf("granger_aic_%s.png", period)))
    granger_all[[period]] <- granger_out

    # ─── 4. INTEGRASI FITUR GRU ───────────────────────────────────────────────
    cat("\n[4] Menyusun Fitur Input GRU (menggunakan PLCMF terkoreksi BH)...\n")
    pacf_lags <- if (!is.null(pacf_lags_all)) pacf_lags_all[[period]] else
                   setNames(lapply(REGIONS, function(r) c(1,2)), REGIONS)

    # Gunakan sig_plcmf_bh (setelah koreksi BH) dan Granger AIC
    feature_df <- build_gru_features(pacf_lags, sig_plcmf_bh,
                                     granger_out$result, period)
    print_gru_summary(feature_df, period)

    write.csv(feature_df,
              file.path(output_dir,
                        sprintf("gru_features_%s.csv", period)),
              row.names = FALSE)

    all_results[[period]] <- list(
      adf_result    = adf_result,
      ccf_list      = ccf_list,
      sig_ccf       = sig_ccf,
      plcmf_out     = plcmf_out,
      sig_plcmf     = sig_plcmf,
      sig_plcmf_bh  = sig_plcmf_bh,
      bh_result     = bh_result,
      granger_out   = granger_out,
      feature_df    = feature_df
    )
  }

  # ─── [BARU] EKSPOR TABEL RINGKASAN LINTAS PERIODE ─────────────────────────
  cat("\n### EKSPOR TABEL RINGKASAN UNTUK NASKAH ###\n")

  # Tabel 2a: ADF lintas semua periode
  export_adf_summary_table(adf_all, output_dir)

  # Tabel 2b: Distribusi p* Granger lintas semua periode
  export_pstar_distribution_table(granger_all, output_dir)

  # Tabel 3 (Section 4.1): BH-FDR sensitivity summary
  bh_all <- do.call(rbind, lapply(names(all_results), function(p)
    all_results[[p]]$bh_result$summary_df))
  path_tbl3 <- file.path(output_dir, "bh_fdr_all_periods_table3.csv")
  write.csv(bh_all, path_tbl3, row.names = FALSE)
  cat(sprintf("\n  Tabel BH-FDR semua periode (Tabel 3) tersimpan: %s\n", path_tbl3))

  # Cetak ringkasan sensitivitas BH-FDR keseluruhan
  cat("\n  RINGKASAN SENSITIVITAS BH-FDR (untuk naskah):\n")
  for (period in PERIODS) {
    ov <- all_results[[period]]$bh_result$overall
    cat(sprintf("  %-12s | Tanpa koreksi: %3d | Setelah BH: %3d | %.1f%% dipertahankan\n",
                toupper(period), ov$n_sig_tanpa_kor,
                ov$n_sig_setelah_bh, ov$pct_dipertahan))
  }

  cat(sprintf("\n  Analisis selesai. Semua output tersimpan di: %s\n", output_dir))
  return(invisible(all_results))
}


# =============================================================================
# ENTRY POINT
# =============================================================================

# ---------------------------------------------------------------------------
# OPSI A: Data sintetis (untuk testing pipeline)
# ---------------------------------------------------------------------------
# data_list <- generate_synthetic_data(n = 500)

# ---------------------------------------------------------------------------
# OPSI B: Data nyata dari CSV  ← AKTIFKAN INI
# ---------------------------------------------------------------------------
BASE_DIR <- "/Users/yunita/Documents/STUDY/DATA_DISERTASI/Data-CCF"

data_list <- load_data(
  path_periode1 = file.path(BASE_DIR, "training-pagi.csv"),
  path_periode2 = file.path(BASE_DIR, "training-siang.csv"),
  path_periode3 = file.path(BASE_DIR, "training-malam.csv")
)

cat("\nPreview data period_1:\n"); print(head(data_list$period_1, 5))
cat("\nPreview data period_2:\n"); print(head(data_list$period_2, 5))
cat("\nPreview data period_3:\n"); print(head(data_list$period_3, 5))

# ---------------------------------------------------------------------------
# PACF lags dari analisis univariate (ganti dengan hasil analisis Anda)
# ---------------------------------------------------------------------------
pacf_lags_all <- list(
  period_1 = list(
    region_1 = c(1,2,7), region_2 = c(1,3),   region_3 = c(1,2),
    region_4 = c(1,4,7), region_5 = c(1,2),   region_6 = c(1,3,7)
  ),
  period_2 = list(
    region_1 = c(1,7),   region_2 = c(1,2,7), region_3 = c(1,2,3),
    region_4 = c(1,2),   region_5 = c(1,7),   region_6 = c(1,2,7)
  ),
  period_3 = list(
    region_1 = c(1,2),   region_2 = c(1,7),   region_3 = c(1,2,7),
    region_4 = c(1,3),   region_5 = c(1,2),   region_6 = c(1,7)
  )
)

# ---------------------------------------------------------------------------
# Jalankan pipeline lengkap
# ---------------------------------------------------------------------------
results <- run_full_analysis(
  data_list     = data_list,
  pacf_lags_all = pacf_lags_all,
  output_dir    = "./output_analysis"
)

