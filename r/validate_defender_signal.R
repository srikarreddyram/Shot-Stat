#!/usr/bin/env Rscript
#
# Independent statistical validation of the defender-quality features, run
# entirely outside the Python/XGBoost stack. Mirrors the same role R plays
# on the ComplaintIQ project: a QA layer that checks whether an engineered
# feature carries genuine signal by classical hypothesis testing, not a
# replacement for the tree-based model.
#
# The question this answers is narrower than "does the model work" — XGBoost
# ablation already answers that, empirically, by measuring log-loss with and
# without the feature. This asks a different, complementary question: is the
# association between the defender feature and the shot outcome
# STATISTICALLY SIGNIFICANT once shot difficulty (zone, distance) is held
# constant, using a method (logistic regression + a likelihood-ratio test)
# that owes nothing to gradient boosting and would flag a feature that is
# pure noise even if a tree model found some spurious use for it.
#
# Input: data/exports/shots_for_r_validation.csv, written by
#   python -m src.analysis.export_for_r_validation
# That export is a snapshot of the existing point-in-time feature pipeline's
# output — this script does not recompute, and could not recompute, the
# point-in-time joins itself.
#
# Usage:
#   Rscript r/validate_defender_signal.R

suppressMessages({
  library(tools)
})

project_root <- normalizePath(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE))), ".."))
if (length(project_root) == 0 || !dir.exists(project_root)) {
  project_root <- getwd()
}
data_path <- file.path(project_root, "data", "exports", "shots_for_r_validation.csv")

if (!file.exists(data_path)) {
  stop(sprintf(
    "Expected export not found at %s.\nRun first: python -m src.analysis.export_for_r_validation",
    data_path
  ))
}

shots <- read.csv(data_path, stringsAsFactors = FALSE)
shots$zone <- factor(shots$zone)

cat(sprintf("Loaded %d shots from %s\n", nrow(shots), data_path))

complete <- shots[
  !is.na(shots$def_fg_pct_zone) &
  !is.na(shots$def_pct_plusminus_zone) &
  !is.na(shots$def_matchup_share) &
  !is.na(shots$shot_distance) &
  !is.na(shots$zone),
]
cat(sprintf(
  "%d shots (%.1f%%) have full defender-feature coverage and are used below\n",
  nrow(complete), 100 * nrow(complete) / nrow(shots)
))

# ── Model 1: shot difficulty alone (zone + distance), no defender info ─────
reduced <- glm(
  shot_made ~ zone + shot_distance,
  data = complete, family = binomial(link = "logit")
)

# ── Model 2: difficulty + defender quality features ────────────────────────
full <- glm(
  shot_made ~ zone + shot_distance + def_fg_pct_zone +
    def_pct_plusminus_zone + def_matchup_share,
  data = complete, family = binomial(link = "logit")
)

cat("\n================================================================\n")
cat("  FULL MODEL — coefficients on the defender features\n")
cat("================================================================\n")
full_summary <- summary(full)$coefficients
defender_terms <- c("def_fg_pct_zone", "def_pct_plusminus_zone", "def_matchup_share")
print(round(full_summary[defender_terms, , drop = FALSE], 4))

cat("\nOdds ratios (exp(coefficient)) with 95% Wald CIs:\n")
coefs <- coef(full)[defender_terms]
ses <- full_summary[defender_terms, "Std. Error"]
or_table <- data.frame(
  term = defender_terms,
  odds_ratio = exp(coefs),
  ci_low = exp(coefs - 1.96 * ses),
  ci_high = exp(coefs + 1.96 * ses)
)
print(or_table, row.names = FALSE)

# ── Likelihood-ratio test: do the defender terms earn their place at all? ──
lr_test <- anova(reduced, full, test = "Chisq")
cat("\n================================================================\n")
cat("  LIKELIHOOD-RATIO TEST — reduced (zone+distance) vs full (+defender)\n")
cat("================================================================\n")
print(lr_test)

lr_p <- lr_test$`Pr(>Chi)`[2]
cat(sprintf(
  "\nLR test p-value: %.3e  ->  %s\n",
  lr_p,
  ifelse(lr_p < 0.001,
         "defender features are significantly associated with shot outcome\n            beyond shot difficulty alone (p < 0.001).",
         ifelse(lr_p < 0.05,
                "defender features are significantly associated with shot outcome\n            beyond shot difficulty alone (p < 0.05).",
                "NOT significant at the 0.05 level — no independent evidence of a\n            defender effect beyond shot difficulty in this sample."))
))

# ── Diagnostic: does a defender's zone FG%-allowed track actual outcomes? ──
cat("\nWriting calibration diagnostic to r/output/defender_fg_calibration.png ...\n")
dir.create(file.path(project_root, "r", "output"), showWarnings = FALSE, recursive = TRUE)

complete$def_fg_bin <- cut(complete$def_fg_pct_zone, breaks = 10)
bin_summary <- aggregate(
  shot_made ~ def_fg_bin, data = complete,
  FUN = function(x) c(mean = mean(x), n = length(x))
)
bin_means <- sapply(bin_summary$shot_made, function(x) x["mean"])
bin_ns <- sapply(bin_summary$shot_made, function(x) x["n"])
bin_mids <- sapply(strsplit(gsub("[()\\[\\]]", "", as.character(bin_summary$def_fg_bin)), ","),
                    function(x) mean(as.numeric(x)))

png(file.path(project_root, "r", "output", "defender_fg_calibration.png"),
    width = 900, height = 600, res = 120)
plot(bin_mids, bin_means,
     xlab = "Defender zone FG% allowed (point-in-time, binned)",
     ylab = "Actual shot-made rate",
     main = "Does a tougher point-in-time defender rating\ntrack a lower actual make rate?",
     pch = 19, cex = pmin(3, 0.5 + bin_ns / max(bin_ns) * 3), col = "#2b6cb0")
abline(lm(bin_means ~ bin_mids), col = "#c05621", lwd = 2, lty = 2)
legend("topleft", legend = "OLS trend", col = "#c05621", lty = 2, lwd = 2, bty = "n")
invisible(dev.off())

cat("\n✓ done.\n")
