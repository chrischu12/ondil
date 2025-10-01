# Comprehensive comparison of different gamCopula family codes
# Testing how bicoppd1d2 behaves with different Clayton rotation families

library(VineCopula)
library(gamCopula)

# Test data
u1 <- c(0.930144, 0.853130, 0.431601, 0.701803, 0.736621)
u2 <- c(0.533249, 0.699036, 0.813250, 0.453937, 0.653818)
theta <- rep(1.47, 5)

cat("=== Comparing different Clayton family codes ===\n")
cat("u1: ", sprintf("%.6f ", u1), "\n")
cat("u2: ", sprintf("%.6f ", u2), "\n")
cat("theta: ", sprintf("%.3f ", theta), "\n\n")

# Test different Clayton family codes
clayton_families <- c(300, 301, 302, 303)  # Different Clayton rotations
family_names <- c("0° (standard)", "90°", "180°", "270°")

cat("=== Family mapping analysis ===\n")
for(i in 1:length(clayton_families)) {
    fam <- clayton_families[i]
    cat(sprintf("Testing family %d (%s):\n", fam, family_names[i]))
    
    tryCatch({
        fams <- gamCopula:::getFams(fam)
        cat(sprintf("  getFams(%d) = [%d, %d]\n", fam, fams[1], fams[2]))
    }, error = function(e) {
        cat("  ERROR with getFams:", e$message, "\n")
        # Try alternative approach
        tryCatch({
            # Test with a sample data point to understand the family mapping
            test_data <- matrix(c(0.5, 0.5, 1.0, 0), nrow=1)
            test_result <- gamCopula:::bicoppd1d2(test_data, fam, p=TRUE)
            cat("  bicoppd1d2 test successful - family exists\n")
        }, error = function(e2) {
            cat("  bicoppd1d2 also failed:", e2$message, "\n")
        })
    })
}
cat("\n")

# Test bicoppd1d2 with different families
cat("=== bicoppd1d2 results for different families ===\n")
data_matrix <- cbind(u1, u2, theta, rep(0, 5))

results <- list()
for(i in 1:length(clayton_families)) {
    fam <- clayton_families[i]
    cat(sprintf("Family %d (%s):\n", fam, family_names[i]))
    
    tryCatch({
        # Get bicoppd1d2 results
        result <- gamCopula:::bicoppd1d2(data_matrix, fam, p=TRUE, d1=TRUE, d2=TRUE)
        results[[as.character(fam)]] <- result
        
        cat("  PDF:      ", sprintf("%.8f ", result[1,]), "\n")
        cat("  1st deriv:", sprintf("%.8f ", result[2,]), "\n")
        cat("  2nd deriv:", sprintf("%.8f ", result[3,]), "\n")
        
        # Try to get getFams info here where we know the family works
        tryCatch({
            fams <- gamCopula:::getFams(fam)
            cat("  getFams() = [", fams[1], ",", fams[2], "]\n")
        }, error = function(e) {
            cat("  getFams() failed:", e$message, "\n")
        })
        
    }, error = function(e) {
        cat("  ERROR:", e$message, "\n")
    })
    cat("\n")
}

# Compare specific families (301 vs 302) - only if both exist
cat("=== Detailed comparison: Family 301 vs 302 ===\n")
if("302" %in% names(results) && "301" %in% names(results)) {
    diff_pdf <- results[["302"]][1,] - results[["301"]][1,]
    diff_d1 <- results[["302"]][2,] - results[["301"]][2,]
    diff_d2 <- results[["302"]][3,] - results[["301"]][3,]
    
    cat("PDF differences (302-301):      ", sprintf("%.10f ", diff_pdf), "\n")
    cat("1st deriv differences (302-301):", sprintf("%.10f ", diff_d1), "\n")
    cat("2nd deriv differences (302-301):", sprintf("%.10f ", diff_d2), "\n")
    
    cat("Are derivatives the same? PDF:", all(abs(diff_pdf) < 1e-10), 
        " D1:", all(abs(diff_d1) < 1e-10), 
        " D2:", all(abs(diff_d2) < 1e-10), "\n\n")
} else {
    cat("Cannot compare - one or both families failed\n\n")
}

# Test with negative theta to see rotation effects
cat("=== Testing with negative parameters ===\n")
theta_neg <- rep(-1.47, 5)
data_matrix_neg <- cbind(u1, u2, theta_neg, rep(0, 5))

cat("With theta = -1.47:\n")
for(i in 1:length(clayton_families)) {
    fam <- clayton_families[i]
    cat(sprintf("Family %d (%s):\n", fam, family_names[i]))
    
    tryCatch({
        result_neg <- gamCopula:::bicoppd1d2(data_matrix_neg, fam, p=TRUE, d1=TRUE, d2=TRUE)
        cat("  PDF:      ", sprintf("%.8f ", result_neg[1,]), "\n")
        cat("  1st deriv:", sprintf("%.8f ", result_neg[2,]), "\n")
        cat("  2nd deriv:", sprintf("%.8f ", result_neg[3,]), "\n")
    }, error = function(e) {
        cat("  ERROR:", e$message, "\n")
    })
    cat("\n")
}

# Test the actual data transformations for rotations
cat("=== Understanding data transformations ===\n")
cat("Original data (first 3 obs):\n")
cat("u1 =", sprintf("%.6f ", u1[1:3]), "\n")
cat("u2 =", sprintf("%.6f ", u2[1:3]), "\n\n")

# Apply rotations manually to understand the transformations
rotations <- list(
    "0°" = list(u1 = u1, u2 = u2, theta = theta),
    "90°" = list(u1 = 1-u1, u2 = u2, theta = -theta),
    "180°" = list(u1 = 1-u1, u2 = 1-u2, theta = theta),
    "270°" = list(u1 = u1, u2 = 1-u2, theta = -theta)
)

cat("Data after rotation transformations:\n")
for(rot_name in names(rotations)) {
    rot_data <- rotations[[rot_name]]
    cat(sprintf("%s rotation:\n", rot_name))
    cat("  u1 =", sprintf("%.6f ", rot_data$u1[1:3]), "\n")
    cat("  u2 =", sprintf("%.6f ", rot_data$u2[1:3]), "\n")
    cat("  theta =", sprintf("%.3f ", rot_data$theta[1:3]), "\n\n")
}

# Test direct VineCopula calls to understand the underlying differences
cat("=== Direct VineCopula family calls ===\n")
for(i in 1:length(clayton_families)) {
    fam <- clayton_families[i]
    fams <- gamCopula:::getFams(fam)
    
    cat(sprintf("Family %d uses VineCopula families [%d, %d]:\n", fam, fams[1], fams[2]))
    
    # Test with positive theta (uses fams[1])
    pdf_pos <- BiCopPDF(u1[1], u2[1], fams[1], theta[1])
    d1_pos <- BiCopDeriv(u1[1], u2[1], fams[1], theta[1], log=TRUE)
    
    cat(sprintf("  With positive theta: PDF=%.8f, D1=%.8f\n", pdf_pos, d1_pos))
    
    # Test with negative theta (uses fams[2])
    if(fams[1] != fams[2]) {
        pdf_neg <- BiCopPDF(u1[1], u2[1], fams[2], abs(theta_neg[1]))
        d1_neg <- BiCopDeriv(u1[1], u2[1], fams[2], abs(theta_neg[1]), log=TRUE)
        cat(sprintf("  With negative theta: PDF=%.8f, D1=%.8f\n", pdf_neg, d1_neg))
    }
    cat("\n")
}

cat("=== SUMMARY ===\n")
cat("Key insights:\n")
cat("1. Different family codes (301 vs 302) use different rotation transformations\n")
cat("2. This affects the data (u1, u2) and parameter signs fed to VineCopula\n")
cat("3. Even if derivatives look similar, the transformations create different likelihood surfaces\n")
cat("4. This leads to different optimal coefficients in gamBiCopFit\n")
cat("5. getFams() reveals which VineCopula families are used for pos/neg parameters\n")