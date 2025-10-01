# Analysis of bicoppd1d2 rotation handling and direct C function calls
# This replicates the bicoppd1d2 logic using direct VineCopula calls

library(VineCopula)
library(gamCopula)

# Data
u1 <- c(0.930144, 0.853130, 0.431601, 0.701803, 0.736621, 0.585327, 0.595364, 0.168512, 0.946548, 0.893125)
u2 <- c(0.533249, 0.699036, 0.813250, 0.453937, 0.653818, 0.571856, 0.855921, 0.153280, 0.862996, 0.977878)
theta <- rep(1.47, 10)

cat("Understanding bicoppd1d2 rotation handling:\n")
cat("u1[1:5]:    ", sprintf("%.6f ", u1[1:5]), "\n")
cat("u2[1:5]:    ", sprintf("%.6f ", u2[1:5]), "\n")
cat("theta[1:5]: ", sprintf("%.6f ", theta[1:5]), "\n\n")

# Test gamCopula family 302 (Clayton rotation 3)
family_gamCopula <- 302
n <- length(u1)

cat("=== 1. Original bicoppd1d2 approach ===\n")
data_matrix <- cbind(u1, u2, theta, rep(0, 10))
bicoppd1d2_result <- gamCopula:::bicoppd1d2(data_matrix, family_gamCopula, p=TRUE, d1=TRUE, d2=TRUE)
cat("bicoppd1d2 PDF: ", sprintf("%.8f ", bicoppd1d2_result[1,]), "\n")
cat("bicoppd1d2 1st deriv: ", sprintf("%.8f ", bicoppd1d2_result[2,]), "\n") 
cat("bicoppd1d2 2nd deriv: ", sprintf("%.8f ", bicoppd1d2_result[3,]), "\n\n")

cat("=== 2. Understanding getFams() function ===\n")
fams <- gamCopula:::getFams(family_gamCopula)
cat("getFams(302) returns:", fams, "\n")
cat("This means: family for positive par =", fams[1], ", family for negative par =", fams[2], "\n\n")

cat("=== 3. Checking parameter signs ===\n")
positive_params <- theta > 0
cat("All parameters positive:", all(positive_params), "\n")
cat("So we use VineCopula family:", fams[1], "\n\n")

cat("=== 4. Direct VineCopula calls with correct family ===\n")
vinecop_family <- fams[1]  # This should be the correct family
cat("Using VineCopula family:", vinecop_family, "\n")

# PDF calculation
pdf_result <- numeric(n)
for(i in 1:n) {
    pdf_result[i] <- BiCopPDF(u1[i], u2[i], vinecop_family, theta[i])
}
cat("VineCopula PDF: ", sprintf("%.8f ", pdf_result), "\n")

# First derivative calculation (log=TRUE to match bicoppd1d2)
d1_result <- numeric(n)
for(i in 1:n) {
    d1_result[i] <- BiCopDeriv(u1[i], u2[i], vinecop_family, theta[i], log=TRUE)
}
cat("VineCopula 1st deriv: ", sprintf("%.8f ", d1_result), "\n")

# Second derivative calculation  
d2_result <- numeric(n)
for(i in 1:n) {
    d2_result[i] <- BiCopDeriv2(u1[i], u2[i], vinecop_family, theta[i])
}
cat("VineCopula 2nd deriv: ", sprintf("%.8f ", d2_result), "\n\n")

cat("=== 5. Comparison ===\n")
cat("PDF differences: ", sprintf("%.10f ", pdf_result - bicoppd1d2_result[1,]), "\n")
cat("1st deriv differences: ", sprintf("%.10f ", d1_result - bicoppd1d2_result[2,]), "\n")
cat("2nd deriv differences: ", sprintf("%.10f ", d2_result - bicoppd1d2_result[3,]), "\n\n")

cat("=== 6. Now test with rotated data ===\n")
# For Clayton rotation 3 (270°), the transformation should be v -> 1-v
u1_rotated <- u1           # u stays the same for 270° rotation
u2_rotated <- 1 - u2       # v -> 1-v for 270° rotation  
theta_rotated <- -theta    # parameter sign change for rotation

cat("Rotated data (first 3 rows):\n")
cat("Original: u1=", sprintf("%.6f ", u1[1:3]), "u2=", sprintf("%.6f ", u2[1:3]), "\n")
cat("Rotated:  u1=", sprintf("%.6f ", u1_rotated[1:3]), "u2=", sprintf("%.6f ", u2_rotated[1:3]), "\n")
cat("Theta: original=", sprintf("%.3f ", theta[1:3]), "rotated=", sprintf("%.3f ", theta_rotated[1:3]), "\n\n")

# Since theta_rotated is negative, getFams should use the second family
cat("With negative parameters, getFams(302) uses family:", fams[2], "\n")

# Test with rotated data
pdf_rotated <- numeric(n)
d1_rotated <- numeric(n)
d2_rotated <- numeric(n)

for(i in 1:n) {
    pdf_rotated[i] <- BiCopPDF(u1_rotated[i], u2_rotated[i], fams[2], abs(theta_rotated[i]))
    d1_rotated[i] <- BiCopDeriv(u1_rotated[i], u2_rotated[i], fams[2], abs(theta_rotated[i]), log=TRUE)
    d2_rotated[i] <- BiCopDeriv2(u1_rotated[i], u2_rotated[i], fams[2], abs(theta_rotated[i]))
}

cat("Rotated results:\n")
cat("PDF: ", sprintf("%.8f ", pdf_rotated), "\n")
cat("1st deriv: ", sprintf("%.8f ", d1_rotated), "\n")
cat("2nd deriv: ", sprintf("%.8f ", d2_rotated), "\n\n")

cat("=== 7. Test direct C function calls ===\n")
# Now try the .C() calls with the correct family
tryCatch({
    d1_c_result <- .C("difflPDF_mod",
                      u = as.double(u1),
                      v = as.double(u2),
                      n = as.integer(n),
                      param = as.double(theta),
                      copula = as.integer(rep(vinecop_family, n)),
                      out = numeric(n),
                      PACKAGE = "VineCopula")
    cat("Direct C call (difflPDF_mod): ", sprintf("%.8f ", d1_c_result$out), "\n")
    cat("Matches VineCopula wrapper:", all(abs(d1_c_result$out - d1_result) < 1e-10), "\n")
}, error = function(e) {
    cat("Direct C call failed:", e$message, "\n")
})

cat("\n=== SUMMARY ===\n")
cat("Key insights:\n")
cat("1. gamCopula family 302 -> VineCopula family", fams[1], "for positive params\n")
cat("2. bicoppd1d2 uses BiCopDeriv with log=TRUE for first derivatives\n")
cat("3. Rotation is handled by data transformation + parameter sign change\n")
cat("4. getFams() function is the key to understanding family mapping\n")