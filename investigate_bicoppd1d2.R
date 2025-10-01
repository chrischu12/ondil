# Find the exact C functions used by bicoppd1d2
library(gamCopula)

# Data setup
u1 <- c(0.930144, 0.853130, 0.431601, 0.701803, 0.736621, 0.585327, 0.595364, 0.168512, 0.946548, 0.893125)
u2 <- c(0.533249, 0.699036, 0.813250, 0.453937, 0.653818, 0.571856, 0.855921, 0.153280, 0.862996, 0.977878)
theta <- rep(1.47, 10)

cat("=== Investigating bicoppd1d2 internals ===\n")

# First, let's see the source of bicoppd1d2
cat("bicoppd1d2 function source:\n")
cat("---------------------------\n")
print(gamCopula:::bicoppd1d2)

cat("\n=== Checking gamCopula namespace ===\n")
# List all functions in gamCopula namespace
gamCopula_functions <- ls(envir = asNamespace("gamCopula"))
c_related <- gamCopula_functions[grepl("^\\.|C_|bicop", gamCopula_functions, ignore.case = TRUE)]
cat("Potential C-related functions:\n")
print(c_related)

cat("\n=== Checking loaded DLLs ===\n")
# Check what C functions are registered
tryCatch({
    dll_info <- getDLLRegisteredRoutines("gamCopula")
    cat("Registered .C routines:\n")
    if(length(dll_info$.C) > 0) {
        for(i in 1:length(dll_info$.C)) {
            cat(sprintf("  %s\n", dll_info$.C[[i]]$name))
        }
    }
    
    cat("\nRegistered .Call routines:\n")
    if(length(dll_info$.Call) > 0) {
        for(i in 1:length(dll_info$.Call)) {
            cat(sprintf("  %s\n", dll_info$.Call[[i]]$name))
        }
    }
}, error = function(e) {
    cat("Could not access DLL info:", e$message, "\n")
})

cat("\n=== Looking for specific function patterns ===\n")
# Look for functions that might compute derivatives
all_objects <- objects(envir = asNamespace("gamCopula"), all.names = TRUE)
derivative_related <- all_objects[grepl("deriv|grad|bicop|clayton", all_objects, ignore.case = TRUE)]
cat("Derivative/copula related functions:\n")
print(derivative_related)

cat("\n=== Trying to trace bicoppd1d2 execution ===\n")
# Let's trace what happens when we call bicoppd1d2
data_matrix <- cbind(u1[1:3], u2[1:3], theta[1:3], rep(0, 3))

# Enable tracing if possible
tryCatch({
    trace(gamCopula:::bicoppd1d2, tracer = function() cat("bicoppd1d2 called\n"))
    result_small <- gamCopula:::bicoppd1d2(data_matrix, 302, p=TRUE, d1=TRUE, d2=TRUE)
    untrace(gamCopula:::bicoppd1d2)
    
    cat("Small test result:\n")
    print(result_small)
    
}, error = function(e) {
    cat("Tracing failed:", e$message, "\n")
})

cat("\n=== Direct approach: Examine function body ===\n")
# Get the actual function body
func_body <- body(gamCopula:::bicoppd1d2)
cat("Function body structure:\n")
str(func_body)

# Try to deparse it to see the code
cat("\nFunction code:\n")
cat(paste(deparse(func_body), collapse="\n"))

cat("\n=== Alternative: Look for similar functions ===\n")
# Check if there are other bicop functions
bicop_functions <- all_objects[grepl("bicop", all_objects, ignore.case = TRUE)]
cat("All bicop functions:\n")
print(bicop_functions)

# Try to access each one to see what they do
for(func_name in bicop_functions) {
    tryCatch({
        func_obj <- get(func_name, envir = asNamespace("gamCopula"))
        if(is.function(func_obj)) {
            cat(sprintf("\nFunction: %s\n", func_name))
            cat("Arguments:", paste(names(formals(func_obj)), collapse=", "), "\n")
        }
    }, error = function(e) {
        # Skip if can't access
    })
}