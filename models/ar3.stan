// Predict Future Values 
data {
    int<lower=0> T; // Time Steps
    int<lower=0> T_pred; // Future Time Steps
    vector[T] y; // Observed data
}

parameters {
    real alpha;
    array[2] real beta;
    real<lower=0> sigma;
}

model {
    // Priors
    alpha ~ normal(10, 0.1);
    beta ~ normal(0, 0.1); // Adjusted mean to 0 and standard deviation to 0.1
    sigma ~ normal(0, 8);

    // Likelihood
    for (t in 3:T) {
        y[t] ~ normal(alpha + beta[1] * y[t-1] + beta[2] * y[t-2], sigma);
    }
}

generated quantities { 
    vector[T] y_rep; 
    vector[T_pred + T] y_pred;
    
    for (t in 1:2) {
        y_rep[t] = normal_rng(9, 0.1);
    }
    for (t in 3:T) {
        y_rep[t] = normal_rng(alpha + beta[1] * y[t-1] + beta[2] * y[t-2], sigma);
    }

    // Initialize y_pred with the same values as y_rep
    for (t in 1:T) {
        y_pred[t] = y_rep[t];
    }

    // Generate future values
    for (t in (T+1):(T+T_pred)) {
        y_pred[t] = normal_rng(alpha + beta[1] * y_pred[t-1] + beta[2] * y_pred[t-2], sigma);
    }
}

