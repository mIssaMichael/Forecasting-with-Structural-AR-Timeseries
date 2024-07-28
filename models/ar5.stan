data {
    int<lower=0> T; // Time Steps
    int<lower=0> T_pred; // Future Time Steps
    vector[T] y; // Observed data

    // Prior parameters
    real alpha_mu;
    real<lower=0> alpha_sigma;

    real beta_mu;
    real<lower=0> beta_sigma;

    real init_mu;
    real<lower=0> init_sigma;

    real<lower=0> error_sigma;


}

parameters {
    real alpha;
    array[2] real beta;
    real<lower=0> sigma;
}

model {
    // Priors
    alpha ~ normal(alpha_mu, alpha_sigma);
    beta ~ normal(beta_mu, beta_sigma); 
    sigma ~ normal(0, error_sigma);

    // Likelihood
    for (t in 3:T) {
        y[t] ~ normal(alpha + beta[1] * y[t-1] + beta[2] * y[t-2], sigma);
    }
}

generated quantities { 
    vector[T] y_rep; 
    vector[T_pred + T] y_pred;
    
    for (t in 1:2) {
        y_rep[t] = normal_rng(init_mu, init_sigma);
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
