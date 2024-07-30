data {
    int<lower=0> T; // Time Steps
    int<lower=0> F; // Number of Fourier Features

    // Prior parameters
    real alpha_mu;
    real<lower=0> alpha_sigma;

    real beta_mu;
    real<lower=0> beta_sigma;
    
    // Initial value priors
    real init_mu;
    real<lower=0> init_sigma;

    real<lower=0> error_sigma;

    // Trend Prior parameters
    real trend_alpha_mu;
    real<lower=0> trend_alpha_sigma;

    real trend_beta_mu;
    real<lower=0> trend_beta_sigma;

    // Seasonality Prior parameters
    real beta_fourier_mu;
    real<lower=0> beta_fourier_sigma;

    matrix[T, F] fourier_features;
}

transformed data {
   matrix[cols(fourier_features), rows(fourier_features)] fourier_features_T = fourier_features';
}

generated quantities {
    real alpha = normal_rng(alpha_mu, alpha_sigma);
    array[2] real beta;
    real<lower=0> sigma = abs(normal_rng(0, error_sigma));

    real trend_alpha = normal_rng(trend_alpha_mu, trend_alpha_sigma);
    real trend_beta = normal_rng(trend_beta_mu, trend_beta_sigma);

    vector[F] beta_fourier;

    // Generate Fourier coefficients
    for (f in 1:F) {
        beta_fourier[f] = normal_rng(beta_fourier_mu, beta_fourier_sigma);
    }

    array[T] real y_sim;

    // Generate initial values for y_prior
    for (i in 1:2) {
        beta[i] = normal_rng(beta_mu, beta_sigma);
    }

    for (t in 1:2) {
        y_sim[t] = normal_rng(init_mu, init_sigma);
    }

    // Generate prior predictive samples
    for (t in 3:T) {
        y_sim[t] = normal_rng(
            alpha + beta[1] * y_sim[t-1] + beta[2] * y_sim[t-2] + 
            trend_alpha + trend_beta * t + 
            dot_product(beta_fourier, fourier_features_T[, t]), 
            sigma
        );
    }
}