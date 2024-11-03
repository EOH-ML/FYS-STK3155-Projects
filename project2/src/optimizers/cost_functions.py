import jax.numpy as jnp

def ols(z, X, coef, _): # So that we dont have to do any if checks for alpha parameter
    return jnp.sum((z - X @ coef)**2)

def ridge(z, X, coef, alpha):
    return jnp.sum((z - X @ coef)**2) + alpha * jnp.sum(coef**2)

def lasso(z, X, coef, alpha):
    return jnp.sum((z - X @ coef)**2) + alpha * jnp.sum(jnp.abs(coef))
