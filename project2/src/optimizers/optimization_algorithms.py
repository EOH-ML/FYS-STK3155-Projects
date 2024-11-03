import jax
import jax.numpy as jnp
import numpy as np

class Optimization:
    '''
    --OPPRETTELSE AV OPTIMIZATION OBJEKTET--
    Må først opprette et Optimization objekt. Har gjort følgende attributer generelle for klassen:

    X - design_matrix
    z - true data
    initial_parameters - Typisk generert med en random metode
    batch_size - batch size
    epochs - epochs
    alpha - regularization parameter

    ALTSÅ: Ønsker man å endre disse må man bare opprette et nytt objekt av klassen

    Videre er følgende bestemte:

    delta = 1e-8 - for numerisk stabilitet (tenker denne ikke trengs å endres)
    number_of_samples = bare lengden av z, antall rader i X - Bare enklere å forholde seg til
    number_of_minibatches = antall samples / batch size - Bare enklere å forholde seg til

    --KLARHETER FOR METODENE I KLASSEN--
    epsilon, rho1, rho2 - paramterere etter Goodfellow notasjon
    momentum - grad av momentum (Oscar notasjon?)
    t0, t1 - Morten sin notasjon for learning_schedule

    cost - et funksjons objekt.
    Anbefaler at denne hentes fra filen cost_functions.py, slik som dette:
    from cost_functions import ols, ridge, lasso
    Da kan man helt enkelt skrive, cost=ols / cost=ridge / cost = lasso i metode kallene. 

    Dersom man skriver cost = ols, trenger ikke alpha være spesifisert (men den kan være, vil da ikke bli brukt)
    Skriver man derimot cost = ridge / lasso, burde alpha være spesifisert (ellers settes alpha = 0, og de vil oppføre seg lignende ols)
    
    '''
    def __init__(self, X, z, initial_parameters, batch_size, epochs, alpha=0): 
        self._X = X
        self._z = z
        self._initial_parameters = initial_parameters
        self._batch_size = batch_size
        self._epochs = epochs
        self._alpha = alpha
        self._delta = 1e-8 # Needed for numerical stability
        
        self._number_of_samples = len(self._z)
        self._number_of_minibatches = int(self._number_of_samples / self._batch_size)

        if len(self._initial_parameters) != self._X.shape[1]:
            raise ValueError("The number of initial parameters must match the number of features in X.")
        if len(self._z) != self._X.shape[0]:
            raise ValueError("The number of observations in z must match the number of rows in X.")

    def gradient_descent_plain(self, cost, eta, momentum, max_iterations):
        grad_fn = jax.grad(cost, argnums=2)
        gd_coef = self._initial_parameters
        change = np.zeros_like(gd_coef)

        for _ in range(max_iterations):
            gradient = grad_fn(self._z, self._X, gd_coef, self._alpha) 
            change = momentum * change + eta * gradient
            gd_coef -= change 

        return gd_coef
 
    def SGD_momentum(self, cost, t0, t1, momentum):
        grad_fn = jax.grad(cost, argnums=2)

        learning_schedule = lambda t: t0 / (t + t1) # Could be diffrent, not specific for SGD

        sgd_coef = self._initial_parameters 
        change = np.zeros_like(sgd_coef)

        for e in range(self._epochs):
            shuffled_indicies = np.random.permutation(self._number_of_samples) # Random shuffles the indices
            for b in range(self._number_of_minibatches):
                indices = shuffled_indicies[b * self._batch_size : (b+1) * self._batch_size] # Gives me batch_size number of random indicies, and they will not be the same
                # For batch_size = 4, indicies could be => [2, 7, 3, 2]

                Xi = self._X[indices]
                zi = self._z[indices] 
                if len(zi) != len(Xi) or len(Xi) != self._batch_size:
                    raise ValueError(f"The number of samples in the batch are inconsistent: len(zi)={len(zi)}, len(Xi)={len(Xi)}, batch_size={self._batch_size}.")

                gradient = (1 / self._batch_size) * grad_fn(zi, Xi, sgd_coef, self._alpha) 

                eta = learning_schedule(e * self._number_of_minibatches + b) # e * number_of_minibatches + b gives the total number of gradient updates that have been performed up to this point in training
                change = momentum * change + eta * gradient 
                sgd_coef = sgd_coef - change 
        return sgd_coef 

    def adaGrad(self, cost, epsilon):
        grad_fn = jax.grad(cost, argnums=2)
        adaGrad_coef = self._initial_parameters

        r = np.zeros_like(adaGrad_coef) # Initialize gradient accumulation variable

        for _ in range(self._epochs):
            shuffled_indicies = np.random.permutation(self._number_of_samples) # Random shuffles the indices
            for b in range(self._number_of_minibatches):
                indices = shuffled_indicies[b * self._batch_size : (b+1) * self._batch_size] # Gives me batch_size number of random indicies, and they will not be the same

                Xi = self._X[indices]
                zi = self._z[indices] 
                if len(zi) != len(Xi) or len(Xi) != self._batch_size:
                    raise ValueError(f"The number of samples in the batch are inconsistent: len(zi)={len(zi)}, len(Xi)={len(Xi)}, batch_size={self._batch_size}.")
                
                gradient = (1 / self._batch_size) * grad_fn(zi, Xi, adaGrad_coef, self._alpha)

                r += gradient * gradient
                change = - (epsilon / (self._delta + np.sqrt(r))) * gradient
                adaGrad_coef += change
        return adaGrad_coef

    def RMSProp(self, cost, epsilon, rho):
        grad_fn = jax.grad(cost, argnums=2)
        RMSProp_coef = self._initial_parameters

        r = np.zeros_like(RMSProp_coef) # Initialize gradient accumulation variable, Morten setter denne på insiden (hvorfor?)

        for _ in range(self._epochs):
            shuffled_indicies = np.random.permutation(self._number_of_samples) # Random shuffles the indices
            for b in range(self._number_of_minibatches):
                indices = shuffled_indicies[b * self._batch_size : (b+1) * self._batch_size] # Gives me batch_size number of random indicies, and they will not be the same

                Xi = self._X[indices]
                zi = self._z[indices] 
                if len(zi) != len(Xi) or len(Xi) != self._batch_size:
                    raise ValueError(f"The number of samples in the batch are inconsistent: len(zi)={len(zi)}, len(Xi)={len(Xi)}, batch_size={self._batch_size}.")
                
                gradient = (1 / self._batch_size) * grad_fn(zi, Xi, RMSProp_coef, self._alpha)

                r = rho * r + (1-rho) * np.square(gradient)
                update = - (epsilon / (np.sqrt(self._delta + r))) * gradient
                RMSProp_coef += update
        return RMSProp_coef


    def Adam(self, cost, epsilon, rho1, rho2):
        grad_fn = jax.grad(cost, argnums=2)
        Adam_coef = self._initial_parameters

        s = np.zeros_like(Adam_coef) # Initialize 1st moment
        r = np.zeros_like(Adam_coef) # Initialize 2nd moment
        t = 0 # Initialize time step

        for _ in range(self._epochs):
            shuffled_indicies = np.random.permutation(self._number_of_samples) # Random shuffles the indices
            for b in range(self._number_of_minibatches):
                indices = shuffled_indicies[b * self._batch_size : (b+1) * self._batch_size] # Gives me batch_size number of random indicies, and they will not be the same

                Xi = self._X[indices]
                zi = self._z[indices] 
                if len(zi) != len(Xi) or len(Xi) != self._batch_size:
                    raise ValueError(f"The number of samples in the batch are inconsistent: len(zi)={len(zi)}, len(Xi)={len(Xi)}, batch_size={self._batch_size}.")

                gradient = (1 / self._batch_size) * grad_fn(zi, Xi, Adam_coef, self._alpha)  
                
                t += 1
                s = rho1 * s + (1- rho1) * gradient
                r = rho2 * r + (1 - rho2) * np.square(gradient)
                s_ = s / (1 - rho1**t)
                r_ = r / (1 - rho2**t)

                update = - (epsilon * s_) / (np.sqrt(r_) + self._delta)

                Adam_coef += update
        return Adam_coef
