class PoissonRegessrorGLM(object):
    def __init__(self, exog, endo, freq):
        model = sm.GLM(endo, 
            exog, 
            family=sm.families.Poisson(), 
            freq_weights=freq,
            missing="drop",
        )
        
        print(model)

        # Fit the model
        result = model.fit(maxiter=300)
        print(result.summary())
        self.coef = result.params
        
        pred = model.predict(exog)
        
        return pred