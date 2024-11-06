import statsmodels.api as sm

class PoissonRegressorGLM:
    def __init__(self, exog, endo, freq):
        self.model = sm.GLM(
            endo, 
            exog, 
            family=sm.families.Poisson(), 
            freq_weights=freq,
            missing="drop"
        )

        self.result = self.model.fit(maxiter=300)
        self.coef_ = self.result.params
        self.predictions = self.result.predict(exog)
        
        print("Model fitting complete.")

    def get_coefficients(self): 
        '''return coefficients of the trained model'''
        return self.coef_

    def print_summary(self):
        print(self.result.summary())
        
    def predict(self, new_exog):
        '''predict for new data'''
        return self.result.predict(new_exog)

