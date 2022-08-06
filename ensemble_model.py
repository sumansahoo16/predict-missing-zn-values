import numpy as np

class EnsembleModel:
  def __init__(self, models):
    self.models = models

  def predict(self, inputs):
    # Do something here with self.models to calculate your predictions.
    # then return them.
    lower_light = self.models[0]
    upper_light = self.models[1]
    
    lower_preds = lower_light.predict(inputs)
    upper_preds = upper_light.predict(inputs)
    
    #lgbm_preds = lower_preds * 0.0
    lgbm_preds = np.zeros((lower_preds.shape[0], len(self.models) - 2))
    for i in range(2, len(self.models)):
        lgbm_preds[:, i - 2] = self.models[i].predict(inputs)
        
    #lgbm_preds = lgbm_preds / (len(self.models) - 2)
    lgbm_preds = np.median(lgbm_preds, axis = 1)
    
    predictions = [lgbm_preds, lower_preds, upper_preds]
    predictions = np.transpose(predictions)
    return predictions
