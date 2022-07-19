
import pickle
import pandas as pd
import numpy as np

pkl = open('prediction.pickle', 'rb')
clf = pickle.load(pkl) 

pred=clf.get_prediction(start=pd.to_datetime('2024-02-01'),dynamic=False)

pred_c1=pred.conf_int()

print(pred.predicted_mean)


predicted_value=pred.predicted_mean[0]

print(predicted_value)