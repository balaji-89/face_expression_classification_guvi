
from Lib.LHC_Net_Model import LHC_ResNet34
from Lib.Utils import Check_Unique, TTA_Inference, etl_data
import tensorflow as tf
from random import random
import numpy as np
import os


os.environ['TF_DETERMINISTIC_OPS'] = '1'


tf.random.set_seed(0)
# random.seed(0)
np.random.seed(0)



# DATA IMPORT
# path_val = "Data/data_val.csv"
# path_test = "Data/data_test.csv"
# validation_imagesRGB, validation_labels = etl_data(path_val)
testing_imagesRGB, testing_labels = etl_data()







# MODEL IMPORT
exec(open("./Lib/LHC_Net_Model.py").read())
Params = {'num_heads': [8, 8, 7, 7, 1],
          'att_embed_dim': [196, 196, 56, 14, 25],
          'kernel_size': [3, 3, 3, 3, 3],
          'pool_size': [3, 3, 3, 3, 3],
          'norm_c': [1, 1, 1, 1, 1]}
model = LHC_ResNet34(input_shape=(224, 224, 3), num_classes=7, att_params=Params)
x0 = np.ones(shape=(10, 224, 224, 3), dtype='float32')
y0 = model(x0)
model.load_weights('./Downloaded_Models/LHC_Net/LHC_Net')



#METRICS
print("LHC_Net Perf:")
pred_test = model.predict(testing_imagesRGB)
perf_test = tf.keras.metrics.CategoricalAccuracy(dtype='float64')(testing_labels, pred_test).numpy()
print('Test Perf: ', '%.17f' % perf_test)
pred_test_uniqueness = Check_Unique(pred_test)
print('Test Pred Repeated: ', pred_test_uniqueness)




# TTA
tf.config.run_functions_eagerly(True)
tta_pred_test = TTA_Inference(model, testing_imagesRGB)
tf.config.run_functions_eagerly(False)





# METRICS TTA
tta_perf_test = tf.keras.metrics.CategoricalAccuracy(dtype='float64')(testing_labels, tta_pred_test).numpy()
print('TTA Test Perf: ', '%.17f' % tta_perf_test)
print("")
tta_pred_test_uniqueness = Check_Unique(tta_pred_test)
print('TTA Test Pred Repeated: ', tta_pred_test_uniqueness)
print("")
print("")
print("")

# RESET
for element in dir():
    if element[0:2] != "__":
        del globals()[element]
del element