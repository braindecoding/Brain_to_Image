## EEGdata labels
keys_to_import = ['EEGdata_TP9','EEGdata_TP10','EEGdata_AF7','EEGdata_AF8']
keys_MNIST_IN = ["EEGdata_AF3","EEGdata_AF4","EEGdata_T7","EEGdata_T8","EEGdata_PZ"]
keys_MNIST_MU = ['EEGdata_TP9','EEGdata_FP1','EEGdata_FP2','EEGdata_TP10']
keys_MNIST_EP = ["EEGdata_AF3","EEGdata_AF4","EEGdata_F7","EEGdata_F8","EEGdata_F3","EEGdata_F4","EEGdata_FC5","EEGdata_FC6","EEGdata_T7","EEGdata_T8","EEGdata_P7","EEGdata_P8","EEGdata_O1","EEGdata_O2"]
## Unused data
keys_to_drop = ['digit_label_png','PPGdata_PPG1','PPGdata_PPG2','PPGdata_PPG3','Accdata_X','Accdata_Y','Accdata_Z','Gyrodata_X','Gyrodata_Y','Gyrodata_Z']

# Define delta lower and upper limits: bands as defined by YASA, package ref https://raphaelvallat.com/yasa
eeg_bands = {"Delta":{"low": 0.5, "high": 4},
             "Theta":{"low": 4, "high": 8},
             "Alpha":{"low": 8, "high": 12},
             "Sigma":{"low": 12, "high": 16},
             "Beta":{"low": 16, "high": 30},
             "Gamma":{"low": 30, "high": 40},
             }

## Mind Big Data 2022_MNIST dataset csv formats.
MDB2022_MNIST_IN_params = {
        "digit_label": {"size":1, "type":int},
        "EEGdata_AF3": {"size":256, "type":float},
        "EEGdata_AF4": {"size":256, "type":float},
        "EEGdata_T7": {"size":256, "type":float},
        "EEGdata_T8": {"size":256, "type":float},
        "EEGdata_PZ": {"size":256, "type":float}
    }

MDB2022_MNIST_MU_params = {
        "digit_label": {"size":1, "type":int},
        "EEGdata_TP9": {"size":440, "type":float},
        "EEGdata_FP1": {"size":440, "type":float},
        "EEGdata_FP2": {"size":440, "type":float},
        "EEGdata_TP10": {"size":440, "type":float}
    }

MDB2022_MNIST_EP_params = {
        "digit_label": {"size":1, "type":int},
        "EEGdata_AF3": {"size":256, "type":float},
        "EEGdata_AF4": {"size":256, "type":float},
        "EEGdata_F7": {"size":256, "type":float},
        "EEGdata_F8": {"size":256, "type":float},
        "EEGdata_F3": {"size":256, "type":float},
        "EEGdata_F4": {"size":256, "type":float},
        "EEGdata_FC5": {"size":256, "type":float},
        "EEGdata_FC6": {"size":256, "type":float},
        "EEGdata_T7": {"size":256, "type":float},
        "EEGdata_T8": {"size":256, "type":float},
        "EEGdata_P7": {"size":256, "type":float},
        "EEGdata_P8": {"size":256, "type":float},
        "EEGdata_O1": {"size":256, "type":float},
        "EEGdata_O2": {"size":256, "type":float}
    }


## Mind Big Data dataset csv formats.
Muse2_v017_params = {
        "digit_label": {"size":1, "type":int},
        "digit_label_png": {"size":784, "type":int},
        "EEGdata_TP9": {"size":512, "type":float},
        "EEGdata_AF7": {"size":512, "type":float},
        "EEGdata_AF8": {"size":512, "type":float},
        "EEGdata_TP10": {"size":512, "type":float},
        "PPGdata_PPG1": {"size":512, "type":float},
        "PPGdata_PPG2": {"size":512, "type":float},
        "PPGdata_PPG3": {"size":512, "type":float},
        "Accdata_X": {"size":512, "type":float},
        "Accdata_Y": {"size":512, "type":float},
        "Accdata_Z": {"size": 512, "type":float},
        "Gyrodata_X": {"size": 512, "type":float},
        "Gyrodata_Y": {"size": 512, "type":float},
        "Gyrodata_Z": {"size":512, "type":float}
    }


Muse2_v016Cut2_params = {
            "dataset": {"size":1, "type":int},
            "origin": {"size":1, "type":int},
            "digit_event": {"size":1, "type":int},
            "origin_event_png": {"size":784, "type":int},
            "timestamp": {"size":1, "type":float},
            "EEGdata_TP9": {"size":512, "type":float},
            "EEGdata_TP10": {"size":512, "type":float},
            "PPGdata_PPG1": {"size":512, "type":float},
            "PPGdata_PPG2": {"size":512, "type":float},
            "PPGdata_PPG3": {"size":512, "type":float},
            "Accdata_X": {"size":512, "type":float},
            "Accdata_Y": {"size":512, "type":float},
            "Accdata_Z": {"size":512, "type":float},
            "Gyrodata_X": {"size":512, "type":float},
            "Gyrodata_Y": {"size":512, "type":float},
            "Gyrodata_Z": {"size":512, "type":float},
        }

Muse2_v016Cut3_params = {
            "dataset": {"size":1, "type":int},
            "origin": {"size":1, "type":int},
            "digit_event": {"size":1, "type":int},
            "origin_event_png": {"size":784, "type":int},
            "timestamp": {"size":1, "type":float},
            "EEGdata_TP9": {"size":512, "type":float},
            "EEGdata_AF7": {"size":512, "type":float},
            "EEGdata_TP10": {"size":512, "type":float},
            "PPGdata_PPG1": {"size":512, "type":float},
            "PPGdata_PPG2": {"size":512, "type":float},
            "PPGdata_PPG3": {"size":512, "type":float},
            "Accdata_X": {"size":512, "type":float},
            "Accdata_Y": {"size":512, "type":float},
            "Accdata_Z": {"size":512, "type":float},
            "Gyrodata_X": {"size":512, "type":float},
            "Gyrodata_Y": {"size":512, "type":float},
            "Gyrodata_Z": {"size":512, "type":float},
        }