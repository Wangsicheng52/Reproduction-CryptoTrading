import pandas as pd
import numpy as np
from NNModel_lib import NNModel
from sklearn.preprocessing import StandardScaler
from config import RUN as run_conf
import random
import tensorflow as tf
from compute_indicators_labels_lib import get_dataset
import shap
import pickle
from imbalanced_lib import get_sampler
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import load_model

samples = 100

# 清理当前Keras会话
tf.keras.backend.clear_session()

# 设置随机种子
random.seed(run_conf['seed'])

# 检查可用的GPU数量
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 标准化
scaler = StandardScaler()

# 获取数据集并进行预处理
data = get_dataset(run_conf)
data.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', "Asset_name"], inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

X, y = data.iloc[:, :-1], data.iloc[:, -1]
X = scaler.fit_transform(X, y)

data = pd.DataFrame(X, columns=data.columns[:-1])
data['label'] = y
data.dropna(inplace=True)

# 获取样本平衡算法
sampler = get_sampler(run_conf['balance_algo'])
data = sampler(data)

# 随机采样
data = data.sample(n=samples, axis=0)
X_train, y = data.iloc[:, :-1], data.iloc[:, -1]

print(len(X))

# 自定义激活函数
custom_objects = {'LeakyReLU': LeakyReLU}

# 加载模型时传递自定义激活函数
model = load_model("model_final_%d_%d.h5" % (run_conf['b_window'], run_conf['f_window']), custom_objects=custom_objects)

# SHAP解释器
explainer = shap.Explainer(model.predict, masker=X_train, algorithm='permutation', feature_names=data.columns)
ex = explainer(X_train)

# 打印SHAP值
print(ex)

# SHAP图表展示
shap.plots.beeswarm(ex, max_display=10)
shap.plots.bar(ex, max_display=10)

# 保存SHAP解释器结果
with open("explainer_%d_%d_%d.pickle" % (run_conf['b_window'], run_conf['f_window'], samples), 'wb') as handle:
    pickle.dump(ex, handle, protocol=pickle.HIGHEST_PROTOCOL)

