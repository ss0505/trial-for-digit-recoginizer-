import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import tensorflow as tf
from tensorflow.keras import layers, models


# データ読み込み
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train_data_len = len(train_data)
test_data_len = len(test_data)

print("train_data Length:{}".format(train_data_len))
print("test_data  Length:{}".format(test_data_len))


train_data_y = train_data["label"]
train_data_x = train_data.drop(columns="label")

train_data_x = train_data_x.astype('float64').values.reshape((train_data_len, 28, 28, 1))
test_data = test_data.astype('float64').values.reshape((test_data_len, 28, 28, 1))

train_data_x /= 255.0
test_data /= 255.0


# TensorFlowのチュートリアル参考
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data_x, train_data_y, epochs=10)

# 予測を取得
prediction = model.predict(test_data)

# 各サンプルの予測クラスを取得
predicted_classes = np.argmax(prediction, axis=1)

# DataFrameに変換
output = pd.DataFrame({
    "ImageId": np.arange(1, len(predicted_classes) + 1),
    "Label": predicted_classes
})

# CSVに保存
output.to_csv('digit_recognizer_CNN1a.csv', index=False)
print("output was successfully saved!")


#TIPS

# https://chatgpt.com/share/674f5001-d878-800e-96e1-8eaa96be30fa
# 上記コードの改善点を挙げてもらった