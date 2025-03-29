Ivan Usachev   
usa0006

# Imports


```python
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetLarge

```


```python
# Check for GPU availability
if tf.test.gpu_device_name():
    print('GPU device found')
    print('Name:', tf.test.gpu_device_name())
else:
    print("No GPU found")
```

    GPU device found
    Name: /device:GPU:0
    


```python
def show_history(history):
    plt.figure()
    for key in history.history.keys():
        plt.plot(history.epoch, history.history[key], label=key)
    plt.legend()
    plt.tight_layout()

def show_example(train_x, train_y, class_names):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_x[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_y[i]])
    plt.show()
```

# Data loading

https://www.kaggle.com/datasets/trainingdatapro/gender-detection-and-classification-image-dataset

The dataset comprises a collection of photos of people, organized into folders labeled "women" and "men." The dataset contains a variety of images capturing female and male individuals from diverse. backgrounds, age groups, and ethnicities. Total number of photos is 300.


```python
df = pd.read_csv('Data/gender_detection.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>gender</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train/women/0.jpg</td>
      <td>woman</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train/women/1.jpg</td>
      <td>woman</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train/women/2.jpg</td>
      <td>woman</td>
      <td>train</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train/women/3.jpg</td>
      <td>woman</td>
      <td>train</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train/women/4.jpg</td>
      <td>woman</td>
      <td>train</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>test/men/145.jpg</td>
      <td>man</td>
      <td>test</td>
    </tr>
    <tr>
      <th>296</th>
      <td>test/men/146.jpg</td>
      <td>man</td>
      <td>test</td>
    </tr>
    <tr>
      <th>297</th>
      <td>test/men/147.jpg</td>
      <td>man</td>
      <td>test</td>
    </tr>
    <tr>
      <th>298</th>
      <td>test/men/148.jpg</td>
      <td>man</td>
      <td>test</td>
    </tr>
    <tr>
      <th>299</th>
      <td>test/men/149.jpg</td>
      <td>man</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 3 columns</p>
</div>




```python
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>gender</th>
      <th>split</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train/men/53.jpg</td>
      <td>man</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>test/men/116.jpg</td>
      <td>man</td>
      <td>test</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train/men/2.jpg</td>
      <td>man</td>
      <td>train</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train/women/9.jpg</td>
      <td>woman</td>
      <td>train</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train/men/83.jpg</td>
      <td>man</td>
      <td>train</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>train/men/38.jpg</td>
      <td>man</td>
      <td>train</td>
    </tr>
    <tr>
      <th>296</th>
      <td>train/women/71.jpg</td>
      <td>woman</td>
      <td>train</td>
    </tr>
    <tr>
      <th>297</th>
      <td>train/women/106.jpg</td>
      <td>woman</td>
      <td>train</td>
    </tr>
    <tr>
      <th>298</th>
      <td>test/men/120.jpg</td>
      <td>man</td>
      <td>test</td>
    </tr>
    <tr>
      <th>299</th>
      <td>train/women/102.jpg</td>
      <td>woman</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 3 columns</p>
</div>



# Data preprocessing
## Mapping classes to numeric values.


```python
class_map = {'woman': 0, 'man': 1}  
df['class_value'] = df['gender'].map(class_map)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>gender</th>
      <th>split</th>
      <th>class_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train/men/53.jpg</td>
      <td>man</td>
      <td>train</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>test/men/116.jpg</td>
      <td>man</td>
      <td>test</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train/men/2.jpg</td>
      <td>man</td>
      <td>train</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train/women/9.jpg</td>
      <td>woman</td>
      <td>train</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train/men/83.jpg</td>
      <td>man</td>
      <td>train</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>train/men/38.jpg</td>
      <td>man</td>
      <td>train</td>
      <td>1</td>
    </tr>
    <tr>
      <th>296</th>
      <td>train/women/71.jpg</td>
      <td>woman</td>
      <td>train</td>
      <td>0</td>
    </tr>
    <tr>
      <th>297</th>
      <td>train/women/106.jpg</td>
      <td>woman</td>
      <td>train</td>
      <td>0</td>
    </tr>
    <tr>
      <th>298</th>
      <td>test/men/120.jpg</td>
      <td>man</td>
      <td>test</td>
      <td>1</td>
    </tr>
    <tr>
      <th>299</th>
      <td>train/women/102.jpg</td>
      <td>woman</td>
      <td>train</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 4 columns</p>
</div>




```python
class_counts = df['gender'].value_counts()
total_samples = len(df)
class_proportions = class_counts / total_samples

split_counts = df['split'].value_counts()
split_proportions = split_counts / total_samples

split_gender_counts = df.groupby(['split', 'gender']).size()
split_gender_proportions = split_gender_counts / df.groupby('split').size()

print("Overall class proportions:\n", class_proportions)
print("\nSplit proportions:\n", split_proportions)
print("\nProportions of each gender within each split:\n", split_gender_proportions)
```

    Overall class proportions:
     gender
    man      0.5
    woman    0.5
    Name: count, dtype: float64
    
    Split proportions:
     split
    train    0.733333
    test     0.266667
    Name: count, dtype: float64
    
    Proportions of each gender within each split:
     split  gender
    test   man       0.5
           woman     0.5
    train  man       0.5
           woman     0.5
    dtype: float64
    

The dataset is balanced, we should use accuracy metrics.


```python
train_set = df[df['split'] == 'train']
test_set = df[df['split'] == 'test']

print(f"Train set size: {train_set.shape[0]}")
print(f"Test set size: {test_set.shape[0]}")
```

    Train set size: 220
    Test set size: 80
    

## Converting images to the RGB color space and resizing them to dimensions of 224x224


```python
train_x = np.zeros((train_set.shape[0], 224, 224, 3))

for i in range(train_set.shape[0]):
    
    image = cv2.imread('Data/' + train_set["file"].values[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    train_x[i] = image / 255.0

train_y = np.array(train_set["class_value"])
```


```python
test_x = np.zeros((test_set.shape[0], 224, 224, 3))

for i in range(test_set.shape[0]):
    
    image = cv2.imread('Data/' + test_set["file"].values[i])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224,224))
    test_x[i] = image / 255.0

test_y = np.array(test_set["class_value"])
```


```python
print('Train Data: ',train_x.shape)
print('Test Data: ',test_x.shape)
```

    Train Data:  (220, 224, 224, 3)
    Test Data:  (80, 224, 224, 3)
    


```python
class_names = ['woman', 'man']
class_count = len(class_names)
class_count
```




    2




```python
show_example(train_x, train_y, class_names)
```


    
![png](final_files/final_18_0.png)
    


# CNN Models
## Model from scratch

We should try model from scratch from the previous project first


```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_18 (Conv2D)           (None, 222, 222, 32)      896       
    _________________________________________________________________
    max_pooling2d_13 (MaxPooling (None, 111, 111, 32)      0         
    _________________________________________________________________
    dropout_12 (Dropout)         (None, 111, 111, 32)      0         
    _________________________________________________________________
    conv2d_19 (Conv2D)           (None, 111, 111, 64)      18496     
    _________________________________________________________________
    conv2d_20 (Conv2D)           (None, 111, 111, 64)      36928     
    _________________________________________________________________
    max_pooling2d_14 (MaxPooling (None, 55, 55, 64)        0         
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 55, 55, 64)        0         
    _________________________________________________________________
    conv2d_21 (Conv2D)           (None, 53, 53, 64)        36928     
    _________________________________________________________________
    conv2d_22 (Conv2D)           (None, 51, 51, 64)        36928     
    _________________________________________________________________
    max_pooling2d_15 (MaxPooling (None, 25, 25, 64)        0         
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 25, 25, 64)        0         
    _________________________________________________________________
    flatten_5 (Flatten)          (None, 40000)             0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 128)               5120128   
    _________________________________________________________________
    dense_11 (Dense)             (None, 1)                 129       
    =================================================================
    Total params: 5,250,433
    Trainable params: 5,250,433
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(optimizer=keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(from_logits=False), metrics = keras.metrics.BinaryAccuracy())
```


```python
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights.best.hdf5',
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
```


```python
history=model.fit(train_x, train_y, epochs=10, validation_split=0.2, batch_size=4, callbacks=model_checkpoint_callback)
show_history(history)
```

    Epoch 1/10
    44/44 [==============================] - 2s 22ms/step - loss: 0.6957 - binary_accuracy: 0.5511 - val_loss: 0.6939 - val_binary_accuracy: 0.4318
    Epoch 2/10
    44/44 [==============================] - 1s 14ms/step - loss: 0.7003 - binary_accuracy: 0.5227 - val_loss: 0.6980 - val_binary_accuracy: 0.4545
    Epoch 3/10
    44/44 [==============================] - 1s 14ms/step - loss: 0.7151 - binary_accuracy: 0.5795 - val_loss: 0.6939 - val_binary_accuracy: 0.4545
    Epoch 4/10
    44/44 [==============================] - 1s 14ms/step - loss: 0.6932 - binary_accuracy: 0.5114 - val_loss: 0.6945 - val_binary_accuracy: 0.4545
    Epoch 5/10
    44/44 [==============================] - 1s 14ms/step - loss: 0.6933 - binary_accuracy: 0.5114 - val_loss: 0.6943 - val_binary_accuracy: 0.4545
    Epoch 6/10
    44/44 [==============================] - 1s 14ms/step - loss: 0.6930 - binary_accuracy: 0.5114 - val_loss: 0.6945 - val_binary_accuracy: 0.4545
    Epoch 7/10
    44/44 [==============================] - 1s 14ms/step - loss: 0.6931 - binary_accuracy: 0.5114 - val_loss: 0.6948 - val_binary_accuracy: 0.4545
    Epoch 8/10
    44/44 [==============================] - 1s 14ms/step - loss: 0.6930 - binary_accuracy: 0.5114 - val_loss: 0.6951 - val_binary_accuracy: 0.4545
    Epoch 9/10
    44/44 [==============================] - 1s 14ms/step - loss: 0.6930 - binary_accuracy: 0.5114 - val_loss: 0.6947 - val_binary_accuracy: 0.4545
    Epoch 10/10
    44/44 [==============================] - 1s 14ms/step - loss: 0.6931 - binary_accuracy: 0.5114 - val_loss: 0.6948 - val_binary_accuracy: 0.4545
    


    
![png](final_files/final_24_1.png)
    



```python
model.load_weights("weights.best.hdf5")
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test accuracy: ', test_acc)
```

    3/3 [==============================] - 0s 41ms/step - loss: 0.6917 - binary_accuracy: 0.5125
    Test accuracy:  0.512499988079071
    

This model doesnt work well with this dataset, we should try something else


```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(16, (3,3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_23 (Conv2D)           (None, 222, 222, 32)      896       
    _________________________________________________________________
    max_pooling2d_16 (MaxPooling (None, 111, 111, 32)      0         
    _________________________________________________________________
    conv2d_24 (Conv2D)           (None, 109, 109, 16)      4624      
    _________________________________________________________________
    flatten_6 (Flatten)          (None, 190096)            0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 64)                12166208  
    _________________________________________________________________
    dense_13 (Dense)             (None, 1)                 65        
    =================================================================
    Total params: 12,171,793
    Trainable params: 12,171,793
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(optimizer=keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(from_logits=False), metrics = keras.metrics.BinaryAccuracy())
```


```python
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights.best.hdf5',
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
```


```python
history=model.fit(train_x, train_y, epochs=10, validation_split=0.2, batch_size=4, callbacks=model_checkpoint_callback)
show_history(history)
```

    Epoch 1/10
    44/44 [==============================] - 1s 16ms/step - loss: 1.0194 - binary_accuracy: 0.4602 - val_loss: 0.6928 - val_binary_accuracy: 0.5000
    Epoch 2/10
    44/44 [==============================] - 0s 7ms/step - loss: 0.6915 - binary_accuracy: 0.7159 - val_loss: 0.6817 - val_binary_accuracy: 0.5455
    Epoch 3/10
    44/44 [==============================] - 0s 7ms/step - loss: 0.6219 - binary_accuracy: 0.6875 - val_loss: 0.6599 - val_binary_accuracy: 0.6364
    Epoch 4/10
    44/44 [==============================] - 0s 7ms/step - loss: 0.3124 - binary_accuracy: 0.8807 - val_loss: 1.3148 - val_binary_accuracy: 0.5682
    Epoch 5/10
    44/44 [==============================] - 0s 7ms/step - loss: 0.5919 - binary_accuracy: 0.8409 - val_loss: 0.6207 - val_binary_accuracy: 0.6591
    Epoch 6/10
    44/44 [==============================] - 0s 7ms/step - loss: 0.3624 - binary_accuracy: 0.9091 - val_loss: 0.9414 - val_binary_accuracy: 0.6136
    Epoch 7/10
    44/44 [==============================] - 0s 7ms/step - loss: 0.1478 - binary_accuracy: 0.9489 - val_loss: 1.0044 - val_binary_accuracy: 0.5227
    Epoch 8/10
    44/44 [==============================] - 0s 7ms/step - loss: 0.0484 - binary_accuracy: 0.9830 - val_loss: 1.8267 - val_binary_accuracy: 0.5455
    Epoch 9/10
    44/44 [==============================] - 0s 7ms/step - loss: 0.0250 - binary_accuracy: 0.9943 - val_loss: 2.3094 - val_binary_accuracy: 0.5682
    Epoch 10/10
    44/44 [==============================] - 0s 7ms/step - loss: 0.0118 - binary_accuracy: 0.9943 - val_loss: 1.4814 - val_binary_accuracy: 0.5227
    


    
![png](final_files/final_30_1.png)
    



```python
model.load_weights("weights.best.hdf5")
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test accuracy: ', test_acc)
```

    3/3 [==============================] - 0s 17ms/step - loss: 0.7510 - binary_accuracy: 0.6000
    Test accuracy:  0.6000000238418579
    

This model showed better results than the model from previous project, but it is overfitted at the beginning. Lets add some more layers and Dropout layer


```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
```

    Model: "sequential_10"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_34 (Conv2D)           (None, 222, 222, 32)      896       
    _________________________________________________________________
    max_pooling2d_26 (MaxPooling (None, 111, 111, 32)      0         
    _________________________________________________________________
    dropout_24 (Dropout)         (None, 111, 111, 32)      0         
    _________________________________________________________________
    conv2d_35 (Conv2D)           (None, 109, 109, 64)      18496     
    _________________________________________________________________
    max_pooling2d_27 (MaxPooling (None, 54, 54, 64)        0         
    _________________________________________________________________
    dropout_25 (Dropout)         (None, 54, 54, 64)        0         
    _________________________________________________________________
    conv2d_36 (Conv2D)           (None, 52, 52, 128)       73856     
    _________________________________________________________________
    max_pooling2d_28 (MaxPooling (None, 26, 26, 128)       0         
    _________________________________________________________________
    dropout_26 (Dropout)         (None, 26, 26, 128)       0         
    _________________________________________________________________
    flatten_10 (Flatten)         (None, 86528)             0         
    _________________________________________________________________
    dense_20 (Dense)             (None, 128)               11075712  
    _________________________________________________________________
    dense_21 (Dense)             (None, 1)                 129       
    =================================================================
    Total params: 11,169,089
    Trainable params: 11,169,089
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(optimizer=keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(from_logits=False), metrics = keras.metrics.BinaryAccuracy())

```


```python
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights.best.hdf5',
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
```


```python
history=model.fit(train_x, train_y, epochs=10, validation_split=0.2, batch_size=4, callbacks=model_checkpoint_callback)
show_history(history)
```

    Epoch 1/10
    44/44 [==============================] - 1s 15ms/step - loss: 1.0739 - binary_accuracy: 0.5398 - val_loss: 0.7217 - val_binary_accuracy: 0.4545
    Epoch 2/10
    44/44 [==============================] - 0s 10ms/step - loss: 0.6967 - binary_accuracy: 0.4886 - val_loss: 0.6934 - val_binary_accuracy: 0.4545
    Epoch 3/10
    44/44 [==============================] - 0s 10ms/step - loss: 0.6937 - binary_accuracy: 0.5568 - val_loss: 0.6935 - val_binary_accuracy: 0.4545
    Epoch 4/10
    44/44 [==============================] - 0s 10ms/step - loss: 0.6935 - binary_accuracy: 0.5625 - val_loss: 0.6921 - val_binary_accuracy: 0.4773
    Epoch 5/10
    44/44 [==============================] - 0s 10ms/step - loss: 0.6953 - binary_accuracy: 0.5511 - val_loss: 0.6932 - val_binary_accuracy: 0.5000
    Epoch 6/10
    44/44 [==============================] - 0s 10ms/step - loss: 0.6851 - binary_accuracy: 0.6193 - val_loss: 0.6477 - val_binary_accuracy: 0.6591
    Epoch 7/10
    44/44 [==============================] - 0s 10ms/step - loss: 0.6235 - binary_accuracy: 0.6080 - val_loss: 0.6716 - val_binary_accuracy: 0.5682
    Epoch 8/10
    44/44 [==============================] - 0s 10ms/step - loss: 0.5658 - binary_accuracy: 0.7386 - val_loss: 0.6775 - val_binary_accuracy: 0.5682
    Epoch 9/10
    44/44 [==============================] - 0s 10ms/step - loss: 0.5926 - binary_accuracy: 0.7443 - val_loss: 0.6335 - val_binary_accuracy: 0.5909
    Epoch 10/10
    44/44 [==============================] - 0s 10ms/step - loss: 0.3801 - binary_accuracy: 0.8011 - val_loss: 0.8603 - val_binary_accuracy: 0.6818
    


    
![png](final_files/final_36_1.png)
    



```python
model.load_weights("weights.best.hdf5")
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test accuracy: ', test_acc)
```

    3/3 [==============================] - 0s 29ms/step - loss: 0.6636 - binary_accuracy: 0.6625
    Test accuracy:  0.6625000238418579
    

This model is a little bit better, but it is still overfitted. Lets add BatchNormalization layer and try more epochs


```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_10 (Conv2D)           (None, 222, 222, 32)      896       
    _________________________________________________________________
    batch_normalization (BatchNo (None, 222, 222, 32)      128       
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 111, 111, 32)      0         
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 111, 111, 32)      0         
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 111, 111, 64)      18496     
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 111, 111, 64)      256       
    _________________________________________________________________
    max_pooling2d_8 (MaxPooling2 (None, 55, 55, 64)        0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 55, 55, 64)        0         
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 53, 53, 128)       73856     
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 53, 53, 128)       512       
    _________________________________________________________________
    max_pooling2d_9 (MaxPooling2 (None, 26, 26, 128)       0         
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 26, 26, 128)       0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 86528)             0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 128)               11075712  
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 128)               512       
    _________________________________________________________________
    dense_7 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 11,170,497
    Trainable params: 11,169,793
    Non-trainable params: 704
    _________________________________________________________________
    


```python
model.compile(optimizer=keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(from_logits=False), metrics = keras.metrics.BinaryAccuracy())
```


```python
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights.best.hdf5',
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
```


```python
history=model.fit(train_x, train_y, epochs=20, validation_split=0.2, batch_size=4, callbacks=model_checkpoint_callback)
show_history(history)

```

    Epoch 1/20
    44/44 [==============================] - 2s 19ms/step - loss: 1.0288 - binary_accuracy: 0.5341 - val_loss: 1.4519 - val_binary_accuracy: 0.4545
    Epoch 2/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.6835 - binary_accuracy: 0.5909 - val_loss: 2.5913 - val_binary_accuracy: 0.3864
    Epoch 3/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.5709 - binary_accuracy: 0.6989 - val_loss: 1.7565 - val_binary_accuracy: 0.4318
    Epoch 4/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.5671 - binary_accuracy: 0.7102 - val_loss: 0.7741 - val_binary_accuracy: 0.5455
    Epoch 5/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.5985 - binary_accuracy: 0.6875 - val_loss: 0.8724 - val_binary_accuracy: 0.5227
    Epoch 6/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.5899 - binary_accuracy: 0.6989 - val_loss: 0.8384 - val_binary_accuracy: 0.5455
    Epoch 7/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.5406 - binary_accuracy: 0.7557 - val_loss: 0.7821 - val_binary_accuracy: 0.5000
    Epoch 8/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.4658 - binary_accuracy: 0.7727 - val_loss: 0.7548 - val_binary_accuracy: 0.6364
    Epoch 9/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.5175 - binary_accuracy: 0.7670 - val_loss: 0.7732 - val_binary_accuracy: 0.4773
    Epoch 10/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.4945 - binary_accuracy: 0.7784 - val_loss: 0.7412 - val_binary_accuracy: 0.6364
    Epoch 11/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.5502 - binary_accuracy: 0.7500 - val_loss: 0.9233 - val_binary_accuracy: 0.5227
    Epoch 12/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.6340 - binary_accuracy: 0.6705 - val_loss: 0.7823 - val_binary_accuracy: 0.7045
    Epoch 13/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.5483 - binary_accuracy: 0.7216 - val_loss: 0.5982 - val_binary_accuracy: 0.7045
    Epoch 14/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.3954 - binary_accuracy: 0.8352 - val_loss: 0.8993 - val_binary_accuracy: 0.6818
    Epoch 15/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.4237 - binary_accuracy: 0.8125 - val_loss: 0.7266 - val_binary_accuracy: 0.6364
    Epoch 16/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.3638 - binary_accuracy: 0.8636 - val_loss: 0.7776 - val_binary_accuracy: 0.6136
    Epoch 17/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.4189 - binary_accuracy: 0.8352 - val_loss: 0.6944 - val_binary_accuracy: 0.7273
    Epoch 18/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.4292 - binary_accuracy: 0.7841 - val_loss: 0.6253 - val_binary_accuracy: 0.5909
    Epoch 19/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.3720 - binary_accuracy: 0.8580 - val_loss: 0.7096 - val_binary_accuracy: 0.5227
    Epoch 20/20
    44/44 [==============================] - 1s 14ms/step - loss: 0.3192 - binary_accuracy: 0.8295 - val_loss: 0.5336 - val_binary_accuracy: 0.7273
    


    
![png](final_files/final_42_1.png)
    



```python
model.load_weights("weights.best.hdf5")
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test accuracy: ', test_acc)
```

    3/3 [==============================] - 0s 33ms/step - loss: 0.6418 - binary_accuracy: 0.7125
    Test accuracy:  0.7124999761581421
    

## ResNet50
### Data preprocessing for ResNet50 and MobileNetV2 Models


```python
train_x = np.zeros((train_set.shape[0], 224, 224, 3))

for i in range(train_set.shape[0]):
    
    image = cv2.imread('Data/' + train_set["file"].values[i])
    image = cv2.resize(image, (224,224))
    train_x[i] = image 

test_x = np.zeros((test_set.shape[0], 224, 224, 3))

for i in range(test_set.shape[0]):
    
    image = cv2.imread('Data/' + test_set["file"].values[i])
    image = cv2.resize(image, (224,224))
    test_x[i] = image 

train_y = np.array(train_set["class_value"])
test_y = np.array(test_set["class_value"])
```


```python
base_model = ResNet50(
    weights='imagenet', 
    input_shape=(224, 224, 3),
    include_top=False) 
```


```python
base_model.trainable = False
```


```python
inputs = keras.Input(shape=(224, 224, 3), dtype=tf.uint8)
x = tf.cast(inputs, tf.float32)
x = tf.keras.applications.resnet50.preprocess_input(x)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)
```


```python
model.compile(optimizer=keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(from_logits=False), metrics = keras.metrics.BinaryAccuracy())
```


```python
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights.best.hdf5',
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
```


```python
history=model.fit(train_x, train_y, epochs=10, validation_split=0.2, batch_size=4, callbacks=model_checkpoint_callback)
show_history(history)
```

    Epoch 1/10
    44/44 [==============================] - 6s 42ms/step - loss: 0.6443 - binary_accuracy: 0.6420 - val_loss: 0.4694 - val_binary_accuracy: 0.7727
    Epoch 2/10
    44/44 [==============================] - 1s 19ms/step - loss: 0.3669 - binary_accuracy: 0.8864 - val_loss: 0.3736 - val_binary_accuracy: 0.8409
    Epoch 3/10
    44/44 [==============================] - 1s 19ms/step - loss: 0.2857 - binary_accuracy: 0.9261 - val_loss: 0.3611 - val_binary_accuracy: 0.8182
    Epoch 4/10
    44/44 [==============================] - 1s 19ms/step - loss: 0.2215 - binary_accuracy: 0.9489 - val_loss: 0.3361 - val_binary_accuracy: 0.8864
    Epoch 5/10
    44/44 [==============================] - 1s 19ms/step - loss: 0.1778 - binary_accuracy: 0.9602 - val_loss: 0.3263 - val_binary_accuracy: 0.8636
    Epoch 6/10
    44/44 [==============================] - 1s 19ms/step - loss: 0.1488 - binary_accuracy: 0.9545 - val_loss: 0.3382 - val_binary_accuracy: 0.8636
    Epoch 7/10
    44/44 [==============================] - 1s 19ms/step - loss: 0.1314 - binary_accuracy: 0.9830 - val_loss: 0.3209 - val_binary_accuracy: 0.8409
    Epoch 8/10
    44/44 [==============================] - 1s 19ms/step - loss: 0.1139 - binary_accuracy: 0.9830 - val_loss: 0.3242 - val_binary_accuracy: 0.8864
    Epoch 9/10
    44/44 [==============================] - 1s 19ms/step - loss: 0.1059 - binary_accuracy: 0.9886 - val_loss: 0.3742 - val_binary_accuracy: 0.8182
    Epoch 10/10
    44/44 [==============================] - 1s 19ms/step - loss: 0.0905 - binary_accuracy: 0.9943 - val_loss: 0.3301 - val_binary_accuracy: 0.8864
    


    
![png](final_files/final_51_1.png)
    



```python
model.load_weights("weights.best.hdf5")
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test accuracy: ', test_acc)
```

    3/3 [==============================] - 3s 406ms/step - loss: 0.3809 - binary_accuracy: 0.8000
    Test accuracy:  0.800000011920929
    

## MobileNetV2


```python
base_model = MobileNetV2(
    weights='imagenet',  
    input_shape=(224, 224, 3),
    include_top=False)  
```


```python
base_model.trainable = False
```


```python
inputs = keras.Input(shape=(224, 224, 3), dtype=tf.uint8)
x = tf.cast(inputs, tf.float32)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)
```


```python
model.compile(optimizer=keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(from_logits=False), metrics = keras.metrics.BinaryAccuracy())
```


```python
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights.best.hdf5',
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
```


```python
history=model.fit(train_x, train_y, epochs=10, validation_split=0.2, batch_size=4, callbacks=model_checkpoint_callback)
show_history(history)
```

    Epoch 1/10
    44/44 [==============================] - 5s 30ms/step - loss: 0.6674 - binary_accuracy: 0.5795 - val_loss: 0.5950 - val_binary_accuracy: 0.5909
    Epoch 2/10
    44/44 [==============================] - 0s 10ms/step - loss: 0.4876 - binary_accuracy: 0.7727 - val_loss: 0.5307 - val_binary_accuracy: 0.7045
    Epoch 3/10
    44/44 [==============================] - 0s 11ms/step - loss: 0.3934 - binary_accuracy: 0.8295 - val_loss: 0.5303 - val_binary_accuracy: 0.7727
    Epoch 4/10
    44/44 [==============================] - 0s 11ms/step - loss: 0.3349 - binary_accuracy: 0.8636 - val_loss: 0.4598 - val_binary_accuracy: 0.7955
    Epoch 5/10
    44/44 [==============================] - 0s 11ms/step - loss: 0.2862 - binary_accuracy: 0.9034 - val_loss: 0.4393 - val_binary_accuracy: 0.7955
    Epoch 6/10
    44/44 [==============================] - 0s 11ms/step - loss: 0.2582 - binary_accuracy: 0.9261 - val_loss: 0.4349 - val_binary_accuracy: 0.7955
    Epoch 7/10
    44/44 [==============================] - 0s 10ms/step - loss: 0.2412 - binary_accuracy: 0.9432 - val_loss: 0.5077 - val_binary_accuracy: 0.8182
    Epoch 8/10
    44/44 [==============================] - 0s 11ms/step - loss: 0.2107 - binary_accuracy: 0.9602 - val_loss: 0.4122 - val_binary_accuracy: 0.8182
    Epoch 9/10
    44/44 [==============================] - 0s 11ms/step - loss: 0.1836 - binary_accuracy: 0.9716 - val_loss: 0.3890 - val_binary_accuracy: 0.8182
    Epoch 10/10
    44/44 [==============================] - 0s 11ms/step - loss: 0.1685 - binary_accuracy: 0.9886 - val_loss: 0.3811 - val_binary_accuracy: 0.8409
    


    
![png](final_files/final_59_1.png)
    



```python
model.load_weights("weights.best.hdf5")
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test accuracy: ', test_acc)
```

    3/3 [==============================] - 2s 217ms/step - loss: 0.4219 - binary_accuracy: 0.8000
    Test accuracy:  0.800000011920929
    

## NASNetLarge
### Data preprocessing for NASNetLarge Model


```python
train_x = np.zeros((train_set.shape[0], 331, 331, 3))

for i in range(train_set.shape[0]):
    
    image = cv2.imread('Data/' + train_set["file"].values[i])
    image = cv2.resize(image, (331,331))
    train_x[i] = image 

test_x = np.zeros((test_set.shape[0], 331, 331, 3))

for i in range(test_set.shape[0]):
    
    image = cv2.imread('Data/' + test_set["file"].values[i])
    image = cv2.resize(image, (331,331))
    test_x[i] = image 

train_y = np.array(train_set["class_value"])
test_y = np.array(test_set["class_value"])
```


```python
base_model = NASNetLarge(
    weights='imagenet',  
    input_shape=(331, 331, 3),
    include_top=False)  
```


```python
base_model.trainable = False
```


```python
from tensorflow.keras.applications.nasnet import preprocess_input

inputs = keras.Input(shape=(331, 331, 3), dtype=tf.uint8)
x = tf.cast(inputs, tf.float32)
x = preprocess_input(x)

x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)
```


```python
model.compile(optimizer=keras.optimizers.Adam(), loss = keras.losses.BinaryCrossentropy(from_logits=False), metrics = keras.metrics.BinaryAccuracy())
```


```python
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights.best.hdf5',
    save_weights_only=True,
    monitor='val_loss',
    mode='auto',
    save_best_only=True)
```


```python
history=model.fit(train_x, train_y, epochs=10, validation_split=0.2, batch_size=4, callbacks=model_checkpoint_callback)
show_history(history)
```

    Epoch 1/10
    44/44 [==============================] - 36s 295ms/step - loss: 0.5382 - binary_accuracy: 0.7614 - val_loss: 0.4431 - val_binary_accuracy: 0.7955
    Epoch 2/10
    44/44 [==============================] - 6s 130ms/step - loss: 0.3251 - binary_accuracy: 0.8920 - val_loss: 0.3736 - val_binary_accuracy: 0.8409
    Epoch 3/10
    44/44 [==============================] - 6s 131ms/step - loss: 0.2508 - binary_accuracy: 0.9261 - val_loss: 0.3443 - val_binary_accuracy: 0.8409
    Epoch 4/10
    44/44 [==============================] - 6s 131ms/step - loss: 0.2005 - binary_accuracy: 0.9432 - val_loss: 0.3474 - val_binary_accuracy: 0.8409
    Epoch 5/10
    44/44 [==============================] - 6s 132ms/step - loss: 0.1621 - binary_accuracy: 0.9773 - val_loss: 0.3384 - val_binary_accuracy: 0.8182
    Epoch 6/10
    44/44 [==============================] - 6s 132ms/step - loss: 0.1390 - binary_accuracy: 0.9773 - val_loss: 0.3333 - val_binary_accuracy: 0.8182
    Epoch 7/10
    44/44 [==============================] - 6s 131ms/step - loss: 0.1196 - binary_accuracy: 0.9830 - val_loss: 0.3476 - val_binary_accuracy: 0.7955
    Epoch 8/10
    44/44 [==============================] - 6s 132ms/step - loss: 0.1020 - binary_accuracy: 0.9943 - val_loss: 0.3511 - val_binary_accuracy: 0.7955
    Epoch 9/10
    44/44 [==============================] - 6s 133ms/step - loss: 0.0904 - binary_accuracy: 0.9943 - val_loss: 0.3336 - val_binary_accuracy: 0.7727
    Epoch 10/10
    44/44 [==============================] - 6s 133ms/step - loss: 0.0788 - binary_accuracy: 1.0000 - val_loss: 0.3540 - val_binary_accuracy: 0.7955
    


    
![png](final_files/final_68_1.png)
    



```python
model.load_weights("weights.best.hdf5")
test_loss, test_acc = model.evaluate(test_x, test_y)
print('Test accuracy: ', test_acc)
```

    3/3 [==============================] - 12s 1s/step - loss: 0.3255 - binary_accuracy: 0.8500
    Test accuracy:  0.8500000238418579
    

# Summary

Four models were tested: a model built from scratch, ResNet50, MobileNetV2 and NASNetLarge model. The NASNetLarge model demonstrated the best performance among all four models, although it had a slow training process. The second-best performers were the ResNet50 and the MobileNetV2 models, which had similar results. The scratch model did not achieve the best performance but it became the best model made by author of this project.
