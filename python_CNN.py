import pickle

from keras.preprocessing.image import ImageDataGenerator
#資料增加
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('目標資料夾',
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('目標資料夾',
                                                 target_size = (128,128),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()
classifier.add(Convolution2D(32,(5,5),input_shape = (128, 128, 3),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2))) 
   
classifier.add(Convolution2D(64,(5,5),activation='relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Flatten())
classifier.add(Dense(output_dim = 64,activation ='relu'))
classifier.add(Dense(output_dim = 32,activation ='relu'))

#output layer
classifier.add(Dense(output_dim= 4,activation = 'softmax'))
classifier.summary()


classifier.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
classifier.fit_generator(training_set,
                         steps_per_epoch = 250,
                         epochs = 1)

test_loss, test_acc = classifier.evaluate_generator(generator = test_set,steps = 9.375)

training_set.class_indices

pred1 = classifier.predict_generator(test_set,steps = 9.375)

with open('training_data','wb') as f:
     pickle.dump(classifier, f)

#測試圖片
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
   
img = image.load_img(path="圖片名稱",grayscale = False,target_size=(128,128,3))
plt.imshow(img)
img = image.img_to_array(img)
img = img.reshape(1,128, 128,3).astype('float32')
img_norm = img/255
result = classifier.predict(img_norm)
img_class = classifier.predict_classes(img_norm)
if img_class[0]== 0:
   classname ='airplane'
elif img_class[0]==1:
  classname ='bus'
elif img_class[0]==2:
  classname ='car'
else:
  classname ='motorcycle'

print('class:',classname)