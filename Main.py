#使用迁移学习进行水母物种分类

#导入模块
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from IPython.display import display, Image
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121
from keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50, DenseNet121, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image


#数据加载和预处理

#定义数据集文件夹的路径
current_folder = os.path.dirname(os.path.abspath(__file__))
Moon_jellyfish_folder = os.path.join(current_folder, "Moon_jellyfish")
barrel_jellyfish_folder = os.path.join(current_folder, "barrel_jellyfish")
blue_jellyfish_folder = os.path.join(current_folder, "blue_jellyfish")
compass_jellyfish_folder = os.path.join(current_folder, "compass_jellyfish")
lions_mane_jellyfish_folder = os.path.join(current_folder, "lions_mane_jellyfish")
mauve_stinger_jellyfish_folder = os.path.join(current_folder, "mauve_stinger_jellyfish")



#函数加载和预处理图像
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (224, 224))  #调整为模型的固定大小
            images.append(img)
    return images

#为每个情绪加载图像和标签
Moon_images = load_images_from_folder(Moon_jellyfish_folder)
barrel_images = load_images_from_folder(barrel_jellyfish_folder)
blue_images = load_images_from_folder(blue_jellyfish_folder)
compass_images = load_images_from_folder(compass_jellyfish_folder)
lions_mane_images = load_images_from_folder(lions_mane_jellyfish_folder)
mauve_stinger_images = load_images_from_folder(mauve_stinger_jellyfish_folder)

#为每个情绪类别创建标签
Moon_labels = [0] * len(Moon_images)
barrel_labels = [1] * len(barrel_images)
blue_labels = [2] * len(blue_images)
compass_labels = [3] * len(compass_images)
lions_mane_labels = [4] * len(lions_mane_images)
mauve_stinger_labels = [5] * len(mauve_stinger_images)

all_labels = Moon_labels+barrel_labels+blue_labels+compass_labels+lions_mane_labels+mauve_stinger_labels 

print('Moon_jellyfish_images: ',len(Moon_images))
print('barrel_jellyfish_images: ',len(barrel_images))
print('blue_jellyfish_images: ',len(blue_images))
print('compass_jellyfish_images: ',len(compass_images))
print('lions_mane_jellyfish_images: ',len(lions_mane_images))
print('mauve_stinger_jellyfish_images: ',len(mauve_stinger_images))

import matplotlib.pyplot as plt
import seaborn as sns

#连接图像和标签
X = np.array(Moon_images + barrel_images + blue_images + compass_images + lions_mane_images + mauve_stinger_images)
y = np.array(Moon_labels + barrel_labels + blue_labels + compass_labels + lions_mane_labels + mauve_stinger_labels)

#将像素值归一化到[0,1]范围
X = X.astype('float32') / 255.0

#一次性对标签进行编码
y = to_categorical(y, num_classes=6)


#将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_shape_resnet = (224, 224, 3)
input_shape_densenet = (224, 224, 3)
input_shape_efficientnet = (224, 224, 3)

def resize_images(images, input_shape):
    resized_images = []
    for img in images:
        img_resized = cv2.resize(img, (input_shape[0], input_shape[1]))
        img_resized = np.expand_dims(img_resized, axis=-1)
        img_resized = np.repeat(img_resized, 3, axis=-1)  #添加三个通道将灰度转换为RGB
        resized_images.append(img_resized)
    return np.array(resized_images)

X_train_resized_resnet = resize_images(X_train, input_shape_resnet)

X_train_resized_densenet = resize_images(X_train, input_shape_densenet)

X_train_resized_efficientnet = resize_images(X_train, input_shape_efficientnet)

#培训与评估

'''
对于ResNet, DenseNet和EfficientNet，我们首先从Keras库中加载它们的预训练版本，并在ImageNet上训练权重。
我们删除了它们的顶级分类层，该层最初是为ImageNet的1000个类设计的，并添加了为我们的任务定制的分类头，该分类头有三个输出类(快乐、悲伤和愤怒)。
这个新的分类头被附加到基本模型的输出中，使用全局平均池化从图像中提取有意义的特征。
在添加自定义分类头后，我们使用分类交叉熵作为损失函数和Adam优化器来编译每个模型。
'''

#加载预训练的ResNet50模型，移除顶层分类层
resnet_base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape_resnet)
resnet_base_model.trainable = False

#向ResNet模型添加自定义分类头
resnet_global_avg_pooling = GlobalAveragePooling2D()(resnet_base_model.output)
resnet_output = Dense(6, activation='softmax')(resnet_global_avg_pooling)
resnet_model = Model(inputs=resnet_base_model.input, outputs=resnet_output)

#编译ResNet模型
resnet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

##加载预训练的DenseNet121模型，移除顶层分类层
densenet_base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape_densenet)
densenet_base_model.trainable = False

#向DenseNet模型添加自定义分类头
densenet_global_avg_pooling = GlobalAveragePooling2D()(densenet_base_model.output)
densenet_output = Dense(6, activation='softmax')(densenet_global_avg_pooling)
densenet_model = Model(inputs=densenet_base_model.input, outputs=densenet_output)

#编译DenseNet模型
densenet_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, min_lr=1e-7)

#训练模型
'''
将模型训练在经过调整大小的训练数据和验证集上，在监控验证损失的同时，使用早停和学习率调度回调函数，必要时停止训练。
'''

resnet_history = resnet_model.fit(X_train_resized_resnet, y_train, batch_size=32, epochs=200, validation_split=0.2,callbacks=[early_stopping, lr_scheduler])
densenet_history = densenet_model.fit(X_train_resized_densenet, y_train, batch_size=32, epochs=200, validation_split=0.2,callbacks=[early_stopping, lr_scheduler])

#将测试图像的大小调整为每个模型所需的输入形状
X_test_resized_densenet = resize_images(X_test, input_shape_densenet)
X_test_resized_resnet = resize_images(X_test, input_shape_resnet)

#根据测试数据对模型进行评估
densenet_loss, densenet_accuracy = densenet_model.evaluate(X_test_resized_densenet, y_test)
resnet_loss, resnet_accuracy = resnet_model.evaluate(X_test_resized_resnet, y_test)

print("\n")
print("DenseNet Test accuracy:", densenet_accuracy)
print("ResNet Test accuracy:", resnet_accuracy)

#学习和准确率曲线

def plot_learning_curves(history, model_name, ax):
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title(f'{model_name} Learning Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()


def plot_accuracy_curves(history, model_name, ax):
    ax.plot(history.history['accuracy'], label='Training Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_title(f'{model_name} Accuracy Curve')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()


fig, axs = plt.subplots(2, 2, figsize=(20, 15))
plot_learning_curves(densenet_history, 'DenseNet', axs[0, 0])
plot_accuracy_curves(densenet_history, 'DenseNet', axs[0, 1])

plot_learning_curves(resnet_history, 'ResNet', axs[1, 0])
plot_accuracy_curves(resnet_history, 'ResNet', axs[1, 1])


plt.tight_layout()
plt.show()

#加载并预处理测试图像
test_image = cv2.imread("D:\Python File\Jellyfish Image Machine Learn\Test_Photos\Test.jpg")
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image, (224, 224))
test_image = test_image.astype('float32') / 255.0
test_image = np.expand_dims(test_image, axis=-1)
test_image = np.repeat(test_image, 3, axis=-1)
test_image = np.expand_dims(test_image, axis=0)

#使用训练好的ResNet模型进行预测
predictions = densenet_model.predict(test_image)

#解码预测结果
class_labels = ['Moon Jellyfish', 'Barrel Jellyfish', 'Blue Jellyfish', 'Compass Jellyfish', 'Lion\'s Mane Jellyfish', 'Mauve Stinger Jellyfish']
predicted_class = np.argmax(predictions)
predicted_label = class_labels[predicted_class]

#打印预测结果
print("Predicted Jellyfish Species: ", predicted_label)