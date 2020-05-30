"""
Model mapping I to I
"""
import numpy as np
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, BatchNormalization, Input, Conv2DTranspose
from keras import optimizers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.callbacks import CSVLogger

data = np.load('ORL_faces.npz')

# load the "Train Images"
x_train = data['trainX']
#normalize every image
x_train = np.array(x_train,dtype='float32')/255

x_test = data['testX']
x_test = np.array(x_test,dtype='float32')/255

# load the Label of Images
y_train= data['trainY']
y_test= data['testY']

# show the train and test Data format
print('x_train : {}'.format(x_train[:]))
print('Y-train shape: {}'.format(y_train))
print('x_test shape: {}'.format(x_test.shape))

im_rows=112
im_cols=92
im_shape=(im_rows, im_cols, 1)

#change the size of images
x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(x_test.shape[0], *im_shape)
# x_valid = x_valid.reshape(x_valid.shape[0], *im_shape)
#
# Image configuration
#
final_image_shape = [112, 92, 1]
#
# Training parameters
#
epochs = 10
batch_size = 12

def conv2d_block(final_image_shape, filt3, kernel_size = 2, batch_norm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=filt3, kernel_size = (kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(final_image_shape)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

dropout = 0.1
batch_norm = True
shape1 = (2, 2)
shape2 = (2, 2)
shape3 = (2, 2)
shape4 = (2, 2)

filt1 = 32
filt2 = 1
filt3 = 1
#
# Contracting Path
#
inp_img = Input(shape=final_image_shape, name='input')
c1 = conv2d_block(inp_img, filt1, kernel_size=3, batch_norm = batch_norm)
p1 = MaxPooling2D(shape1)(c1)
p1 = Dropout(dropout)(p1)

c2 = conv2d_block(p1, filt2, kernel_size=3, batch_norm = batch_norm)
p2 = MaxPooling2D(shape2)(c2)
p2 = Dropout(dropout)(p2)

# c33 = conv2d_block(p2, filt3, kernel_size=3, batch_norm = batch_norm)
# p33 = MaxPooling2D(shape3)(c33)
# p33 = Dropout(dropout)(p33)
#
# Expansive Path
#
# c3 = conv2d_block(p33, filt3, kernel_size=3, batch_norm = batch_norm)
# c44 = UpSampling2D(shape3)(c3)
# c44 = Dropout(dropout)(c44)

c44 = p2
u4 = conv2d_block(c44, filt2, kernel_size=3, batch_norm = batch_norm)
c4 = UpSampling2D(shape2)(u4)
c4 = Dropout(dropout)(c4)

u5 = conv2d_block(c4, filt1, kernel_size=3, batch_norm = batch_norm)
c5 = UpSampling2D(shape1)(u5)
c5 = Dropout(dropout)(c5)

output1 = Conv2D(1, (1, 1), activation='sigmoid')(c5)
model1 = keras.Model(inputs=inp_img, outputs=output1)
sub_net_1 = keras.models.Model(inputs=inp_img, outputs=output1)
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
sub_net_1.compile(loss=['mse'], optimizer=sgd)

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=100)
mc = ModelCheckpoint('best_state_i_x_i.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
csv_logger = CSVLogger('training_i_x_i.log', separator=',', append=False)
history = sub_net_1.fit(x_train, x_train, batch_size=batch_size, epochs=epochs,
                            verbose=1, validation_split=0.1, callbacks=[es, mc, csv_logger])
#
# Save model
#
sub_net_1.save("model_i_x_i.h5")
print("Network saved to disk".format("model_i_x_i.h5"))

