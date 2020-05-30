"""
Model mapping I to I and simultaneously getting G from the latent representation(X)
"""
import numpy as np
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, BatchNormalization, Input, Conv2DTranspose,\
    concatenate
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from sklearn.externals import joblib
from keras.callbacks import CSVLogger
import h5py
from keras.models import load_model
from sklearn.decomposition import PCA

def conv2d_block(final_image_shape, filt3, kernel_size = 2, batch_norm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=filt3, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal',
               padding='same')(final_image_shape)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

data = np.load('ORL_faces.npz')

# load the "````````````1```````````` Images"
x_train = data['trainX']
#normalize every image
x_train = np.array(x_train,dtype='float32')/255

x_test = data['testX']
x_test = np.array(x_test,dtype='float32')/255

# load the Label of Images
y_train = data['trainY']
y_test = data['testY']

x_data = np.append(x_train, x_test, axis = 0)
pca = PCA(n_components=min(x_data.shape[0], x_data.shape[1]))
model = pca.fit(x_data)
x_transformed = model.transform(x_data)

x_transformed1 = x_transformed[:, :6]

# show the train and test Data format
# print('x_train : {}'.format(x_train[:]))
# print('Y-train shape: {}'.format(y_train))
# print('x_test shape: {}'.format(x_test.shape))

# x_train, x_valid, y_train, y_valid= train_test_split(x_train, y_train, test_size=.3, random_state=0)

im_rows=112
im_cols=92
batch_size=512
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
epochs = 100
batch_size = 12

neuralnetwork = load_model('best_state_i_x_i.h5')

morph_train1 = x_transformed1[:len(x_train), :]
morph_test1 = x_transformed1[len(x_train):len(x_train) + len(x_test), :]
# x_train1, x_test1, morph_train1, morph_test1 = train_test_split(all_data, morph_desc, test_size=0.3, random_state=100)

encoder = keras.models.Model(inputs=neuralnetwork.get_layer('input').input,
                             outputs=neuralnetwork.get_layer('dropout_2').output)
enc_out = encoder.predict(x_train)
dropout = .1

inp_enc = Input(shape=(enc_out.shape[1], enc_out.shape[2], enc_out.shape[3]))
c1 = conv2d_block(inp_enc, 2, kernel_size=3, batch_norm=True)
p1 = MaxPooling2D((2, 2))(c1)
p1 = Dropout(dropout)(p1)

c2 = conv2d_block(p1, 64, kernel_size=3, batch_norm=True)
p2 = MaxPooling2D((2, 2))(c2)
p2 = Dropout(dropout)(p2)

image_vector_size = 64*4*4
input_vec = Flatten()(p1)
a1 = (Dense(64, activation='relu', input_shape=(image_vector_size, ),
                 kernel_initializer='normal'))(input_vec)

a2 = (Dense(32, activation='relu', kernel_initializer='normal'))(a1)
a3 = (Dense(16, activation='relu', kernel_initializer='normal'))(a2)
hidden_yl5 = (Dense(6, activation='relu', kernel_initializer='normal', name='low_fid'))(a3)

sub_net_1 = keras.models.Model(inputs=inp_enc, outputs=hidden_yl5)
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
sub_net_1.compile(loss='mse', optimizer=sgd)

es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=150)
mc = ModelCheckpoint('best_state_x_gg.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
csv_logger = CSVLogger('training_x_gg.log', separator=',', append=False)
history = sub_net_1.fit(enc_out, morph_train1, batch_size=batch_size, epochs=epochs,
                            verbose=1, validation_split=0.3, callbacks=[es, mc, csv_logger])
#
# Save model
#
sub_net_1.save("model_x_gg.h5")
print("Network saved to disk".format("model_x_gg.h5"))

