# unet_model.py
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import load_model

# Building and loading the model:


def unet(input_shape):
    inputs = Input(input_shape, name='input_layer')

    # Contracting path
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    p1 = MaxPooling2D((2, 2), name='pool1')(c1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(p1)
    p2 = MaxPooling2D((2, 2), name='pool2')(c2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(p2)
    p3 = MaxPooling2D((2, 2), name='pool3')(c3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(p3)
    p4 = MaxPooling2D((2, 2), name='pool4')(c4)

    # Bottleneck
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='conv5_1')(p4)

    # Expansive path
    u6 = UpSampling2D((2, 2), name='up6')(c5)
    u6 = Concatenate(name='concat6')([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv6_1')(u6)
    u7 = UpSampling2D((2, 2), name='up7')(c6)
    u7 = Concatenate(name='concat7')([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv7_1')(u7)
    u8 = UpSampling2D((2, 2), name='up8')(c7)
    u8 = Concatenate(name='concat8')([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv8_1')(u8)
    u9 = UpSampling2D((2, 2), name='up9')(c8)
    u9 = Concatenate(name='concat9')([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv9_1')(u9)

    # Blue mask output
    blue_mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='blue_mask_output')(c9)

    # Red mask output
    red_mask_output = Conv2D(1, (1, 1), activation='sigmoid', name='red_mask_output')(c9)

    return Model(inputs=[inputs], outputs=[blue_mask_output, red_mask_output])



def compile_and_load_model(model_path, input_shape=(1024, 1280, 3)):
    model = unet(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], run_eagerly=True)
    model = load_model(model_path)
    print('Model loaded successfully!')
    return model
