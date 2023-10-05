# unet_model.py
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

def unet(input_shape):
    # ... [your U-net function here]
    return Model(inputs=[inputs], outputs=[blue_mask_output, red_mask_output])
