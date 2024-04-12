# adapted from https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406

def encode_block(inputs, n_filters):
    conv = Conv2D(n_filters, kernel_size=(3,3), activation='relu', padding='same')(inputs)
    conv = Conv2D(n_filters, kernel_size=(3,3), activation='relu', padding='same')(conv)
    out = MaxPooling2D(strides=(2,2))(conv)
    skip = conv
    return out, skip

def decode_block(inputs, skip, n_filters):
    upconv = Conv2DTranspose(n_filters, kernel_size=(3,3), strides=(2,2), padding='same')(inputs)
    concat = Concatenate()([upconv, skip])
    conv = Conv2D(n_filters, kernel_size=(3,3), activation='relu', padding='same')(concat)
    out = Conv2D(n_filters, kernel_size=(3,3), activation='relu', padding='same')(conv)
    return out

def bottleneck(inputs, n_filters):
    conv = Conv2D(n_filters, kernel_size=(3,3), activation='relu', padding='same')(inputs)
    conv = Conv2D(n_filters, kernel_size=(3,3), activation='relu', padding='same')(conv)
    return conv

def create_unet(input_shape, n_initial_filters=64, n_blocks=4):
    inputs = Input(input_shape)
    filter_sizes = [n_initial_filters * np.power(2, i) for i in range(n_blocks)]
    reverse_filter_sizes = filter_sizes[::-1]

    block, skip = encode_block(inputs, n_initial_filters)
    encode_blocks = [block]
    skip_connections = [skip]
    for filter_size in filter_sizes[1:]:
        block, skip = encode_block(encode_blocks[i-1], filter_size)
        encode_blocks.append(block)
        skip_connections.append(skip)

    latent_dim = bottleneck(encode_blocks[-1], filter_sizes[-1])
    
    decode_blocks = [decode_block(latent_dim, skip_connections[-1], filter_sizes[-2])]
    for filter_size, i in enumerate(reverse_filter_sizes[2:]):
        decode_blocks.append(decode_block(decode_blocks[i-1], skip_connections[n_blocks-i], filter_size))

    out = Conv2D(3, kernel_size=(1,1), padding='same')(decode_blocks[-1])

    model = tf.keras.Model(inputs=inputs, outputs=out)
    return model        

unet = create_unet(x_train[0].shape)
unet.summary()