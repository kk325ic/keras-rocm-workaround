# a working around max pooling 3d, since rocm's max pooling3d not working yet
from keras.utils import conv_utils
class MaxPooling3D(Layer):
    def __init__(self, pool_size=(2,2,2), strides=None, padding='valid', 
            data_format=None, **kwargs):
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        
        self.strides = strides
        self.padding = padding
        self.data_format = K.normalize_data_format(data_format)
        
        super(MaxPooling3D, self).__init__(**kwargs)

    def build(self, input_shape):
        # input shape
        if self.data_format == 'channels_first':
            chn_dim = input_shape[1]
            len_dim1 = input_shape[2]
            len_dim2 = input_shape[3]
            len_dim3 = input_shape[4]
        elif self.data_format == 'channels_last':
            len_dim1 = input_shape[1]
            len_dim2 = input_shape[2]
            len_dim3 = input_shape[3]
            chn_dim = input_shape[4]

        odim = (len_dim1, len_dim2, len_dim3, chn_dim)

        # computer the out shape
        len_dim1 = conv_utils.conv_output_length(len_dim1, self.pool_size[0],
                self.padding, self.strides[0])
        len_dim2 = conv_utils.conv_output_length(len_dim2, self.pool_size[1],
                self.padding, self.strides[1])
        len_dim3 = conv_utils.conv_output_length(len_dim3, self.pool_size[2],
                self.padding, self.strides[2])

        if self.data_format == 'channels_first':
            self.output_dim = (input_shape[0], input_shape[1], len_dim1, len_dim2, len_dim3)

            shape_2d_0 = (chn_dim, odim[0]*odim[1], odim[2])
            shape_1d_1 = (chn_dim, odim[0]*len_dim2*len_dim3)
            shape_f = (chn_dim, len_dim1, len_dim2, len_dim3)

        elif self.data_format == 'channels_last':
            self.output_dim = (input_shape[0], len_dim1, len_dim2, len_dim3, input_shape[4])
            shape_2d_0 = (odim[0]*odim[1], odim[2], chn_dim)
            shape_2d_1 = (odim[0], len_dim2*len_dim3, chn_dim)
            shape_f = (len_dim1, len_dim2, len_dim3, chn_dim)


        # a list of layers needed for 3D MaxPooling
        (p1, p2, p3) = self.pool_size
        
        self.layers = []
        self.layers.append(Reshape(shape_2d_0))
        self.layers.append(MaxPooling2D((p2,p3), padding=self.padding))
    
        self.layers.append(Reshape(shape_2d_1))
        self.layers.append(MaxPooling2D((p1, 1), padding=self.padding))

        self.layers.append(Reshape(shape_f))
 
        super(MaxPooling3D, self).build(input_shape)

    def call(self, x):
        for l in self.layers:
            x = l(x)
        return x 

    def compute_output_shape(self, input_shape):
        return self.output_dim

    def get_config(self):
        config = {
                'pool_size': self.pool_size,
                'padding': self.padding,
                'strides': self.strides,
                'data_format': self.data_format
                }
        base_config = super(MaxPooling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

