import tensorflow   as tf
import numpy        as np

def base_name(var):
    ''' Extracts value passed to name when creating a variable; credit to tf_rl '''
    return var.name.split('/')[-1].split(':')[0]

def create_variables(shape, name=''): 
    ''' Creates both weight and bias Variables '''
    # Create a Variable from the normal distribution truncated at 0.1
    W_init = tf.truncated_normal_initializer()
    b_init = tf.constant_initializer(0.1)

    W = tf.get_variable(name+'_W', shape,        initializer=W_init)
    b = tf.get_variable(name+'_b', [shape[-1],], initializer=b_init)

    tf.histogram_summary(name + '_Weights', W)
    tf.histogram_summary(name + '_Bias', b) 
    return W, b


class ConvLayer(object):
    def __init__(self, layer_number, previous_layer, shape, stride, scope=None):
        ''' Creates a convolutional layer '''

        self.layer_number = layer_number
        self.shape        = shape 
        self.stride       = stride
        self.scope        = scope or 'ConvLayer_%i' % layer_number

        with tf.variable_scope(self.scope) as sc:
            W, b = create_variables(shape, 'Conv')#, self.scope if type(self.scope) is str else self.scope.name)#sc.name)
            conv = tf.nn.conv2d(input   = previous_layer, 
                                filter  = W, 
                                strides = [1, stride, stride, 1], 
                                padding = 'SAME',
                                name    = 'Convolution_%i' % layer_number)

            relu = tf.nn.relu(conv + b, name='ReLU_%i' % layer_number)
            pool = tf.nn.max_pool(value   = relu, 
                                  ksize   = [1, 2, 2, 1],
                                  strides = [1, 2, 2, 1], 
                                  padding = 'SAME',
                                  name    = '2x2_Pool')
        self.layer = pool
        self.vars  = [b, W]


    def copy(self, previous_layer, scope=None):
        scope = scope or self.scope + '_copy'

        with tf.variable_scope(scope) as sc:
            for v in self.vars:
                tf.get_variable(base_name(v), v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: v.initialized_value())
            sc.reuse_variables()
            return ConvLayer(self.layer_number, previous_layer, 
                             self.shape, self.stride, scope=sc)



class FullLayer(object):
    def __init__(self, layer_number, previous_layer, shape, use_relu=False, scope=None):
        ''' Creates a fully connected layer '''

        self.layer_number = layer_number
        self.shape        = shape 
        self.use_relu     = use_relu
        self.scope        = scope or 'FullLayer_%i' % layer_number

        with tf.variable_scope(self.scope) as sc:
            # Previous layer must be flat
            if previous_layer.get_shape().as_list()[-1] != 256:
                previous_layer = tf.reshape(previous_layer, [-1, 256])
            W, b = create_variables(shape, 'Full')#, sc.name)
            full = tf.nn.xw_plus_b(previous_layer, W, b, name='Wx_b_%i' % layer_number)   

            if use_relu:
                full = tf.nn.relu(full, name='FullReLU_%i' % layer_number)
            
        self.layer = full
        self.vars  = [b, W]


    def copy(self, previous_layer, scope=None):
        scope = scope or self.scope + '_copy'

        with tf.variable_scope(scope) as sc:
            for v in self.vars:
                tf.get_variable(base_name(v), v.get_shape(),
                        initializer=lambda x,dtype=tf.float32: v.initialized_value())
            sc.reuse_variables()
            return FullLayer(self.layer_number, previous_layer, 
                             self.shape, self.use_relu, scope=sc)



class CNN(object):

    initialized = False
    
    def __init__(self, output_size, scope=None, given_layers=None, input_placeholder=None):
        self.input_placeholder = input_placeholder
        self.output_size       = output_size
        self.scope             = scope or 'CNN'

        if input_placeholder is not None:
            self.initialize(given_layers)

    def __call__(self, xs):
        with tf.variable_scope(self.scope):

            if type(xs) is not np.ndarray:
                if not self.initialized:
                    self.input_placeholder = xs
                    self.initialize()
                return self.layers[-1].layer
            return self.layers[-1].layer.eval({self.input_placeholder : xs})


    def initialize(self, given_layers=None):
        with tf.variable_scope(self.scope):
            if given_layers is not None:
                self.layers      = given_layers
            else:
                depth       = self.input_placeholder.get_shape().as_list()[-1]
                self.layers = []

                # Convolutional Layers
                self.layers.append( ConvLayer(1, self.input_placeholder, [8,8,depth,32], 4) )
                self.layers.append( ConvLayer(2, self.layers[-1].layer,  [4, 4, 32, 64], 2) )
                self.layers.append( ConvLayer(3, self.layers[-1].layer,  [3, 3, 64, 64], 1) )

                # Fully Connected Layers
                self.layers.append( FullLayer(1, self.layers[-1].layer, [256, 256], True) )
                self.layers.append( FullLayer(2, self.layers[-1].layer, [256, self.output_size]) )
        self.initialized = True
   

    def variables(self):
        return [v for layer in self.layers for v in layer.vars]


    def copy(self, scope=None):
        scope = scope or self.scope + '_copy'

        if self.initialized:
            previous_layer = self.input_placeholder
            given_layers   = []
            for layer in self.layers:
                given_layers.append( layer.copy(previous_layer) )
                previous_layer = given_layers[-1].layer

            return CNN(self.output_size, scope, given_layers, self.input_placeholder)
        return CNN(self.output_size, scope)

