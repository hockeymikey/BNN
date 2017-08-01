from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

# Imports
from data_prep.data_utils import *
from data_prep.custom_classes import v_point
from data_prep.moz_utils import points_2_direction
from data_prep.lbl_utils import get_sorted_veg_from_excel

tf.logging.set_verbosity(tf.logging.INFO)


def get_dateset():
  cols = get_sorted_veg_from_excel()
  dn = '../Extraction/imgs_veg'
  dir = os.listdir(dn)
  # Check type on the array if it should be float or what
  d = np.zeros(shape=(0, 46, 46, 4), dtype=np.float32)
  # dir.__len__()
  l = np.array([], dtype=np.int32)
  l_s = {}
  l_c = {}
  i = 0
  
  for file in dir:
    if file.endswith('.tif'):
      ti = tiff.TiffFile(dn + '/' + file)
      t2 = ti.asarray()
      try:
        # d[:,i] = np.reshape(ti.asarray(), -1, order='F')
        # d = np.vstack((d, np.reshape(ti.asarray(), -1, order='F')))
        
        d = np.concatenate((d, [t2]), axis=0)
        pp = file_name_to_class(file.split('.')[0], cols)
        
        l_s.setdefault(pp, 0)
        l_s[pp] += 1
        l_c.setdefault(pp, l_c.__len__())
        l = np.append(l, l_c[pp])
      except Exception as exx:
        print(exx)
      
      i += 1
  return (d, l, l_s, l_c)


def file_name_to_class(name, cols):
  # WHI11_WHJ21-1
  
  name = name.split('_')
  tmp = name[1].split('-')
  name = [name[0], tmp[0], tmp[1]]
  
  a = v_point(name[0])
  b = v_point(name[1])
  
  left, right, dir = points_2_direction(a, b)
  try:
    dd = cols[left.zone][left.col][left.row][dir][1]
    dd = dd[int(name[2]) + 1]
    return dd
  except Exception as ex:
    print(str(ex) + '---> ' + name[2])
    return 'B'

def main(unused_argv):
  # Load training and eval data
  #mnist = learn.datasets.load_dataset("mnist")
  #train_data = mnist.train.images # Returns np.array
  #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  
  train_data, train_labels,unique, l_c = get_dateset()
  

  n = 2
  i = 1
  while (n * 2) < unique.__len__():
    n *= 2
    i += 1

  # tmp = np.eye(6)[labels]
  #hot = tf.one_hot(train_labels, i, 1.0, 0.0)
  
  #eval_data,eval_labels = get_veg_data()
  eval_data = train_data
  eval_labels = train_labels
  
  #eval_data = mnist.test.images # Returns np.array
  #eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  
  # Create the Estimator
  mnist_classifier = learn.Estimator(
    model_fn=cnn_model_fn, model_dir="mnist_convnet_model")
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  mnist_classifier.fit(
    x=train_data,
    y=train_labels,
    batch_size=50,
    steps=20000,
    monitors=[logging_hook])

  # Configure the accuracy metric for evaluation
  metrics = {
    "accuracy":
      learn.MetricSpec(
        metric_fn=tf.metrics.accuracy, prediction_key="classes"),
  }

  # Evaluate the model and print results
  eval_results = mnist_classifier.evaluate(
    x=eval_data, y=eval_labels, metrics=metrics)
  print(eval_results)


# Our application logic will be added here
def cnn_model_fn(features, labels, mode):
  image_width = 46
  image_height = 46
  
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features, [-1, image_width, image_height, 4])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=20,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=30,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
          input=logits, axis=1),
      "probabilities": tf.nn.softmax(
          logits, name="softmax_tensor")
  }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)

if __name__ == "__main__":
  tf.app.run()
