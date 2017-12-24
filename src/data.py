import tensorflow as tf

import dataset_util
import csv, os
import cv2

#flags = tf.app.flags
#flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
#FLAGS = flags.FLAGS


def create_tf_example(example):
  # TODO(user): Populate the following variables from your example.

  filename = example['image_id'] # Filename of the image. Empty if image is not from file
  filepath = '../data/img_celeba/' + filename
  print("Processing ", filename)

  image_raw = cv2.imread(filepath)
  height, width, _ = image_raw.shape  # Image height and width
  image_format = 'jpeg' # b'jpeg' or b'png'
  encoded_image_data = open(filepath).read()  # Encoded image bytes

  xmins = [float(example['x_1'])] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [float(example['x_1']) + float(example['width'])] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [float(example['y_1'])] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [float(example['y_1']) + float(example['height'])] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = ['face'] # List of string class name of bounding box (1 per box)
  classes = [1] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def main(_):
  path = "../data/"
  writerTrain = tf.python_io.TFRecordWriter(path + 'train.record')
  writerVal = tf.python_io.TFRecordWriter(path + 'val.record')
  writerTest = tf.python_io.TFRecordWriter(path + 'test.record')

  annoFile = open(os.path.join(path, 'list_bbox_celeba.txt'), 'r')
  partitionFile = open(os.path.join(path, 'list_eval_partition.txt'), 'r')

  fieldsAnno = ['image_id', 'x_1', 'y_1', 'width', 'height']
  fieldsPart = ['image_id', 'status']
  annoReader = csv.DictReader(annoFile, fieldnames=fieldsAnno,  delimiter=' ', skipinitialspace=True)
  partitionReader = csv.DictReader(partitionFile, fieldnames= fieldsPart, delimiter=' ')

  examplesTrain = []
  examplesVal = []
  examplesTest = []

  next(annoReader)
  next(annoReader)
  for row in annoReader:
      partDict = next(partitionReader)
      im_namePart, status = partDict['image_id'], float(partDict['status'])
      print(row)

      if status == 0:
          examplesTrain.append(row)
      elif  status == 1:
          examplesVal.append(row)
      else:
          examplesTest.append(row)

  print("num Train: ", len(examplesTrain), " num Val: "
        , len(examplesVal), " num Test: ", len(examplesTest))
  for example in examplesTrain:
    tf_example = create_tf_example(example)
    writerTrain.write(tf_example.SerializeToString())

  writerTrain.close()

  for example in examplesVal:
    tf_example = create_tf_example(example)
    writerVal.write(tf_example.SerializeToString())

  writerVal.close()

  for example in examplesTest:
    tf_example = create_tf_example(example)
    writerTest.write(tf_example.SerializeToString())

  writerTest.close()


if __name__ == '__main__':
  tf.app.run()