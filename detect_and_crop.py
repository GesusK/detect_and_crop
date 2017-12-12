import os
import sys
import tensorflow as tf
import logging
import numpy as np
import PIL
import argparse

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


NUM_CLASSES = 1

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger('detector')
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_graph(model_dir):
    path_to_ckpt = os.path.join(model_dir, 'frozen_inference_graph.pb')

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def load_label_map(label_map_name, num_class):
    label_map = label_map_util.load_labelmap(label_map_name)
    categories = label_map_util.convert_label_map_to_categories(label_map,
                                max_num_classes=num_class, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def detect_object(detection_graph, sess, image, category_index, image_name, output_dir):
    with detection_graph.as_default():
        with sess.as_default() as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            #   image = Image.open(image_path)
              # the array based representation of the image will be used later in order to prepare the
              # result image with boxes and labels on it.
            # image_np = load_image_into_numpy_array(image)
            image_np = image
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

            crop_target(image_np, np.squeeze(boxes), np.squeeze(scores), image_name, output_dir, score_min=0.6)
            # Visualization of the results of a detection.
            image_np = np.array(image_np)
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=6,
              min_score_thresh = 0.6)
            return image_np


def crop_target(image_np, boxes, scores, image_name, output_dir, score_min=0.6, crop_max=2, y_expend=6, x_expend=4):
    if boxes.shape[0] == 0 or scores is None or scores[0] < score_min:
        logger.info('No target recognized in image {}'.format(image_name))

    for i in range(min(crop_max, boxes.shape[0])):
        if scores[i] < score_min:
            break
        box = tuple(boxes[i].tolist())
        score = scores[i]
        im_crop = image_np.crop((max(box[1] * image_np.width - x_expend, 0),
                                max(box[0] * image_np.height - y_expend, 0),
                                min(box[3] * image_np.width + x_expend, image_np.width),
                                min(box[2] * image_np.height + y_expend, image_np.height)))
        name_tmp = '{}-{}'.format(image_name, i+1)\
                   + '-%.2f.jpg' % score
        im_crop.save(os.path.join(output_dir, name_tmp))
        logger.debug('Saved cropped target image {}'
                     .format(os.path.join(output_dir, name_tmp)))


def image_list_gen(image_path):
    logger.info('Generate candidate images list...')
    res_list = list()
    for ind, im in enumerate(os.listdir(image_path)):
        im_path = os.path.join(image_path, im)
        #jpg = PIL.Image.open(im_path)
        res_list.append((ind, im[:-4], im_path))
        logger.debug('Put {} into image candidate list.'.format(im_path))

    return res_list


# a process to do the detection_graph

def object_detection_worker(images, detection_graph, category_index, result_path):
    logger.info('Start to draw boxes on images.')
    config = tf.ConfigProto()
    sess = tf.Session(graph=detection_graph, config=config)
    for ind, image in enumerate(images):
        with PIL.Image.open(image[2]) as im_data:
            res_image = detect_object(detection_graph, sess, im_data, category_index, image[1], result_path)
            PIL.Image.fromarray(res_image)\
                .save(os.path.join(result_path, '{1}_outfile_{0}.jpg'
                                   .format(image[0], image[1])))
            logger.debug('Successfully draw boxes on image: {}'.format(image[1]))


def parse():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model_dir', type=str, required=True,
                        help="The model directory with .pb extention to use.")
    arg_parser.add_argument('-i', '--image_dir', type=str, required=True,
                        help="The image directory where source images locate.")
    arg_parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help="The output directory for result images.")

    args = arg_parser.parse_args()
    return args


def main():
    
    args = parse()
    image_list = image_list_gen(args.image_dir)
    detection_graph = load_graph(args.model_dir)
    category_index = load_label_map(label_map_name=os.path.join(args.model_dir, 'jmlake-predef-classes.pbtxt'), num_class=NUM_CLASSES)

    object_detection_worker(image_list, detection_graph, category_index, args.output_dir)



if __name__ == "__main__":
    main()
