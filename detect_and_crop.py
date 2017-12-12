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
PROBABILITY_THRESHOLD = 0.6

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


def detect_object(detection_graph, sess, image, category_index, image_name, output_dir, draw_result=False):
    with detection_graph.as_default():
        with sess.as_default() as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            image_np = image
            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
              [detection_boxes, detection_scores, detection_classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})

            crop_target(image_np,
                        np.squeeze(boxes),
                        np.squeeze(scores),
                        image_name, output_dir,
                        score_min=PROBABILITY_THRESHOLD)

            if draw_result:
                image_np = np.array(image_np)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=6,
                    min_score_thresh=PROBABILITY_THRESHOLD)

                save_path = os.path.join(output_dir, '{}_demo_output.jpg'.format(image_name))
                PIL.Image.fromarray(image_np).save(save_path)
                logger.debug('Successfully draw boxes on image: {}_demo_output.jpg'.format(image_name))


def crop_target(image_np, boxes, scores, image_name, output_dir, score_min=0.6, crop_max=2, y_expend=6, x_expend=4):
    if boxes.shape[0] == 0 or scores is None or scores[0] < score_min:
        logger.info('No target recognized in image {}.jpg'.format(image_name))

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
        im = im.lower()
        if '.jpg' not in im:
            continue
        im_path = os.path.join(image_path, im)
        res_list.append((ind, im[:-4], im_path))
        logger.debug('Put {} into image candidate list.'.format(im_path))

    return res_list


# a process to do the detection_graph

def object_detection_worker(images, detection_graph, category_index, result_path, if_demo=False):
    logger.info('Start to draw boxes on images.')
    config = tf.ConfigProto()
    sess = tf.Session(graph=detection_graph, config=config)
    for ind, image in enumerate(images):
        with PIL.Image.open(image[2]) as im_data:
            detect_object(detection_graph, sess, im_data, category_index,
                          image[1], result_path, draw_result=if_demo)



def parse():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model_dir', type=str, required=True,
                            help="The model directory with .pb extention to use.")
    arg_parser.add_argument('-i', '--image_dir', type=str, required=True,
                            help="The image directory where source images locate.")
    arg_parser.add_argument('-o', '--output_dir', type=str, required=True,
                            help="The output directory for result images.")
    arg_parser.add_argument('-d', '--demo_mode', action='store_true',
                            help="Enable the demo mode, to draw boxes on targets in images.", required=False)

    args = arg_parser.parse_args()
    return args


def main():
    args = parse()
    image_list = image_list_gen(args.image_dir)
    detection_graph = load_graph(args.model_dir)
    category_index = load_label_map(label_map_name=os.path.join(args.model_dir, 'jmlake-predef-classes.pbtxt'), num_class=NUM_CLASSES)
    if args.demo_mode:
        demo = True
    else:
        demo = False
    object_detection_worker(image_list, detection_graph, category_index, args.output_dir, if_demo=demo)


if __name__ == "__main__":
    main()
