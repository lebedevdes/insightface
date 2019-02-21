from __future__ import division
from __future__ import print_function


import sys
import numpy as np
import cv2
import mxnet as mx
from sklearn import preprocessing
from skimage import transform as trans
from landmarks import FaceAlignment, LandmarksType


class Landmarks:
    def __init__(self):
        self.right_eye = None
        self.left_eye = None
        self.nose = None
        self.right_month = None
        self.left_month = None
        self.chin = None


class FaceRecognition:
    def __init__(self):
        self.landmarks_threshold = 0.5

        # Initialize face alignment network
        print("Initialize FAN model")
        try:
            fan = FaceAlignment(LandmarksType.point_2D,
                                flip_input=False,
                                device='cuda',
                                model_dir="../models/face_landmarks")
            self.fan = fan
        except Exception as err:
            print("Failed initialize FAN model: {}".format(err))
            sys.exit(1)

        # Initialize arcnet model
        print("Initialize FR model")
        try:
            ctx = mx.gpu(0)
            image_size = (112, 112)
            prefix = "../models/model-r100s"
            epoch = 0
            sym, arg_params, aux_params = mx.model.load_checkpoint(prefix,
                                                                   epoch)
            all_layers = sym.get_internals()
            sym = all_layers['fc1_output']
            model = mx.mod.Module(symbol=sym, context=ctx, label_names=[])
            model.bind(
                data_shapes=[('data', (1, 3, image_size[0], image_size[1]))]
            )
            model.set_params(arg_params, aux_params)
            self.fr_model = model
        except Exception as err:
            print("Failed initialize FR model: ".format(err))
            sys.exit(1)

    @staticmethod
    def preprocess(img, landmark=None, **kwargs):
        m = None
        image_size = []
        str_image_size = kwargs.get('image_size', '')
        if len(str_image_size) > 0:
            image_size = [int(x) for x in str_image_size.split(',')]
            if len(image_size) == 1:
                image_size = [image_size[0], image_size[0]]
            assert len(image_size) == 2
            assert image_size[0] == 112
            assert image_size[0] == 112 or image_size[1] == 96
        if landmark is not None:
            assert len(image_size) == 2
            src = np.array([
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041]], dtype=np.float32)
            if image_size[1] == 112:
                src[:, 0] += 8.0
            dst = landmark.astype(np.float32)

            tform = trans.SimilarityTransform()
            tform.estimate(dst, src)
            m = tform.params[0:2, :]

        assert len(image_size) == 2
        warped = cv2.warpAffine(img, m, (image_size[1], image_size[0]),
                                borderValue=0.0)
        return warped

    @staticmethod
    def get_center_line_point(p1, p2):
        x = p1[0] + (p2[0] - p1[0]) / 2
        y = p1[1] + (p2[1] - p1[1]) / 2
        return [x, y]

    def get_landmarks(self, image):
        # Unpading image
        h, w, _ = image.shape
        dh = h - h / 1.5
        dw = w - w / 1.5
        x1 = int(dw / 2)
        y1 = int(dh / 2)
        x2 = w - int(dw / 2)
        y2 = h - int(dh / 2)
        detection = np.array([x1, y1, x2, y2])
        try:
            points, scores = self.fan.get_landmarks(image, detection)
        except IOError as err:
            print ("Can't open image file: {}".format(err))
            sys.exit(1)
        except Exception as err:
            print("Failed get landmarks: ".format(err))
            sys.exit(1)

        filtred_scores = [scores[36], scores[39], scores[42], scores[45],
                          scores[30], scores[48], scores[54], scores[8]]
        cond = all(l > self.landmarks_threshold for l in filtred_scores)

        if not cond:
            print(
                "Skip bad landmarks, mean score = {}".format(
                    np.mean(filtred_scores)
                )
            )
            return None
        landmarks = Landmarks()
        landmarks.right_eye = self.get_center_line_point(points[36],
                                                         points[39])
        landmarks.left_eye = self.get_center_line_point(points[42],
                                                        points[45])
        landmarks.nose = points[30]
        landmarks.right_month = points[48]
        landmarks.left_month = points[54]
        landmarks.chin = points[8]
        return landmarks

    def get_face_features(self, image_path):
        try:
            img = cv2.imread(image_path)
            la = self.get_landmarks(img)
            landmarks = np.array([la.right_eye, la.left_eye,
                                  la.nose,
                                  la.right_month, la.left_month])
            alignmented = self.preprocess(img, landmark=landmarks,
                                          image_size='112,112')
            rimg = cv2.cvtColor(alignmented, cv2.COLOR_BGR2RGB)
            pimg = np.transpose(rimg, (2, 0, 1)).astype(np.float32)
            pimg -= 127.5
            pimg *= 0.0078125
            input_blob = np.expand_dims(pimg, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            self.fr_model.forward(db, is_train=False)
            embedding = self.fr_model.get_outputs()[0].asnumpy()
            embedding = preprocessing.normalize(embedding).flatten()
            return embedding
        except Exception as err:
            print("Failed to run image: {}".format(err))
            sys.exit(1)


if __name__ == '__main__':
    img1 = 'camera-001_18_M_0-20_1544405111.614.jpg'
    img2 = 'camera-001_20_M_0-20_1544405366.065.jpg'

    fr = FaceRecognition()
    f1 = fr.get_face_features(img1)
    f2 = fr.get_face_features(img2)

    print(f1)
    print(f2)

    dist = np.sum(np.square(f1 - f2))

    print(dist)




