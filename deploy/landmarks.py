from __future__ import print_function


import os
from enum import Enum
import numpy as np
import torch
import torch.backends.cudnn
from skimage import io
from skimage import color
from models import FAN, ResNetDepth
from utils import crop, flip, get_preds_fromhm, draw_gaussian


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect.

    ``point_2D`` - the detected points ``(x,y)`` are detected in a 2D space and
     follow the visible contour of the face
    ``point_half2D`` - points represent the projection of the 3D points into 2D
    ``point_3D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    point_2D = 1
    point_half2D = 2
    point_3D = 3


class NetworkSize(Enum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


models_urls = {
    '2DFAN-4': '2DFAN4-11f355bf06.pth.tar',
    '3DFAN-4': '3DFAN4-7835d9f11d.pth.tar',
    'depth': 'depth-2a464da4ea.pth.tar',
}


class FaceAlignment:
    def __init__(self, landmarks_type, network_size=NetworkSize.LARGE,
                 device='cuda', flip_input=False,
                 verbose=False, model_dir=None):
        self.device = device
        self.flip_input = flip_input
        self.landmarks_type = landmarks_type
        self.verbose = verbose

        network_size = int(network_size)

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Initialise the face alignemnt networks
        self.face_alignment_net = FAN(network_size)
        if landmarks_type == LandmarksType.point_2D:
            model_name = '2DFAN-' + str(network_size)
        else:
            model_name = '3DFAN-' + str(network_size)

        if not os.path.exists(model_dir):
            raise Exception('Landmarks model directory not found')
        filename = models_urls[model_name]
        model_file = os.path.join(model_dir, filename)
        if not os.path.isfile(model_file):
            raise Exception(
                'Landmarks model file not found: {}'.format(model_file)
            )
        fan_weights = torch.load(model_file)

        self.face_alignment_net.load_state_dict(fan_weights)

        self.face_alignment_net.to(device)
        self.face_alignment_net.eval()

        # Initialiase the depth prediciton network
        if landmarks_type == LandmarksType.point_3D:
            self.depth_prediciton_net = ResNetDepth()

            filename = models_urls['depth']
            model_file = os.path.join(model_dir, filename)
            if not os.path.exists(model_file):
                raise Exception('Landmarks depth model file not found')
            depth_weights = torch.load(model_file)

            depth_dict = {
                k.replace('module.', ''): v for k,
                v in depth_weights['state_dict'].items()}
            self.depth_prediciton_net.load_state_dict(depth_dict)

            self.depth_prediciton_net.to(device)
            self.depth_prediciton_net.eval()

    def get_landmarks(self, image_or_path, detected_face=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each
         image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} --
            The input image or path to it.

        Keyword Arguments:
            detected_faces {numpy.array} -- bounding box for founded face
            in the image (default: None)
        """
        if isinstance(image_or_path, str):
            image = io.imread(image_or_path)
        else:
            image = image_or_path

        if image.ndim == 2:
            image = color.gray2rgb(image)
        elif image.ndim == 4:
            image = image[..., :3]

        if detected_face is None:
            raise Exception('No faces were received.')

        torch.set_grad_enabled(False)
        x1, y1, x2, y2 = detected_face

        center = torch.FloatTensor([x2 - (x2 - x1) / 2.0, y2 - (y2 - y1) / 2.0])
        center[1] = center[1] - (y2 - y1) * 0.12
        scale = (x2 - x1 + y2 - y1) / 195

        inp = crop(image, center, scale)
        inp = torch.from_numpy(inp.transpose(
            (2, 0, 1))).float()

        inp = inp.to(self.device)
        inp.div_(255.0).unsqueeze_(0)

        out = self.face_alignment_net(inp)[-1].detach()
        if self.flip_input:
            out += flip(self.face_alignment_net(flip(inp))
                        [-1].detach(), is_label=True)
        out = out.cpu()

        pts, pts_img, score = get_preds_fromhm(out, center, scale)

        score = score.view(68)
        pts = pts.view(68, 2) * 4
        pts_img = pts_img.view(68, 2)

        if self.landmarks_type == LandmarksType.point_3D:
            heatmaps = np.zeros((68, 256, 256), dtype=np.float32)
            for j in range(68):
                if pts[j, 0] > 0:
                    heatmaps[j] = draw_gaussian(
                        heatmaps[j], pts[j], 2)
            heatmaps = torch.from_numpy(heatmaps).unsqueeze_(0)

            heatmaps = heatmaps.to(self.device)
            depth_pred = self.depth_prediciton_net(
                torch.cat((inp, heatmaps), 1)).data.cpu().view(68, 1)
            pts_img = torch.cat(
                (pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

        landmarks = pts_img.numpy()
        scores = score.numpy()

        return landmarks, scores
