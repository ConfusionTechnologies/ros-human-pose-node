from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Union

import numpy as np
import rclpy
from cv_bridge import CvBridge
from foxglove_msgs.msg import ImageMarkerArray
from geometry_msgs.msg import Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nicefaces.msg import BBox2Ds, ObjDet2Ds, WholeBody, WholeBodyArray
from nicepynode import Job, JobCfg
from nicepynode.utils import (
    RT_PUB_PROFILE,
    RT_SUB_PROFILE,
    append_array,
    convert_bboxes,
    declare_parameters_from_dataclass,
)
from onnxruntime import (
    ExecutionMode,
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)
from rclpy.node import Node
from ros2topic.api import get_msg_class
from sensor_msgs.msg import Image
from visualization_msgs.msg import ImageMarker

from onnx_wholebody_ros.processing import bbox_xyxy2cs, crop_bbox, heatmap2keypoints

NODE_NAME = "mmpose_model"

cv_bridge = CvBridge()


# Tuning Guide: https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/docs/ONNX_Runtime_Perf_Tuning.md
# and https://onnxruntime.ai/docs/performance/tune-performance.html
SESS_OPTS = SessionOptions()
# opts.enable_profiling = True
SESS_OPTS.enable_mem_pattern = True  # is default
SESS_OPTS.enable_mem_reuse = True  # is default
SESS_OPTS.execution_mode = ExecutionMode.ORT_PARALLEL  # does nothing on CUDA
SESS_OPTS.intra_op_num_threads = 2  # does nothing on CUDA
SESS_OPTS.inter_op_num_threads = 2  # does nothing on CUDA
SESS_OPTS.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL  # is default

# CUDAExecutionProvider Options: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
PROVIDER_OPTS = [
    dict(
        device_id=0,
        gpu_mem_limit=2 * 1024 ** 3,
        arena_extend_strategy="kSameAsRequested",
        do_copy_in_default_stream=False,
        cudnn_conv_use_max_workspace=True,
        cudnn_conv1d_pad_to_nc1d=True,
        cudnn_conv_algo_search="EXHAUSTIVE",
        # enable_cuda_graph=True,
    )
]


@dataclass
class WholeBodyCfg(JobCfg):
    model_path: str = "/models/vipnas_res50.onnx"
    """Local path of model."""
    frames_in_topic: str = "~/frames_in"
    """Video frames to predict on."""
    bbox_in_topic: str = "~/bbox_in"
    """Input topic for bboxes to crop."""
    preds_out_topic: str = "~/preds_out"
    """Output topic for predictions."""
    markers_out_topic: str = "~/dot_markers"
    """Output topic for visualization markers."""
    onnx_providers: list[str] = field(default_factory=lambda: ["CUDAExecutionProvider"])
    """ONNX runtime providers."""
    # TODO: img_wh should be embedded within exported model metadata
    img_wh: tuple[int, int] = (192, 256)
    """Input resolution."""
    score_threshold: float = 0.5
    """Minimum confidence level for filtering (ONLY VISUALIZATION MARKERS)"""
    crop_pad: float = 1.25
    """How much additional padding to crop around the bbox."""
    # idk are these imagenet's standardization values? anyways, vipnas_res50 was trained on these...
    # TODO: mean_rgb & std_rgb should be in model's metadata
    mean_rgb: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std_rgb: tuple[float, float, float] = (0.229, 0.224, 0.225)
    include_kps: list[Union[int, tuple[int]]] = field(
        default_factory=lambda: [
            (1, 19),
            20,
            21,
            23,
            66,
            69,
            63,
            60,
            78,
            72,
            111,
            132,
            99,
            120,
            95,
            116,
        ]
    )
    """list of tuples or int indexes. tuples are converted to slices to index. there are 133 keypoints in coco wholebody. filtering improves performance."""
    tracked_kps: list[Union[int, tuple[int]]] = field(
        default_factory=lambda: [1, 6, 7, 12, 13]
    )
    """list of tuples or int indexes. tuples are converted to slices to index. which keypoints to expose in TrackData for tracker nodes. filtering improves performance."""


# TODO:
# - move wholebody to mediapipe conversion to frontend
# - refactor out common parts with Yolo code


@dataclass
class WholeBodyPredictor(Job[WholeBodyCfg]):

    ini_cfg: WholeBodyCfg = field(default_factory=WholeBodyCfg)

    def attach_params(self, node: Node, cfg: WholeBodyCfg):
        super(WholeBodyPredictor, self).attach_params(node, cfg)

        declare_parameters_from_dataclass(
            node, cfg, exclude_keys=["onnx_providers", "img_wh", "mean_rgb", "std_rgb"]
        )

    def attach_behaviour(self, node: Node, cfg: WholeBodyCfg):
        super(WholeBodyPredictor, self).attach_behaviour(node, cfg)

        self._init_model(cfg)

        self.log.info(f"Waiting for publisher@{cfg.frames_in_topic}...")
        self._frames_sub = Subscriber(
            node,
            # blocks until image publisher is up!
            get_msg_class(node, cfg.frames_in_topic, blocking=True),
            cfg.frames_in_topic,
            qos_profile=RT_SUB_PROFILE,
        )

        self._bbox_sub = Subscriber(
            node, ObjDet2Ds, cfg.bbox_in_topic, qos_profile=RT_SUB_PROFILE
        )

        self._synch = ApproximateTimeSynchronizer(
            (self._frames_sub, self._bbox_sub),
            30,  # max 10 frame difference between pred & frame
            1 / 30,  # min 30 FPS waiting to sync
        )
        self._synch.registerCallback(self._on_input)

        self._pred_pub = node.create_publisher(
            WholeBodyArray, cfg.preds_out_topic, RT_PUB_PROFILE
        )
        self._marker_pub = node.create_publisher(
            ImageMarkerArray, cfg.markers_out_topic, RT_PUB_PROFILE
        )
        self.log.info("Ready")

    def detach_behaviour(self, node: Node):
        super().detach_behaviour(node)
        node.destroy_subscription(self._frames_sub.sub)
        node.destroy_subscription(self._bbox_sub.sub)
        node.destroy_publisher(self._pred_pub)
        node.destroy_publisher(self._marker_pub)
        # ONNX Runtime has no python API for destroying a Session
        # So I assume the GC will auto-handle it (based on its C API)

    def on_params_change(self, node: Node, changes: dict):
        self.log.info(f"Config changed: {changes}.")
        if not all(n in ("crop_pad",) for n in changes):
            self.log.info(f"Config change requires restart.")
            return True
        return False

    def _init_model(self, cfg: WholeBodyCfg):
        self.log.info("Initializing ONNX...")

        self.log.info(f"Model Path: {cfg.model_path}")
        self.session = InferenceSession(
            cfg.model_path,
            providers=cfg.onnx_providers,
            # performance gains measured to be negligable...
            sess_options=SESS_OPTS,
            provider_options=PROVIDER_OPTS,
        )
        # self.log.info(f"Options: {self.session.get_provider_options()}")
        # https://onnxruntime.ai/docs/api/python/api_summary.html#modelmetadata
        self.metadata = self.session.get_modelmeta()

        self._ratio_wh = cfg.img_wh[0] / cfg.img_wh[1]
        self._sess_out_name = self.session.get_outputs()[0].name
        self._sess_in_name = self.session.get_inputs()[0].name

        # mmpose doesn't attach metadata, we need to attach it ourselves
        # model_details = self.metadata.custom_metadata_map
        # self.log.info(f"Model Info: {self.metadata.custom_metadata_map}")

        # calculate keypoints to include
        if len(cfg.include_kps) > 0:
            self._include_key = np.r_[
                tuple(i if isinstance(i, int) else slice(*i) for i in cfg.include_kps)
            ]
        else:
            # includes all 133 keypoints
            self._include_key = np.r_[1:134]

        if len(cfg.tracked_kps) > 0:
            ids = np.r_[
                tuple(i if isinstance(i, int) else slice(*i) for i in cfg.tracked_kps)
            ]
            self._tracked_key = np.where(np.in1d(self._include_key, ids))[0]
        else:
            # includes all keypoints NOTE: LAGGY
            self._tracked_key = np.r_[0 : self._include_key.size]

        self.log.info(f"Included Keypoints: {self._include_key.size}")
        self.log.debug(f"{self._include_key}")
        self.log.info(f"Keypoints sent to Tracker: {self._tracked_key.size}")
        self.log.debug(f"{self._tracked_key}")

        self.log.info("ONNX initialized")

    def _forward(self, img: np.ndarray, dets: BBox2Ds):
        dets = convert_bboxes(
            dets, to_type=BBox2Ds.XYXY, normalize=False, img_wh=img.shape[1::-1]
        )
        boxes = np.stack((dets.a, dets.b, dets.c, dets.d)).transpose(1, 0)

        c, s = bbox_xyxy2cs(boxes, self._ratio_wh, self.cfg.crop_pad)
        crops = crop_bbox(img, c, s, self.cfg.img_wh)
        crops = (crops / 255 - self.cfg.mean_rgb) / self.cfg.std_rgb
        x = crops.transpose(0, 3, 1, 2).astype(np.float32)  # NHWC to NCHW

        # output shape is (N, 133, 64, 48), 133 is wholebody kepoints, (64, 48) is heatmap
        y = self.session.run([self._sess_out_name], {self._sess_in_name: x})[0]

        # filter by include_kps (increases performance)
        y = y[
            :, self._include_key - 1, ...
        ]  # convert id -> index (id start from 1, index from 0)

        coords, conf = heatmap2keypoints(y, c, s)
        # shape is (n, include_kps, 4), where 4 is (x, y, conf, id)
        poses = np.concatenate(
            (coords, conf, np.tile(self._include_key, (conf.shape[0], 1))[..., None]),
            axis=2,
        )
        return poses

    def _on_input(self, imgmsg: Image, detsmsg: ObjDet2Ds):
        if (
            self._pred_pub.get_subscription_count()
            + self._marker_pub.get_subscription_count()
            < 1
        ):
            return

        infer_start = self.get_timestamp()

        if isinstance(imgmsg, Image):
            img = cv_bridge.imgmsg_to_cv2(imgmsg, "rgb8")
        else:
            img = cv_bridge.compressed_imgmsg_to_cv2(imgmsg, "rgb8")
        if 0 in img.shape:
            self.log.debug("Image has invalid shape!")
            return

        if len(detsmsg.boxes.a) < 1:
            poses = np.zeros((0, self._include_key.size, 4), dtype=np.float32)
        else:
            # shape is (n, include_kps, 4), where 4 is (x, y, conf, id)
            poses = self._forward(img, detsmsg.boxes)  # is float64 for some reason

        infer_end = self.get_timestamp()

        # if the dets has already been tracked earlier
        has_tracks = any(t.id != -1 for t in detsmsg.tracks)
        # NOTE: filtering confidence here has no performance benefit here

        if self._pred_pub.get_subscription_count() > 0:
            norm_x = poses[:, self._tracked_key, 0] / img.shape[1]
            norm_y = poses[:, self._tracked_key, 1] / img.shape[0]

            arrmsg = WholeBodyArray(header=imgmsg.header)
            arrmsg.profiling.infer_start_time = infer_start
            arrmsg.profiling.infer_end_time = infer_end

            for i, pose in enumerate(poses):
                posemsg = WholeBody(header=imgmsg.header)
                append_array(posemsg.x, pose[:, 0])
                append_array(posemsg.y, pose[:, 1])
                append_array(posemsg.scores, pose[:, 2])
                append_array(posemsg.ids, pose[:, 3], dtype=np.int16)
                posemsg.is_norm = False

                if has_tracks:
                    posemsg.track = detsmsg.tracks[i]
                else:
                    posemsg.track.label = "person"
                    append_array(posemsg.track.x, norm_x[i])
                    append_array(posemsg.track.y, norm_y[i])
                    append_array(posemsg.track.scores, pose[self._tracked_key, 2])

                arrmsg.poses.append(posemsg)

            self._pred_pub.publish(arrmsg)

        if self._marker_pub.get_subscription_count() > 0:
            markersmsg = ImageMarkerArray()

            for pose in poses:
                marker = ImageMarker(header=imgmsg.header)
                marker.scale = 1.0
                marker.type = ImageMarker.POINTS
                marker.outline_color.r = 1.0
                marker.outline_color.a = 1.0
                for kp in pose[pose[:, 2] > self.cfg.score_threshold]:
                    marker.points.append(Point(x=kp[0], y=kp[1]))
                markersmsg.markers.append(marker)
            self._marker_pub.publish(markersmsg)


def main(args=None):
    if __name__ == "__main__" and args is None:
        args = sys.argv

    try:
        rclpy.init(args=args)

        node = Node(NODE_NAME)

        cfg = WholeBodyCfg()
        WholeBodyPredictor(node, cfg)

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
