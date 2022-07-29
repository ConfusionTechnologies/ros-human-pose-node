from __future__ import annotations
from dataclasses import dataclass, field
import sys
from copy import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from ros2topic.api import get_msg_class
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

import numpy as np
from onnxruntime import (
    InferenceSession,
    SessionOptions,
    ExecutionMode,
    GraphOptimizationLevel,
)

from sensor_msgs.msg import Image
from nicefaces.msg import (
    ObjDet2DArray,
    WholeBodyArray,
    BBox2D,
    WholeBody,
    BodyKeypoint,
    TrackData,
)
from foxglove_msgs.msg import ImageMarkerArray
from visualization_msgs.msg import ImageMarker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from nicepynode import Job, JobCfg
from nicepynode.utils import convert_bbox

from onnx_wholebody_ros.processing import bbox_xyxy2cs, crop_bbox, heatmap2keypoints

NODE_NAME = "mmpose_model"

cv_bridge = CvBridge()

# Realtime Profile: don't bog down publisher when model is slow
RT_PROFILE = copy(QoSPresetProfiles.SENSOR_DATA.value)
RT_PROFILE.depth = 0

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


# TODO:
# - move wholebody to mediapipe conversion to frontend
# - refactor out common parts with Yolo code


@dataclass
class WholeBodyPredictor(Job[WholeBodyCfg]):

    ini_cfg: WholeBodyCfg = field(default_factory=WholeBodyCfg)

    def attach_params(self, node: Node, cfg: WholeBodyCfg):
        super(WholeBodyPredictor, self).attach_params(node, cfg)

        node.declare_parameter("model_path", cfg.model_path)
        node.declare_parameter("frames_in_topic", cfg.frames_in_topic)
        node.declare_parameter("bbox_in_topic", cfg.bbox_in_topic)
        node.declare_parameter("preds_out_topic", cfg.preds_out_topic)
        node.declare_parameter("markers_out_topic", cfg.markers_out_topic)
        # onnx_providers is hardcoded
        # img_wh is hardcoded
        node.declare_parameter("score_threshold", cfg.score_threshold)
        node.declare_parameter("crop_pad", cfg.crop_pad)
        # mean_rgb is hardcoded
        # std_rgb is hardcoded

    def attach_behaviour(self, node: Node, cfg: WholeBodyCfg):
        super(WholeBodyPredictor, self).attach_behaviour(node, cfg)

        self._init_model(cfg)

        # Could use ros2topic to determine image type at runtime, GIVEN publisher
        # goes up first (it won't always). Trying to detect when publisher goes up
        # is too complex, so just make it a config option...
        self.log.info(f"Waiting for publisher@{cfg.frames_in_topic}...")
        self._frames_sub = Subscriber(
            node,
            # blocks until image publisher is up!
            get_msg_class(node, cfg.frames_in_topic, blocking=True),
            cfg.frames_in_topic,
            qos_profile=RT_PROFILE,
        )

        self._bbox_sub = Subscriber(
            node, ObjDet2DArray, cfg.bbox_in_topic, qos_profile=RT_PROFILE
        )

        self._synch = ApproximateTimeSynchronizer(
            (self._frames_sub, self._bbox_sub), 10, 0.06
        )
        self._synch.registerCallback(self._on_input)

        self._pred_pub = node.create_publisher(WholeBodyArray, cfg.preds_out_topic, 5)
        self._marker_pub = node.create_publisher(
            ImageMarkerArray, cfg.markers_out_topic, 5
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

    def step(self, delta: float):
        # Unused for this node
        return super().step(delta)

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

        self.log.info("ONNX initialized")

    def _on_input(self, imgmsg: Image, detsmsg: ObjDet2DArray):
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
        if len(detsmsg.dets) < 1:
            return

        boxes = np.array(
            [
                convert_bbox(
                    d.box, to_type=BBox2D.XYXY, normalize=False, img_wh=img.shape[1::-1]
                ).rect
                for d in detsmsg.dets
            ]
        )

        c, s = bbox_xyxy2cs(boxes, self._ratio_wh, self.cfg.crop_pad)
        crops = crop_bbox(img, c, s, self.cfg.img_wh)
        crops = (crops / 255 - self.cfg.mean_rgb) / self.cfg.std_rgb
        x = crops.transpose(0, 3, 1, 2).astype(np.float32)  # NHWC to NCHW

        # output shape is (N, 133, 64, 48), 133 is wholebody kepoints, (64, 48) is heatmap
        y = self.session.run([self._sess_out_name], {self._sess_in_name: x})[0]

        coords, conf = heatmap2keypoints(y, c, s)
        # coco wholebody ids
        ids = (np.arange(conf.size) + 1).reshape(conf.shape)
        # shape is (n, 133, 4), where 4 is (x, y, conf, id)
        poses = np.concatenate((coords, conf, ids), axis=2)

        infer_end = self.get_timestamp()

        # NOTE: since filtering confidence here has no performance benefit here
        # it should be done client side

        if self._pred_pub.get_subscription_count() > 0:
            self._pred_pub.publish(
                WholeBodyArray(
                    header=imgmsg.header,
                    poses=[
                        WholeBody(
                            header=imgmsg.header,
                            # extra point #0 since WholeBody counts from 1
                            pose=[
                                BodyKeypoint(x=x, y=y, score=conf, id=int(id))
                                for (x, y, conf, id) in pose
                            ],
                            is_norm=False,
                            track=TrackData(
                                label="person",
                                keypoints=[
                                    Point(
                                        x=kp[0] / img.shape[1], y=kp[1] / img.shape[0]
                                    )
                                    for kp in pose
                                ],
                                keypoint_scores=[kp[2] for kp in pose],
                            ),
                        )
                        for pose in poses
                    ],
                    infer_start_time=infer_start,
                    infer_end_time=infer_end,
                )
            )

        if self._marker_pub.get_subscription_count() > 0:
            self._marker_pub.publish(
                ImageMarkerArray(
                    markers=[
                        ImageMarker(
                            header=imgmsg.header,
                            scale=1.0,
                            type=ImageMarker.POINTS,
                            outline_color=ColorRGBA(r=1.0, a=1.0),
                            points=[
                                Point(x=kp[0], y=kp[1])
                                for kp in pose
                                if kp[2] > self.cfg.score_threshold
                            ],
                        )
                        for pose in poses
                    ]
                )
            )


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
