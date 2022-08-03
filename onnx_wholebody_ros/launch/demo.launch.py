from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

# By right, Nicepipe main launch file responsible for orchestration
# This demo file is just for convenience, hence the undeclared dependencies

# TODO:
# Use https://github.com/ros-tooling/topic_tools to convert ObjDet2DArray
# to BBox2DArray for wholebody node


def generate_launch_description():
    wholebody_node = Node(
        package="onnx_wholebody_ros",
        namespace="/models",
        executable="wholebody",
        name="wholebody",
        # remappings=[
        #     ("~/frames_in", "/rtc/rtc_receiver/frames_out"),
        #     ("~/bbox_in", "/models/yolov5/preds_out"),
        #     ("~/preds_out", "/data_out"),
        # ],
        parameters=[
            {
                "frames_in_topic": "/rtc/rtc_receiver/frames_out",
                "bbox_in_topic": "/models/yolov5/preds_out",
                "preds_out_topic": "/data_out",
            }
        ],
        respawn=True,
    )

    yolov5_node = Node(
        package="onnx_yolov5_ros",
        namespace="/models",
        executable="yolov5",
        name="yolov5",
        # equivalent to --remap yolov5/frames_in:=/rtc/rtc_receiver/frames_out
        # remappings=[("~/frames_in", "/rtc/rtc_receiver/frames_out")],
        parameters=[{"frames_in_topic": "/rtc/rtc_receiver/frames_out"}],
        respawn=True,
    )

    aiortc_cfg = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            str(Path(get_package_share_directory("aiortc_ros")) / "main.launch.py")
        ),
        launch_arguments=[("namespace", "/rtc")],
    )

    return LaunchDescription([wholebody_node, yolov5_node, aiortc_cfg])

