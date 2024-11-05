"""COLMAP visualizer

Visualize COLMAP sparse reconstruction outputs. To get demo data, see `./assets/download_colmap_garden.sh`.
"""
import json
import random
import time
import typing as t
from itertools import chain
from pathlib import Path
from typing import List

import imageio.v3 as iio
import numpy as np
import open3d as o3d
import tyro
from jaxtyping import Float
from loguru import logger
from tqdm.auto import tqdm

import viser
import viser.transforms as tf
from nerfstudio.cameras.camera_utils import quaternion_matrix, quaternion_from_matrix
from viser.extras.colmap._colmap_utils import Camera, Image

_opencv2slam: Float[np.ndarray, "4 4"] = np.array(
    [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
)
_slam2opencv: Float[np.ndarray, "4 4"] = np.linalg.inv(_opencv2slam)


def _read_slam_intrinsic_and_extrinsic(
    json_path: Path | str,
    image_folder: Path | str,
    output_convention: t.Literal["opencv", "slam"] = "opencv",
) -> t.Tuple[t.Dict[int, Camera], t.Dict[int, Image]]:
    json_path = Path(json_path)
    image_folder = Path(image_folder)
    assert json_path.exists(), f"Path {json_path} does not exist."
    assert (
        image_folder.exists() and image_folder.is_dir()
    ), f"Path {image_folder} does not exist."

    with open(json_path, "r") as f:
        meta_file = json.load(f)

    camera_calibrations = meta_file["calibrationInfo"]
    available_cam_list = list(camera_calibrations.keys())
    cameras = {}

    for camera_id, (cur_camera_name, camera_detail) in enumerate(
        camera_calibrations.items(), start=1
    ):
        model = "PINHOLE"
        width = camera_detail["intrinsics"]["width"]
        height = camera_detail["intrinsics"]["height"]
        params = [
            camera_detail["intrinsics"]["camera_matrix"][0],
            camera_detail["intrinsics"]["camera_matrix"][4],
            camera_detail["intrinsics"]["camera_matrix"][2],
            camera_detail["intrinsics"]["camera_matrix"][5],
        ]
        params = np.array(params).astype(float)

        cameras[camera_id] = Camera(
            id=camera_id,
            model=model,
            width=width,
            height=height,
            params=params,
        )

    def iterate_word2cam_matrix(meta_file, image_folder):
        available_image_names = [
            x.relative_to(image_folder).as_posix()
            for x in chain(image_folder.rglob("*.png"), image_folder.rglob("*.jpeg"))
        ]

        for cur_frame in meta_file["data"]:
            for cur_camera_name, cur_c2w in cur_frame["worldTcam"].items():
                if cur_camera_name in cur_frame["imgName"]:
                    if cur_frame["imgName"][cur_camera_name] in available_image_names:
                        yield cur_frame["imgName"][
                            cur_camera_name
                        ], cur_camera_name, cur_c2w

    images = {}

    extrinsic_raw = list(iterate_word2cam_matrix(meta_file, image_folder))
    logger.info(f"Found {len(extrinsic_raw)} images in the meta file.")

    for image_id, (cur_name, cur_camera_name, cur_frame) in enumerate(extrinsic_raw):
        px = cur_frame["px"]
        py = cur_frame["py"]
        pz = cur_frame["pz"]
        qw = cur_frame["qw"]
        qx = cur_frame["qx"]
        qy = cur_frame["qy"]
        qz = cur_frame["qz"]
        R = np.eye(4)
        qvec = np.array([qw, qx, qy, qz])
        norm = np.linalg.norm(qvec)
        qvec /= norm

        R[:3, :3] = quaternion_matrix(qvec)[:3, :3]
        R[:3, 3] = np.array([px, py, pz])

        if output_convention == "opencv":
            R = _slam2opencv.dot(R)  # this is the c2w in opencv convention.

        world2cam = np.linalg.inv(R)  # this is the w2c in opencv convention.

        # nerfstudio

        Q_ns = quaternion_from_matrix(world2cam[:3, :3])
        T_ns = world2cam[:3, 3]

        # get the camera_id:
        camera_id = available_cam_list.index(cur_camera_name)

        images[image_id] = Image(
            id=image_id,
            qvec=Q_ns,
            tvec=T_ns,
            camera_id=camera_id + 1,
            name=cur_name,
            xys=None,  # noqa
            point3D_ids=None,  # noqa
        )
    return cameras, images


def main(
    meta_json: Path,
    images_path: Path,
    pcd_path: Path,
    downsample_factor: int = 2,
) -> None:
    """Visualize COLMAP sparse reconstruction outputs.

    Args:
        colmap_path: Path to the COLMAP reconstruction directory.
        images_path: Path to the COLMAP images directory.
        downsample_factor: Downsample factor for the images.
    """
    assert meta_json.exists(), f"Path {meta_json} does not exist."
    assert images_path.exists(), f"Path {images_path} does not exist."
    assert pcd_path.exists(), f"Path {pcd_path} does not exist."

    server = viser.ViserServer()
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Load the colmap info.
    cameras, images = _read_slam_intrinsic_and_extrinsic(
        meta_json, images_path, output_convention="opencv"
    )

    pcd = o3d.io.read_point_cloud(pcd_path.as_posix())
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)
    from loguru import logger

    # remove some images
    timeslots = sorted(set([img.name.split("/")[-1] for img in images.values()]))[::5]
    images = {k: v for k, v in images.items() if v.name.split("/")[-1] in timeslots}

    logger.info(
        f"Loaded {len(cameras)} cameras, {len(images)} images, {len(points)} points3d"
    )
    # for p_id in points3d.values():
    #     logger.info(p_id)
    # logger.info(next(iter(points3d.values())))
    gui_reset_up = server.gui.add_button(
        "Reset up direction",
        hint="Set the camera control 'up' direction to the current camera's 'up'.",
    )

    @gui_reset_up.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
            [0.0, -1.0, 0.0]
        )

    gui_points = server.gui.add_slider(
        "Max points",
        min=1,
        max=len(points),
        step=1,
        initial_value=min(len(points), 50_000),
    )
    gui_frames = server.gui.add_slider(
        "Max frames",
        min=1,
        max=len(images),
        step=1,
        initial_value=min(len(images), 100),
    )
    gui_point_size = server.gui.add_slider(
        "Point size", min=0.001, max=0.1, step=0.001, initial_value=0.05
    )

    point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
    point_cloud = server.scene.add_point_cloud(
        name="/colmap/pcd",
        points=points[point_mask],
        colors=colors[point_mask],
        point_size=gui_point_size.value,
    )
    frames: List[viser.FrameHandle] = []

    def visualize_frames() -> None:
        """Send all COLMAP elements to viser for visualization. This could be optimized
        a ton!"""

        # Remove existing image frames.
        for frame in frames:
            frame.remove()
        frames.clear()

        # Interpret the images and cameras.
        img_ids = [im.id for im in images.values()]
        random.shuffle(img_ids)
        img_ids = sorted(img_ids[: gui_frames.value])

        def attach_callback(
            frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle
        ) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(img_ids):
            img = images[img_id]
            cam = cameras[img.camera_id]

            # Skip images that don't exist.
            image_filename = images_path / img.name
            if not image_filename.exists():
                continue

            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(img.qvec), img.tvec
            ).inverse()
            frame = server.scene.add_frame(
                f"/colmap/frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.1,
                axes_radius=0.005,
            )
            frames.append(frame)

            # For pinhole cameras, cam.params will be (fx, fy, cx, cy).
            if cam.model != "PINHOLE":
                print(f"Expected pinhole camera, but got {cam.model}")

            H, W = cam.height, cam.width
            fy = cam.params[1]
            image = iio.imread(image_filename)
            image = image[::downsample_factor, ::downsample_factor]
            frustum = server.scene.add_camera_frustum(
                f"/colmap/frame_{img_id}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.15,
                image=image,
            )
            attach_callback(frustum, frame)

    need_update = True

    @gui_points.on_update
    def _(_) -> None:
        point_mask = np.random.choice(points.shape[0], gui_points.value, replace=False)
        point_cloud.points = points[point_mask]
        point_cloud.colors = colors[point_mask]

    @gui_frames.on_update
    def _(_) -> None:
        nonlocal need_update
        need_update = True

    @gui_point_size.on_update
    def _(_) -> None:
        point_cloud.point_size = gui_point_size.value

    while True:
        if need_update:
            need_update = False
            visualize_frames()

        time.sleep(1e-3)


if __name__ == "__main__":
    tyro.cli(main)
