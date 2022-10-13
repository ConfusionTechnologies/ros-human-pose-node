"""
Below was extracted from https://github.com/Interpause/nicepipe/blob/main/nicepipe/analyze/mmpose.py

Below is a replica of the pre & post-processing pipeline of 
https://github.com/open-mmlab/mmpose/blob/master/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/vipnas_res50_coco_wholebody_256x192_dark.py

I took liberties when implementing it for conciseness & performance
"""
from __future__ import annotations

import cv2
import numpy as np

# the feels when 50% of the lag is from normalizing the image
# should normalize the crops instead i guess
# TODO: figure out to how optimize or parallelize the taylor (70%) and guassian (20%) parts
# seriously post-processing shouldnt take more time than the model inference...

# Original Pipeline steps
# 1. convert image from BGR to RGB
# 2. For each bbox (abs_XYWH) calculate abs centre (x,y) & std scale (sx,sy)
#   a. padding is 1.25 by default for coco wholebody
#   b. (sx,sy) is divided by 200 for some inter-dataset compatability reason
# 3. Do some weird affine transformation cropping shit. Screw that tho
#   a. Crop out image with padding using centrescale as (HWC)
#   b. at this stage, image is still uint8 0-255 RGB HWC
#   c. image isnt resized, only cropped. If bbox exceeds image, pad with black


def bbox_xyxy2cs(bbox: np.ndarray, ratio_wh=192 / 256, pad=1.25):
    """Converts abs bbox N*(x,y,x,y) to abs centre N*(x,y) & abs scale N*(sx,sy)"""
    x1, y1, x2, y2 = bbox.T[:4]
    w, h = x2 - x1, y2 - y1
    centers = np.stack((x1 + 0.5 * w, y1 + 0.5 * h), axis=1)

    mask = w > ratio_wh * h
    h[mask] = w[mask] / ratio_wh
    w[~mask] = h[~mask] * ratio_wh
    scales = np.stack((w, h), axis=1) * pad

    return centers, scales


def crop_bbox(
    img: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
    crop_wh: tuple[int, int] = (192, 256),
):
    """From 1 HWC img, crop N*HWC crops from N*(x,y) & N*(sx,sy)"""
    im_height, im_width = img.shape[:2]
    dw, dh = crop_wh
    N = centers.shape[0]
    crops = np.zeros((N, dh, dw, 3), dtype=img.dtype)

    # source xyxy
    s = scales / 2
    rects = np.tile(centers, 2) + np.concatenate((-s, s), axis=1)

    # calculate margin required when rect exceeds image
    ml = np.maximum(-rects[:, 0], 0).astype(int)
    mt = np.maximum(-rects[:, 1], 0).astype(int)
    # mr = np.maximum(rects[:, 2] - im_width, 0)
    # mb = np.maximum(rects[:, 3] - im_height, 0)

    # clip rects to within image
    rects[:, (0, 2)] = rects[:, (0, 2)].clip(0, im_width)
    rects[:, (1, 3)] = rects[:, (1, 3)].clip(0, im_height)
    rects = rects.astype(int)
    scales = np.ceil(scales).astype(int)

    for n, ((x1, y1, x2, y2), (w, h), l, t) in enumerate(zip(rects, scales, ml, mt)):
        roi = img[y1:y2, x1:x2, :]
        crop = np.zeros((h, w, 3), dtype=img.dtype)
        crop[t : t + y2 - y1, l : l + x2 - x1] = roi
        crops[n, ...] = cv2.resize(crop, (dw, dh))

    return crops


# yanked straight from mmpose. my brain too puny to vectorize this.
def taylor(heatmap, coord):
    """Distribution aware coordinate decoding method.

    Note:
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmap (np.ndarray[H, W]): Heatmap of a particular joint type.
        coord (np.ndarray[2,]): Coordinates of the predicted keypoints.

    Returns:
        np.ndarray[2,]: Updated coordinates.
    """
    H, W = heatmap.shape[:2]
    px, py = int(coord[0]), int(coord[1])
    if 1 < px < W - 2 and 1 < py < H - 2:
        dx = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        dy = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        dxx = 0.25 * (heatmap[py][px + 2] - 2 * heatmap[py][px] + heatmap[py][px - 2])
        dxy = 0.25 * (
            heatmap[py + 1][px + 1]
            - heatmap[py - 1][px + 1]
            - heatmap[py + 1][px - 1]
            + heatmap[py - 1][px - 1]
        )
        dyy = 0.25 * (
            heatmap[py + 2 * 1][px] - 2 * heatmap[py][px] + heatmap[py - 2 * 1][px]
        )
        derivative = np.array([[dx], [dy]])
        hessian = np.array([[dxx, dxy], [dxy, dyy]])
        if dxx * dyy - dxy ** 2 != 0:
            hessianinv = np.linalg.inv(hessian)
            offset = -hessianinv @ derivative
            offset = np.squeeze(np.array(offset.T), axis=0)
            coord += offset
    return coord


# yanked straight from mmpose, vectorizing this is too hard for my puny brain esp cause cv2 is used
def gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border), dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps


def remap_keypoints(
    coords: np.ndarray,  # (n, keypoints, 2)
    center: np.ndarray,  # (n, 2)
    scale: np.ndarray,  # (n, 2)
    heatmap_wh: tuple[int, int],
):
    factor = (scale / heatmap_wh).reshape(-1, 1, 2)
    center = center.reshape(-1, 1, 2)
    scale = scale.reshape(-1, 1, 2)
    return coords * factor + center - scale * 0.5


def heatmap2keypoints(
    heatmaps: np.ndarray, centers: np.ndarray, scales: np.ndarray, post_process=True
):
    n, k, h, w = heatmaps.shape

    # processing heatmap into coords and conf
    tmp1 = heatmaps.reshape((n, k, -1))
    ind = np.argmax(tmp1, 2).reshape((n, k, 1))
    maxvals = np.amax(tmp1, 2).reshape((n, k, 1))

    preds = np.tile(ind.astype(np.float32), (1, 1, 2))
    preds[..., 0] = preds[..., 0] % w
    preds[..., 1] = preds[..., 1] // w
    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)

    if post_process:
        heatmaps = np.log(np.maximum(gaussian_blur(heatmaps, 11), 1e-10))
        for i in range(n):
            for j in range(k):
                preds[i][j] = taylor(heatmaps[i][j], preds[i][j])

    # mask = 1 < preds[:, :, 0] < w - 1 & 1 < preds[:, :, 1] < h - 1
    # diff = np.array(tuple(heatmaps))

    return remap_keypoints(preds, centers, scales, (w, h)), maxvals
