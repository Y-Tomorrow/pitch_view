from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt


class ViewTransformer:
    """
    Maps points / images from a camera view to a top-down pitch using a
    homography from point correspondences.

    After the warp, destination pixels that would sample outside the frame (or
    from a thin strip on the image border) are cleared. That border strip is
    what otherwise gets smeared across the unseen half of the 2D pitch.
    """

    def __init__(
            self,
            source: npt.NDArray[np.float32],
            target: npt.NDArray[np.float32],
            image_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self._image_shape = image_shape
        # RANSAC drops a point that disagrees with the others. The threshold is
        # in target pixels, so scale it with the pitch drawing size.
        span = float(np.ptp(target, axis=0).max()) if len(target) else 0.0
        thresh = max(3.0, 0.015 * span)
        self.m, _ = cv2.findHomography(source, target, cv2.RANSAC, thresh)
        if self.m is None:
            self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")
        self.m = self.m / self.m[2, 2]

    def transform_points(
            self,
            points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        out = cv2.perspectiveTransform(reshaped, self.m)
        return out.reshape(-1, 2).astype(np.float32)

    def transform_image(
            self,
            image: npt.NDArray[np.uint8],
            resolution_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        if len(image.shape) not in {2, 3}:
            raise ValueError("Image must be either grayscale or color.")
        dsize = (int(resolution_wh[0]), int(resolution_wh[1]))
        border = (0, 0, 0) if image.ndim == 3 else 0
        warped = cv2.warpPerspective(
            image, self.m, dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border,
        )
        h, w = image.shape[:2]
        if self._image_shape is not None:
            h, w = int(self._image_shape[0]), int(self._image_shape[1])
        # Do not sample the frame edge: those few pixels are stretched over
        # the rest of the pitch plane by the homography.
        inset = np.zeros((h, w), dtype=np.uint8)
        m = max(3, int(0.012 * min(h, w)))
        inset[m:h - m, m:w - m] = 255
        cover = cv2.warpPerspective(
            inset, self.m, dsize,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped[cover == 0] = 0
        return warped
