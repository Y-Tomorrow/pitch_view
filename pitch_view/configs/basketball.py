from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class League(Enum):
    """
    Supported court dimension standards. FIBA is the default; NBA differs
    slightly (longer/wider court, longer three-point line, narrower lane).
    """
    FIBA = "FIBA"
    NBA = "NBA"


@dataclass
class BasketballCourtConfiguration:
    """
    Dimensions of a basketball court in centimetres, laid out with the origin
    (0, 0) at the top-left corner, the x-axis along the length and the y-axis
    along the width.

    Defaults are FIBA. See ``from_league`` to build an NBA court.
    """

    length: int = 2800  # [cm] full court length (28 m)
    width: int = 1500  # [cm] full court width (15 m)
    three_point_distance: int = 675  # [cm] arc radius from the hoop centre
    three_point_corner_distance: int = 660  # [cm] straight segment from centreline
    free_throw_distance: int = 580  # [cm] baseline -> free-throw line
    free_throw_circle_radius: int = 180  # [cm]
    centre_circle_radius: int = 180  # [cm]
    lane_width: int = 490  # [cm] paint / key width
    basket_distance: int = 157  # [cm] hoop centre from the baseline
    backboard_distance: int = 120  # [cm] backboard from the baseline

    @classmethod
    def from_league(cls, league: League) -> "BasketballCourtConfiguration":
        """
        Build a configuration for a specific league standard.
        """
        if league is League.FIBA:
            return cls()
        if league is League.NBA:
            return cls(
                length=2865,  # 94 ft
                width=1524,  # 50 ft
                three_point_distance=724,  # 23.75 ft
                three_point_corner_distance=670,  # 22 ft
                free_throw_distance=579,  # 19 ft
                lane_width=488,  # 16 ft
            )
        raise ValueError(f"Unsupported league: {league}")

    # ------------------------------------------------------------------ #
    # Structural keypoints used for homography (pixel -> 2D court map).
    #
    # IMPORTANT: this list defines the EXPECTED keypoint order of the court
    # detection model. ``keypoint_vertices[i]`` is the 2D court coordinate that
    # keypoint ``i`` (in the model's output) corresponds to. If your model uses
    # a different keypoint order, edit this list / the labels to match it.
    # ------------------------------------------------------------------ #

    @property
    def court_corners(self) -> List[Tuple[int, int]]:
        """Four court corners in order [top-left, top-right, bottom-right, bottom-left]."""
        return [
            (0, 0),
            (self.length, 0),
            (self.length, self.width),
            (0, self.width),
        ]

    @property
    def keypoint_labels(self) -> List[str]:
        """Names of the court keypoints, in the model's output order."""
        return [
            "court_corner_tl",  # 0
            "court_corner_tr",  # 1
            "court_corner_br",  # 2
            "court_corner_bl",  # 3
            "basket_left",  # 4
            "basket_right",  # 5
        ]

    @property
    def keypoint_vertices(self) -> List[Tuple[int, int]]:
        """2D court coordinates (x, y) for each keypoint, same order as labels."""
        return [
            (0, 0),  # 0 court_corner_tl
            (self.length, 0),  # 1 court_corner_tr
            (self.length, self.width),  # 2 court_corner_br
            (0, self.width),  # 3 court_corner_bl
            (self.basket_distance, self.width // 2),  # 4 basket_left
            (self.length - self.basket_distance, self.width // 2),  # 5 basket_right
        ]
