"""End-to-end offside review on a synthetic broadcast frame with known ground truth.

Renders a perspective view of the pitch with two kits, feeds "detections" and
a keypoint calibration through the same `review_frame` used on real footage,
and checks the call against the true pitch positions.

    python examples/synthetic_var_review.py
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

from varflash.paths import OUTPUTS
from varflash.vision.homography import from_keypoints
from varflash.vision.offside import judge_offside
from varflash.vision.pitch import KEYPOINTS, draw_pitch
from varflash.vision.var_review import review_frame

W, H = 1600, 900
SCALE, MARGIN = 8.0, 4.0
KITS = {"attack": (219, 170, 117), "defend": (30, 43, 213), "ref": (30, 30, 30)}  # BGR

# Pitch metres -> broadcast pixels, looking at the right half from the main stand.
CAMERA = cv2.getPerspectiveTransform(
    np.float32([[45, 0], [105, 0], [105, 68], [45, 68]]),
    np.float32([[380, 170], [1500, 170], [1720, 860], [-120, 860]]),
)


def to_image(pts):
    return cv2.perspectiveTransform(np.float64(pts).reshape(-1, 1, 2), CAMERA).reshape(-1, 2)


def render(players):
    top, px = draw_pitch(SCALE, MARGIN)
    to_top = np.array([[SCALE, 0, MARGIN * SCALE], [0, SCALE, MARGIN * SCALE], [0, 0, 1]])
    frame = cv2.warpPerspective(top, CAMERA @ np.linalg.inv(to_top), (W, H), borderValue=(30, 90, 30))
    dets = []
    for x, y, kit in sorted(players, key=lambda p: p[1]):  # far players drawn first
        u, v = to_image([[x, y]])[0]
        h = 40 + 70 * (v / H)  # nearer = taller
        w = h * 0.38
        x1, y1, x2, y2 = u - w / 2, v - h, u + w / 2, v
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y1 + 0.55 * h)), KITS[kit], -1)
        cv2.rectangle(frame, (int(x1 + 2), int(y1 + 0.55 * h)), (int(x2 - 2), int(y2)), (20, 20, 20), -1)
        cv2.circle(frame, (int(u), int(y1 - w * 0.3)), int(w * 0.3), (150, 180, 220), -1)
        dets.append({"kind": "person", "conf": 0.9, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "u": u, "v": v})
    return frame, pd.DataFrame(dets)


def main():
    rng = np.random.default_rng(7)
    attackers = [(88.6, 30.0), (80.0, 45.0), (72.0, 20.0), (65.0, 38.0)]
    defenders = [(88.2, 25.0), (86.0, 40.0), (84.5, 52.0), (78.0, 15.0), (70.0, 30.0)]
    players = [(x, y, "attack") for x, y in attackers] + [(x, y, "defend") for x, y in defenders]
    players.append((75.0, 60.0, "ref"))
    frame, dets = render(players)

    # Simulate a human/automatic keypoint calibration with ~2 px click error.
    names = ["halfway_top", "halfway_bottom", "rbox_top", "rbox_bottom", "rsix_top", "rsix_bottom", "corner_tr", "circle_right"]
    clicks = to_image([KEYPOINTS[n] for n in names]) + rng.normal(0, 2, (len(names), 2))
    homography = from_keypoints(dict(zip(names, map(tuple, clicks))))

    review = review_frame(frame, dets, homography, attack_right=True, attacking_color="#75AADB")
    truth = judge_offside([a[0] for a in attackers], [d[0] for d in defenders], ball_x=60.0)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    mm = cv2.resize(review.minimap, (W // 3, int(review.minimap.shape[0] * (W // 3) / review.minimap.shape[1])))
    out = review.overlay.copy()
    out[H - mm.shape[0] - 10 : H - 10, W - mm.shape[1] - 10 : W - 10] = mm
    cv2.imwrite(str(OUTPUTS / "synthetic_var_review.jpg"), out)

    c = review.call
    print(f"calibration error: {homography.reprojection_error_m:.2f} m over {homography.n_points} points")
    print(f"teams found: {review.players['team'].value_counts().to_dict()}")
    print(f"AI VAR : offside={c.offside}  margin={c.margin_m:+.2f} m  P(offside)={c.probability:.2f}  line x={c.line_x:.2f}")
    print(f"truth  : offside={truth.offside}  margin={truth.margin_m:+.2f} m  line x={truth.line_x:.2f}")
    print(f"image  -> {OUTPUTS / 'synthetic_var_review.jpg'}")


if __name__ == "__main__":
    main()
