"""Workspace cell fingerprint from one camera frame.

Put a few printed ArUco markers on fixed parts of the cell (table edge, fixture, wall). At the
start of every session, capture one frame per camera and fingerprint it. Marker pixel
positions catch camera bumps and mounting drift; exposure, sharpness and colour balance
catch lighting change and lens dirt. With camera intrinsics and marker size, marker poses
turn pixel drift into millimetres and degrees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from robotruth.fingerprint.common import DriftFinding, DriftReport, grade, stable_hash

ARUCO_DICTS = {
    "4x4_50": "DICT_4X4_50", "4x4_100": "DICT_4X4_100", "5x5_50": "DICT_5X5_50", "5x5_100": "DICT_5X5_100",
    "6x6_50": "DICT_6X6_50", "6x6_250": "DICT_6X6_250", "apriltag_36h11": "DICT_APRILTAG_36h11",
}


@dataclass
class MarkerObs:
    marker_id: int
    center_px: tuple[float, float]
    corners_px: list[tuple[float, float]]
    size_px: float
    tvec_m: Optional[tuple[float, float, float]] = None
    rvec: Optional[tuple[float, float, float]] = None


@dataclass
class Photometrics:
    mean_luma: float
    std_luma: float
    clipped_dark_frac: float
    clipped_bright_frac: float
    sharpness: float           # variance of Laplacian
    color_balance: tuple[float, float, float]  # mean B, G, R normalised by luma


@dataclass
class CellFingerprint:
    camera_name: str
    width: int
    height: int
    markers: dict[int, MarkerObs]
    photometrics: Photometrics
    aruco_dict: str
    marker_size_m: Optional[float] = None
    notes: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return stable_hash({"cam": self.camera_name, "wh": [self.width, self.height],
                            "markers": {k: [v.center_px, v.size_px] for k, v in sorted(self.markers.items())}})

    def to_json(self, path: Path | str) -> Path:
        d = {"camera_name": self.camera_name, "width": self.width, "height": self.height, "aruco_dict": self.aruco_dict,
             "marker_size_m": self.marker_size_m, "fingerprint": self.fingerprint, "notes": self.notes,
             "photometrics": asdict(self.photometrics), "markers": {str(k): asdict(v) for k, v in self.markers.items()}}
        p = Path(path)
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")
        return p

    @classmethod
    def from_json(cls, path: Path | str) -> "CellFingerprint":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        markers = {}
        for k, v in d["markers"].items():
            v = dict(v)
            v["center_px"] = tuple(v["center_px"])
            v["corners_px"] = [tuple(c) for c in v["corners_px"]]
            v["tvec_m"] = tuple(v["tvec_m"]) if v.get("tvec_m") else None
            v["rvec"] = tuple(v["rvec"]) if v.get("rvec") else None
            markers[int(k)] = MarkerObs(**v)
        ph = dict(d["photometrics"])
        ph["color_balance"] = tuple(ph["color_balance"])
        return cls(d["camera_name"], d["width"], d["height"], markers, Photometrics(**ph), d["aruco_dict"],
                   d.get("marker_size_m"), d.get("notes", []))


def _photometrics(img_bgr: np.ndarray) -> Photometrics:
    import cv2
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    luma = float(gray.mean())
    if img_bgr.ndim == 3:
        means = img_bgr.reshape(-1, 3).mean(axis=0) / max(luma, 1e-6)
        cb = (float(means[0]), float(means[1]), float(means[2]))
    else:
        cb = (1.0, 1.0, 1.0)
    return Photometrics(luma, float(gray.std()), float((gray <= 5).mean()), float((gray >= 250).mean()),
                        float(lap.var()), cb)


def cell_fingerprint(image, camera_name: str = "cam", aruco_dict: str = "4x4_50",
                     camera_matrix: Optional[np.ndarray] = None, dist_coeffs: Optional[np.ndarray] = None,
                     marker_size_m: Optional[float] = None) -> CellFingerprint:
    """image: path or BGR ndarray."""
    import cv2
    img = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
    if img is None:
        raise FileNotFoundError(str(image))
    h, w = img.shape[:2]
    dict_id = getattr(cv2.aruco, ARUCO_DICTS[aruco_dict])
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(img)
    markers: dict[int, MarkerObs] = {}
    notes: list[str] = []
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            pts = c.reshape(-1, 2)
            centre = pts.mean(axis=0)
            size = float(np.mean([np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]))
            obs = MarkerObs(int(i), (float(centre[0]), float(centre[1])), [(float(x), float(y)) for x, y in pts], size)
            if camera_matrix is not None and marker_size_m:
                s = marker_size_m / 2
                obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
                ok, rvec, tvec = cv2.solvePnP(obj, pts.astype(np.float64), camera_matrix,
                                              dist_coeffs if dist_coeffs is not None else np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    obs.tvec_m = tuple(float(x) for x in tvec.flatten())
                    obs.rvec = tuple(float(x) for x in rvec.flatten())
            markers[int(i)] = obs
    else:
        notes.append("no markers detected")
    return CellFingerprint(camera_name, w, h, markers, _photometrics(img), aruco_dict, marker_size_m, notes)


DEFAULT_CELL_TOL = {
    "marker_shift_px": (2.0, 6.0),       # PhAIL: a moved camera shifted completion 22 points
    "marker_size_rel": (0.02, 0.05),     # zoom or distance change
    "marker_shift_mm": (3.0, 10.0),      # when poses are available
    "marker_rot_deg": (1.0, 3.0),
    "mean_luma": (15.0, 40.0),           # of 255
    "sharpness_rel": (0.3, 0.6),         # loss of Laplacian variance (dirt, defocus)
    "color_balance": (0.05, 0.12),
    "clipped_frac": (0.03, 0.10),
}


def diff_cell(ref: CellFingerprint, cur: CellFingerprint, tol: Optional[dict] = None) -> DriftReport:
    import cv2
    tol = {**DEFAULT_CELL_TOL, **(tol or {})}
    rep = DriftReport("cell")
    if (ref.width, ref.height) != (cur.width, cur.height):
        rep.findings.append(DriftFinding("resolution", (ref.width, ref.height), (cur.width, cur.height), 1.0, 0.0, "fail", "Camera resolution changed."))
    missing = sorted(set(ref.markers) - set(cur.markers))
    if missing:
        rep.findings.append(DriftFinding("markers_missing", sorted(ref.markers), sorted(cur.markers), float(len(missing)), 0.0, "fail",
                                         f"Reference markers not seen now: {missing} (occluded, removed, or camera moved a lot)."))
    for mid in sorted(set(ref.markers) & set(cur.markers)):
        a, b = ref.markers[mid], cur.markers[mid]
        shift = float(np.hypot(b.center_px[0] - a.center_px[0], b.center_px[1] - a.center_px[1]))
        w, f = tol["marker_shift_px"]
        rep.findings.append(DriftFinding(f"marker[{mid}].shift_px", a.center_px, b.center_px, shift, f, grade(shift, w, f),
                                         f"marker moved {shift:.1f} px in the image."))
        rel = (b.size_px - a.size_px) / max(a.size_px, 1e-6)
        w, f = tol["marker_size_rel"]
        rep.findings.append(DriftFinding(f"marker[{mid}].size_rel", a.size_px, b.size_px, float(rel), f, grade(rel, w, f),
                                         f"apparent marker size changed {100*rel:+.1f}% (camera distance or zoom)."))
        if a.tvec_m and b.tvec_m:
            dmm = 1000 * float(np.linalg.norm(np.array(b.tvec_m) - np.array(a.tvec_m)))
            w, f = tol["marker_shift_mm"]
            rep.findings.append(DriftFinding(f"marker[{mid}].shift_mm", a.tvec_m, b.tvec_m, dmm, f, grade(dmm, w, f),
                                             f"marker pose moved {dmm:.1f} mm relative to the camera."))
            if a.rvec and b.rvec:
                Ra, _ = cv2.Rodrigues(np.array(a.rvec)); Rb, _ = cv2.Rodrigues(np.array(b.rvec))
                ang = float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))
                w, f = tol["marker_rot_deg"]
                rep.findings.append(DriftFinding(f"marker[{mid}].rot_deg", a.rvec, b.rvec, ang, f, grade(ang, w, f),
                                                 f"marker rotated {ang:.2f} degrees relative to the camera."))
    pa, pb = ref.photometrics, cur.photometrics
    d = pb.mean_luma - pa.mean_luma
    w, f = tol["mean_luma"]
    rep.findings.append(DriftFinding("mean_luma", pa.mean_luma, pb.mean_luma, float(d), f, grade(d, w, f), f"brightness changed {d:+.0f}/255."))
    rel = (pa.sharpness - pb.sharpness) / max(pa.sharpness, 1e-6)
    w, f = tol["sharpness_rel"]
    rep.findings.append(DriftFinding("sharpness", pa.sharpness, pb.sharpness, float(rel), f, grade(rel, w, f) if rel > 0 else "ok",
                                     f"sharpness dropped {100*max(rel,0):.0f}% (defocus, dirt, or motion blur)."))
    cbd = float(np.max(np.abs(np.array(pb.color_balance) - np.array(pa.color_balance))))
    w, f = tol["color_balance"]
    rep.findings.append(DriftFinding("color_balance", pa.color_balance, pb.color_balance, cbd, f, grade(cbd, w, f), f"colour balance shifted {cbd:.3f} (lighting temperature)."))
    clip = max(pb.clipped_bright_frac - pa.clipped_bright_frac, pb.clipped_dark_frac - pa.clipped_dark_frac)
    w, f = tol["clipped_frac"]
    rep.findings.append(DriftFinding("clipping", (pa.clipped_dark_frac, pa.clipped_bright_frac), (pb.clipped_dark_frac, pb.clipped_bright_frac),
                                     float(clip), f, grade(clip, w, f) if clip > 0 else "ok", f"clipped pixel share rose {100*max(clip,0):.1f} points."))
    return rep
