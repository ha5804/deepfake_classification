# src/dataset.py
from pathlib import Path
from typing import List
import cv2
import numpy as np
from PIL import Image
import dlib

# landmark model path (config에서 받아도 됨)
LANDMARK_MODEL_PATH = Path("./preprocessing/shape_predictor_81_face_landmarks.dat")

# ---- dlib global initialization ----
if not LANDMARK_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Landmark model not found: {LANDMARK_MODEL_PATH}\n"
        "Please download shape_predictor_81_face_landmarks.dat"
    )

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(str(LANDMARK_MODEL_PATH))

def blur_score(frame_rgb: np.ndarray) -> float:
    """프레임 선명도 측정 (클수록 선명)"""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def importance_frame_indices(
    frames: list[np.ndarray],
    num_frames: int
) -> np.ndarray:
    """
    중요도 기반 프레임 샘플링
    - 얼굴이 크고
    - 블러가 적고
    - 얼굴 검출이 성공한 프레임 우선
    """
    if len(frames) <= num_frames:
        return np.arange(len(frames))

    scores = []

    for idx, frame in enumerate(frames):
        # 기본 점수
        score = 0.0

        # 1️⃣ 얼굴 검출
        faces = face_detector(frame, 1)
        if len(faces) == 0:
            continue  # 얼굴 없으면 importance 낮음

        # 가장 큰 얼굴 사용
        face = max(faces, key=lambda r: r.width() * r.height())
        face_area = face.width() * face.height()

        # 2️⃣ blur score
        sharpness = blur_score(frame)

        # importance score 계산
        score = (
            0.6 * face_area +      # 얼굴 크기
            0.4 * sharpness        # 선명도
        )

        scores.append((idx, score))

    if len(scores) == 0:
        # fallback: uniform sampling
        return uniform_frame_indices(len(frames), num_frames)

    # importance 높은 순으로 정렬
    scores.sort(key=lambda x: x[1], reverse=True)

    # 상위 num_frames 선택
    selected = [idx for idx, _ in scores[:num_frames]]

    return np.array(sorted(selected))


def read_rgb_frames(file_path: Path, num_frames: int = NUM_FRAMES) -> List[np.ndarray]:
    """이미지 또는 비디오에서 RGB 프레임 추출"""
    ext = file_path.suffix.lower()
    
    if ext in IMAGE_EXTS:
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                return []
            return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
        except Exception:
            return []
    
    if ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(file_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total <= 0:
            cap.release()
            return []
        
        frame_indices = uniform_frame_indices(total, num_frames)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames
    
    return []

def get_5_keypoints(image_rgb: np.ndarray, face: dlib.rectangle) -> np.ndarray:
    """
    81개 랜드마크에서 5개의 core point 추출
    - left eye (#37), right eye (#44), nose (#30)
    - left mouth (#49), right mouth (#55)
    """
    shape = landmark_predictor(image_rgb, face)
    
    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2)
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)
    
    pts = np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)
    return pts


def align_and_crop_face(img_rgb: np.ndarray, landmarks: np.ndarray, 
                        outsize: Tuple[int, int] = (224, 224), 
                        scale: float = 1.3) -> np.ndarray:
    """
    5개 랜드마크를 사용하여 얼굴 정렬 및 crop
    """
    target_size = [112, 112]
    dst = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]
    ], dtype=np.float32)

    if target_size[1] == 112:
        dst[:, 0] += 8.0

    dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0]
    dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

    target_size = outsize

    margin_rate = scale - 1
    x_margin = target_size[0] * margin_rate / 2.
    y_margin = target_size[1] * margin_rate / 2.

    dst[:, 0] += x_margin
    dst[:, 1] += y_margin

    dst[:, 0] *= target_size[0] / (target_size[0] + 2 * x_margin)
    dst[:, 1] *= target_size[1] / (target_size[1] + 2 * y_margin)

    src = landmarks.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(src, dst)
    M = tform.params[0:2, :]

    aligned = cv2.warpAffine(img_rgb, M, (target_size[1], target_size[0]))
    
    if outsize is not None:
        aligned = cv2.resize(aligned, (outsize[1], outsize[0]))
    
    return aligned


def extract_aligned_face_fast(img_rgb: np.ndarray, res: int = 224, scale: float = 0.8) -> Optional[np.ndarray]:
    """
    얼굴 검출 및 정렬 (축소된 이미지에서 검출)
    - scale: 이미지 축소 비율 (0.8 = 80% 크기로 축소) -> time cost 감소
    - 얼굴이 없으면 None 반환
    """
    small = cv2.resize(img_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    faces = face_detector(small, 1)
    
    if len(faces) == 0:
        return None
    
    face = max(faces, key=lambda r: r.width() * r.height())
    landmarks = get_5_keypoints(small, face)
    aligned = align_and_crop_face(small, landmarks, outsize=(res, res))
    
    return aligned