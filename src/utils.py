import cv2
from skimage import transform as trans
import numpy as np

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

def align_and_crop_face(
    img_rgb,
    landmarks,
    outsize=(224, 224),
    scale=1.3
):
    target = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041],
    ], dtype=np.float32)
    #randmark 얼굴위치, 눈, 코, 입 ..

    target[:, 0] *= outsize[0] / 112
    target[:, 1] *= outsize[1] / 112
    #224 입력 크기에 맞추기 위해 스케일 조정

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks, target)
    M = tform.params[:2]
    #현재 얼굴의 랜드마크에서 target 랜드 마크로 가는 변환

    aligned = cv2.warpAffine(img_rgb, M, outsize)
    #얼굴 정렬
    return aligned

def extract_face(
    img_rgb,
    face_detector,
    landmark_predictor,
    scale=1.0,
    outsize=224
):
    small = cv2.resize(img_rgb, None, fx=scale, fy=scale)
    faces = face_detector(small, 1)

    if len(faces) == 0:
        return None

    face = max(faces, key=lambda r: r.width() * r.height())
    landmarks = get_5_keypoints(landmark_predictor, small, face)

    return align_and_crop_face(
        small, landmarks, outsize=(outsize, outsize)
    )
