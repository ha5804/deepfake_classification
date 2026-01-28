# deepfake_classification

1번 프로젝트 목적: deepfake 감지 2번 데이터: 영상, 이미지 3번 데이터 처리: 영상은 프레임을 구간별로 나누어, 이미지로 확인하고, 이미지는 그냥 이미지 자체로 처리한다. 이때 구간별로 나눈 이미지와 그냥 이미지의 얼굴을 crop(배경 제거)하고, align(수평 정렬)한다. 모델이 제대로 학습하게 하기 위해서. 이때 해당 처리를 마친 이미지들을 모델에 맞게 크기 조정과 흐림 정도 처리 작업을 한다. 4번 사전학습된 모델에 crop된 이미지들을 넣어 분류한다. ----- 핵심 평가요소는 결국 영상과 이미지를 어떤식으로 처리하는지, 즉 데이터 가공에 존재한다. 사용하는 모델을 pretraining 되어있으므로, test_data의 이미지를 모델이 잘 파악하도록 하는게 핵심이다.

# 이 프로젝트는
“딥페이크를 잘 분류하는 모델을 만드는 것”이 아니라
“딥페이크 흔적이 가장 잘 보이도록 얼굴을 가공하는 프로젝트”다

# feature of deepfake

(1): eyes - 눈 깜빡임, 좌우 눈 비대칭, 동공 위치, 눈동자 반사광...

(2): lips - 입꼬리 픽셀, 치아 경계, 립싱크, 발음 전환 시 프레임

(3): face boundary - crop의 이유, 턱선이 흐릿, 귀 비틀림, 머리카락 경계, 얼굴 배경 색감 불일치

(4): skin texture - 모공, 노이즈 패턴 반복, 실제 피부에 없는 균일함

(5): temproal inconsistency - 시간적 불일치로, 프레임 간 눈 입 위치 미세가헥 튐, 표정 변화 부자연스러움, 색감이 프레임마다 바뀜.

# file_directory

(1) model/model.pt : 최종 inference에 쓰는 단 하나의 weight이다. huggingFace에서 재 다운로드 혹은 외부 인터넷 없이, model.pt의 weight를 이용하여 inference가 끝나야한다. 즉 models.py에 정의된 모델 구조 + model.pt가 하나의 짝이다.

(2) src/dataset.py : 이미 존재하는 데이터를 읽어서 넘긴다. test데이터의 이미지와 프레임을 로딩하고 모델에 맞게 resize, normalize, tensor 변환을 수행한다.

(3) src/utils.py : 얼굴을 검출(crop)하고, 정렬, frame sampling, blur score, importance sampling, patch 처리 logging과 같은 함수 로직을 작성한다.

(4) config : 실험/ 환경 변경을 코드 수정없이 사용하기 위해, 하드코딩 제거역할

