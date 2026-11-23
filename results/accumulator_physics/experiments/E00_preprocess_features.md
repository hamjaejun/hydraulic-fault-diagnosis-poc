# E00_전처리_특징추출

원시 UCI txt → `hyd.npz` → 특징 `feats.npz`(전체 통계 187개 W, 구간·과도응답 1184개 I).

- `python ../load.py ../../data/raw hyd.npz`
- `python ../feat.py`

# 유압 데이터: 독창성 검증 실험 결과 (2026-11-21)

데이터: UCI Hydraulic, 2,205 사이클. 모델: RandomForest(300) 중심, LR과 GB(HistGB)로 교차 확인.
스크립트: `load.py` → `feat.py` → `cv.py`(A) → `cv2.py`(ablation, 조기 진단) → `cv3.py`(축압기 run 구조) → `cv4.py`(냉각기 하나 빼기)
실행 방법: 스크래치 폴더에 `hyd.npz`를 만든 뒤 순서대로 실행.



