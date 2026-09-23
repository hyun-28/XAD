# PREREG_W2 — W2 사전 등록 (erasure artifact study)

**작성:** 2026-09-23 · **근거:** BRIEF3 §2 (Task P0), 연구자 확정본
**수정 이력:** 예측 ②의 복원 지우개를 `NearestNeighborWindow` → `ContextReconstruct`로 변경
(2026-09-23, 연구자 결정, 파일럿 실행 전)
**상태:** 이 파일은 W2-0 파일럿(`scripts/w2_pilot.py`)을 **실행하기 전에** 커밋된다.
커밋 해시와 시각은 `reports/w2_pilot.md`에 기록된다.

---

## 예측

- **예측 ①** 지운 길이 r이 창 길이 m보다 커질수록, 탐지기 Z(z-정규화 MP)에서만 상수 계열 지우개(Zero, SampleMean, OutOfDistHigh) 간 자국 차이가 줄어든다. 탐지기 R(비정규화 MP)에서는 줄지 않는다.
- **예측 ②** 복원 지우개(ContextReconstruct)가 가장 작은 자국을 남긴다.
- **예측 ③** (W3) 자국을 뺀 보정 전후로 설명기 순위가 바뀐다.

## 지표

- **주 지표:** 가짜 경보 여부 = 지운 뒤 영향받은 창의 최대 점수 > 원래 정상 테스트 점수의 99 백분위수 (게이트 1 P1과 같은 임계값 정의).
- **보조 지표:** ΔS(영향받은 창 점수 평균 변화, 내부/경계 창 분리).

## 분석 단위

- content_group 89, Wilcoxon + Holm.

## 부록 — 예측 ②가 가리키는 지우개의 정의 (연구자 정의, 2026-09-23)

**ContextReconstruct** (우리 정의, Šimić et al.에 없음)

- 지운 구간 `[a, a+r)` 바로 앞 m점 `x[a-m:a]`를 쿼리로, 학습 구간에서 z-정규화 거리 최근접 위치 j를 찾음 (단 `j+m+r ≤ train_end`)
- 채우는 값: `x_train[j+m : j+m+r] + (x[a-1] - x_train[j+m-1])` (왼쪽 이음매 오프셋 정렬)
- 지울 구간 자체는 참조하지 않음

`NearestNeighborWindow`(Šimić 원 정의, 위치 기준 좌·우 이웃 반씩 이어 붙임)는 원본 정의 그대로
유지하며 보고서에서는 "이웃 붙이기"로 표기한다.

## 공개 사항 — 이 문서 커밋 전에 본 수치

계획 수립 중(2026-09-23) 파일럿 후보 시리즈 #066에서 scratch 실측을 했다. 목적은 sanity check
기준(S1)이 수치적으로 성립하는지 확인하는 것이었다. 이때 본 것:

- 테스트 구간 한 위치(a=500, 테스트 기준)에서 Zero / 시리즈 평균 / OutOfDistHigh(원시 스케일)로
  r ∈ {13, 26, 52, 104}를 지운 뒤 전체를 재계산했을 때, 영향받지 않은 창의 잔차: Z ≤ 7.8e-12, R ≤ 3.9e-10.
- 같은 실행에서 본 내부 창 점수: Z는 세 지우개 모두 √26 = 5.0990. R은 0.867 / 0.845 / 210.9
  (그때 쓴 채움 값은 원시 스케일이었고, 확정된 표준화 공간 정의와 다르다).
- 게이트 1 `detection.csv`의 #066 Z 임계값 `thr_bw = 0.8045`.

ContextReconstruct, NearestNeighborWindow, Inverse, UniformNoise100, LinearInterpolation은 이 문서를
커밋하기 전에 한 번도 실행하지 않았다. 파일럿 수치는 논문 주장에 쓰지 않는다(BRIEF3 §0).
