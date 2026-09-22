# 부록 브리프 — Task A 후속 조치 (Task B 착수 전 필수)

> **대상:** Claude Code · **작성:** 2026-09-22 · 연구 파트너(claude.ai) → 구현(Claude Code) 인계
> **전제:** `BRIEF_W1.md` Task A 완료본(`reports/data_audit.md`, 2026-09-21 생성)을 연구자와 연구 파트너가 검토함.
> **원칙:** `BRIEF_W1.md` §1 비협상 원칙과 §7 STOP 조건은 그대로 적용된다 (`CLAUDE.md` 참조).
> **범위:** 이 문서의 F1–F7을 끝내기 전에는 Task B를 시작하지 않는다.

---

## 0. 검토 결과 요약

Task A는 통과. 문서·실측 근거 수집, Goswami 매핑 원본 확보, NOISE 변형 발견, −999 탐지까지 기대 이상.
다만 결과 안에서 브리프가 다루지 않은 문제 두 개가 드러났고, 둘 다 이후 모든 통계에 영향을 준다.

1. **250개 시리즈는 독립 표본이 아니다.** 이름에서 `DISTORTED`/`NOISE` 접두사를 떼면 고유 이름은 100개.
   같은 녹음의 변형들을 독립 표본으로 검정에 넣으면 유사반복(pseudoreplication)으로 p값이 과소 추정된다.
2. **−999 결측 표시값이 정상 구간에 있다.** 23개 plain 시리즈에서 발견, 일부는 학습 구간에만 수십 개.
   우리 연구의 핵심 측정량(정상 구간에서 섭동이 점수를 얼마나 올리는가)의 기준선을 오염시킨다.
   또한 23개 전부 plain이라는 점에서, DISTORTED/NOISE 쌍둥이에서는 노이즈 때문에 정확히 −999가 아니어서
   정확 일치 검사로는 놓쳤을 가능성이 높다.

---

## F1. 연구자 결정 기록 (코드 변경 없음)

`docs/DECISIONS.md`에 아래 세 항목을 추가한다.

### D-A2-3 — D3 인덱스 규약 승인 (2026-09-22, 연구자 승인)

* `matlab_1based_inclusive` 확정. `configs/conventions.yaml`의 "sign-off pending" 문구를 "approved 2026-09-22"로 교체.
* 근거: 문서(슬라이드 4, 5, 6, 52–53, 90–99) + 실측 결정적 사례 4개(077/185 resperation11의 −999 스파이크,
  092/200 tiltAPB4 드롭아웃 — 0-based 해석 시 스파이크가 구간 밖).
* 108번(NOISEresperation2)의 0-based 판정은 두 후보 지점 모두 노이즈 구간 내부로 판정 불가로 간주. 쌍둥이 079·187도 판정 불가.
* `reports/data_audit.md` §6의 "D3 status" 문구도 스크립트에서 갱신되게 할 것.

### D-A1-4 — 데이터 라이선스 확정 (2026-09-22, 연구자 승인)

* 출처 페이지 `https://www.cs.ucr.edu/~eamonn/time_series_data_2018/` 재확인: 라이선스·이용조건 문구 없음.
  페이지의 인용(Dau et al. 2019)은 분류 아카이브용이므로 사용하지 않는다.
* CC BY-NC 4.0 항목 삭제: 연구자 개인 웹사이트(wu.renjie.im) 하단 표기로, 아카이브 라이선스가 아님.
* 결정: 명시 라이선스 없는 공개 연구 데이터로 취급. 원본 시리즈 재배포 금지(.gitignore 유지),
  README에 공식 URL + SHA256 대조 절차 안내. 매니페스트·점수·그림 등 파생물은 공개.
* 인용 2건: (1) `Start_Here_This_is_a_Read_me.pptx`의 인용 요청 원문 그대로 — Keogh, Dutta Roy, Naik & Agrawal (2021),
  Multi-dataset Time-Series Anomaly Detection Competition, SIGKDD 2021. (2) Wu & Keogh, IEEE TKDE 35(3):2421–2429, 2023.
* `docs/DATA_LICENSE.md`를 이 내용으로 갱신하고 "unresolved" 표기 제거.

### D-A5-4 — 서브셋 정의 분리 (2026-09-22, 연구자 승인)

* `is_medical` 정의(D-A5-3)는 유지. 단 "비의료"는 연구자 지도교수가 선호하는 "기계·산업"과 다른 개념이다.
* 신규 서브셋 `physical` = `domain_goswami ∈ {Acceleration, Air Temperature, NASA, Power Demand}`. 구현은 F4.

---

## F2. 녹음 그룹 (`recording_group`) — 가장 중요

### 왜

통계 검정의 표본 단위는 서로 독립인 관측이어야 한다. plain·DISTORTED·NOISE 변형은 같은 녹음에서 나온 데이터라
탐지기·explainer의 성능이 강하게 상관된다. 250개를 독립으로 취급하면 실제보다 표본이 크다고 믿게 된다.

### 무엇을

두 수준의 그룹을 만든다. 보수적인 쪽을 주 분석 단위로 쓴다.

**(a) `name_group`** — 이름에서 `^(DISTORTED|NOISE)` 접두사만 제거한 문자열. 현재 기준 100개 그룹.

* 이 그룹은 과소 병합(서로 다른 녹음인데 이름만 비슷한 경우는 이미 분리됨)보다는 과다 병합
  (예: `STAFFIIIDatabase` 9개, `ECG4` 9개가 실제로는 다른 환자·다른 구간일 수 있음) 위험이 있다.
  과다 병합은 표본을 줄이는 방향이라 p값을 보수적으로 만든다. 그래서 주 분석 단위로 적합하다.

**(b) `content_group`** — 실제 신호가 같은지로 묶은 그룹. `name_group`의 검증용.

* 같은 `name_group` 안의 모든 쌍에 대해: 길이, `train_end`, 그리고 학습 구간 앞부분(최대 5,000점)의
  z-정규화 후 피어슨 상관을 계산.
* 상관 ≥ 0.95면 같은 녹음으로 판정(임계값은 설정값, 근거와 함께 DECISIONS에 기록). DISTORTED는
  노이즈·기저선 변동이 더해진 변형이므로 상관이 1이 아닐 수 있다 — 실측 분포를 보고 임계값이 합리적인지 보고할 것.
* 이름이 다른 그룹 사이의 중복도 확인: 같은 `domain_goswami` 안에서 서로 다른 `name_group` 쌍에 대해 같은 검사.
  (전체 쌍이 많으면 길이가 ±1% 이내인 쌍만 검사해도 된다. 필터 규칙을 기록할 것.)

### 산출물

* 매니페스트에 열 추가: `name_group`, `content_group`, `name_group_size`, `content_group_size`.
* `reports/recording_groups.csv`: 그룹 내·그룹 간 모든 비교 쌍의 상관값과 판정.
* `reports/data_audit.md`에 새 절 §7 Recording groups:
  * `name_group` 크기 분포, `content_group` 크기 분포
  * 두 그룹핑이 어긋나는 시리즈 목록 (이름은 같은데 내용이 다름 / 이름은 다른데 내용이 같음)
  * 모든 서브셋 표에 `n_series`와 `n_groups`를 나란히 표기

### 이후 모든 분석에 적용할 규칙 (`configs/stats.yaml`에 명시)

```yaml
unit_of_analysis: name_group        # 주 분석. content_group은 민감도 분석
aggregate_within_group: mean        # 그룹 내 시리즈 결과를 평균한 뒤 검정
bootstrap_resample_unit: name_group # 부트스트랩은 그룹 단위로 재표집
min_groups_for_hypothesis_test: 30  # 이 미만인 서브셋은 기술통계만, 검정 금지
```

---

## F3. −999 결측 표시값 (sentinel)

### 왜

우리 연구는 정상 구간에 섭동을 넣고 점수 변화를 잰다. 정상 구간에 이미 −999(≈ −1000) 스파이크가 있으면
(1) 탐지기가 그 지점에서 이미 높은 점수를 내서 기준선이 오염되고, (2) 섭동 구간이 스파이크를 덮으면 섭동이
오히려 점수를 낮출 수도 있다. 사실상 데이터 안에 이미 존재하는 하드 마스킹이다.

### 무엇을

**F3-1. 전수 탐지 (값 기준)**

* 250개 전부에서 정확히 −999인 지점과, `|x + 999| ≤ tol`인 지점을 따로 센다. `tol`은 설정값, 기본은 해당 시리즈
  학습 구간 MAD의 3배.
* 결과를 `exact`와 `near` 두 열로 분리해 기록.

**F3-2. 쌍둥이 검사 (위치 기준) — 핵심**

* plain 시리즈에서 −999가 나온 각 인덱스에 대해, 같은 `name_group`의 DISTORTED/NOISE 쌍둥이의 같은 인덱스에서:
  * 해당 점이 국소 이상치인지: `|x[i] − rolling_median(x, w)[i]| > k × rolling_MAD(x, w)[i]` (w, k는 설정값, 기본 w=51, k=10)
  * 결과: `confirmed`(쌍둥이에도 스파이크) / `absent`(쌍둥이에 없음) / `length_mismatch`(인덱스가 쌍둥이 길이를 넘음)
* 쌍둥이의 길이가 plain과 다르면 인덱스 정렬이 성립하지 않을 수 있다 → `length_mismatch`로 기록하고 정렬을 추측하지 말 것.

**F3-3. 위치 분류** — 각 sentinel 지점을 다음 중 하나로 분류:

* `in_gt` — 레이블된 이상 구간 안 (예: 185번 resperation11은 −999 자체가 이상)
* `in_train` — 학습 구간(`x[:train_end]`)
* `in_test_normal` — 학습 구간 이후, 이상 구간 밖

**F3-4. 제외 인덱스 저장 — 데이터는 절대 수정하지 않는다**

* 원본 값을 보간·대체하지 않는다. (보간 여부는 연구 설계 결정이며 이번 범위가 아니다.)
* 시리즈별 sentinel 인덱스(0-based, `in_gt` 제외)를 `data/derived/sentinels/<num>.npy`로 저장.
* 이 인덱스 ± 여백은 이후 정상 구간 표본 추출과 백분위 임계값 계산에서 제외하는 데 쓰인다. 여백 크기는 Task B에서
  탐지기 윈도우가 정해진 뒤 결정하므로, 지금은 원시 인덱스만 저장.

### 산출물

* 매니페스트 열 추가: `sentinel_exact_n`, `sentinel_near_n`, `sentinel_in_gt_n`, `sentinel_in_train_n`,
  `sentinel_in_test_normal_n`, `sentinel_twin_status` (`confirmed`/`absent`/`length_mismatch`/`NA`)
* 감사 플래그 등급 변경: `in_test_normal_n > 0` 또는 `in_train_n > 0`이면 `HAS_MINUS999`를 INFO → WARN.
  (종료 코드 규칙 D-A3-2에 따라 WARN은 exit 1. 이번 실행에서는 예상된 WARN이므로 보고서에 "예상됨, F3 처리 완료"로 표기.)
* `reports/data_audit.md` 새 절 §8 Sentinels: 시리즈별 표, 쌍둥이 검사 결과, `in_test_normal`이 있는 시리즈 목록과 도메인 분포.
* 그림: sentinel이 가장 많은 plain 시리즈 3개와 그 쌍둥이를 같은 인덱스 구간으로 나란히 그린 그림 → `reports/figures/sentinels/`.

---

## F4. 서브셋 정의 (`configs/subsets.yaml`)

```yaml
subsets:
  all:          {}
  non_medical:  {domain_goswami_not_in: [ABP, ECG, RESP]}
  physical:     {domain_goswami_in: [Acceleration, Air Temperature, NASA, Power Demand]}
  plain_only:   {variant: plain}
  no_sentinel:  {sentinel_in_train_n: 0, sentinel_in_test_normal_n: 0}   # 민감도 분석용
```

* 서브셋은 조합 가능해야 한다 (예: `physical ∩ plain_only`).
* `reports/data_audit.md` 새 절 §9 Subsets: 각 서브셋과 주요 조합의 `n_series` / `n_groups` 표,
  `n_groups < 30`인 행은 "기술통계 전용" 표시.
* 참고로 현재 추정치: `physical` 42 시리즈 / 19 이름그룹, `non_medical` 100 / 44. 스크립트 출력으로 확정할 것.

---

## F5. 레거시 모듈 경로 정리

* `src/data_ucr.py:80`의 하드코딩 `lo = anom_start - 1`을 `src/data/conventions.py` 경유로 교체.
* 동등성 테스트 추가: 레거시 경로와 새 경로가 250개 전부에서 동일한 `[start0, stop0)`를 내는지. 동일하면
  기존 dry-run(`docs/R5-SCREEN.md`, `experiments/exp_a_artifact.py`) 결과가 유효함이 테스트로 입증된다.
* 불일치가 한 건이라도 있으면 STOP.

---

## F6. 짧은 이상 기록 (분석 없음, 기록만)

* 이상 길이 ≤ 2인 시리즈 수를 도메인별로 `reports/data_audit.md` §4에 추가 (현재 RESP 중앙값 2점).
* 이유: Task B·C에서 섭동 구간 크기(region size)가 이상 길이보다 훨씬 크면 localization 지표 해석이 달라진다.
  지금은 수치만 확보한다.

---

## F7. 테스트 추가

```
tests/test_recording_groups.py
  - 접두사 제거 규칙 (DISTORTED, NOISE, 둘 다 없음)
  - 합성 데이터: 같은 신호+노이즈 → 같은 content_group, 다른 신호 → 다른 그룹
tests/test_sentinels.py
  - exact / near 탐지
  - in_gt / in_train / in_test_normal 분류 경계 (1-based inclusive 규약 반영)
  - 쌍둥이 길이 불일치 시 length_mismatch, 추측 정렬 없음
  - 원본 배열이 수정되지 않음 (함수 호출 전후 해시 동일)
tests/test_subsets.py
  - 조합 서브셋 계산, n_groups 계산
tests/test_legacy_equivalence.py
  - F5 동등성 (250개)
```

---

## STOP 조건 (이 부록 한정 추가)

1. F5 레거시 동등성 불일치.
2. `content_group`이 `name_group`보다 더 크게 묶이는 경우(서로 다른 이름이 같은 녹음)가 발견됨 — 목록 보고 후 대기.
   이 경우 주 분석 단위를 다시 정해야 한다.
3. sentinel이 `in_gt`인데 해당 시리즈의 공식 슬라이드 설명이 −999를 이상으로 언급하지 않는 경우 — 레이블 해석 문제.

---

## 완료 정의

* [ ] F1 세 결정이 `DECISIONS.md`, `conventions.yaml`, `DATA_LICENSE.md`에 반영됨
* [ ] 매니페스트에 F2·F3 열 추가, 250행 전부 채워짐
* [ ] `reports/data_audit.md`에 §7 Recording groups, §8 Sentinels, §9 Subsets 추가 (스크립트 생성)
* [ ] `configs/stats.yaml`, `configs/subsets.yaml` 생성
* [ ] `data/derived/sentinels/` 생성, 원본 불변 테스트 통과
* [ ] `pytest` 전부 통과
* [ ] `VERIFICATION/` 재동기화 (`sync.sh`) — 새 보고서 절과 그림 포함
