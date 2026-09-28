# E5 0단계 — PREREG_W2 정의 목록화 (실행·계산 없음)

> **작성:** 2026-09-28 · **대상:** `docs/PREREG_W2.md` @ `0d041d7`의 예측 ①·②
> **이 문서가 하지 않는 것:** 어떤 지표도 계산하지 않았고, `results/` 아래 값을 열거나 해석 근거로 쓰지 않았다.
> 모호한 항목은 선택지만 나열하며 고르지 않는다. 선택지는 `docs/DECISIONS.md`의 PENDING 섹션 E5-P1 … E5-P15로 옮겼다.
> **판정 표기:** 결정됨 = 문구만으로 구현이 하나로 정해짐 / 모호함 = 문구가 있으나 둘 이상으로 읽힘 / 없음 = 문구 없음.

---

## 1. 문서 무결성

| 확인 | 결과 |
|---|---|
| `git log --follow -- tsad-xai/docs/PREREG_W2.md` | 커밋 1개뿐: `0d041d7` (2026-09-23T13:55:43+09:00) |
| `git diff 0d041d7 HEAD -- tsad-xai/docs/PREREG_W2.md` | 차이 없음 |
| 작업 트리 vs HEAD | 차이 없음 |
| SHA256 (작업 트리 / `git show 0d041d7:…`) | 둘 다 `20aba3f54722efd4e7918c97d1a59b087ea1c4c1e946bf4929129fab29755178` |
| **결론** | **0d041d7 이후 바뀌지 않음.** 멈출 사유 없음. |

**참고(추적성):** PREREG_W2 L3가 근거로 적은 "BRIEF3 §2"의 원문은 저장소에 없다(채팅으로 전달됨; `git ls-files`에
BRIEF3 파일 없음). 따라서 PREREG에 없는 정의(탐지기 인자, r 격자, 위치 규칙 등)는 저장소 안에서는 `docs/DECISIONS.md`의
D-P* 항목과 W2-0 파일럿 코드에만 남아 있다.

---

## 2·3. 정의 원문 인용과 판정

인용은 `docs/PREREG_W2.md`의 줄 번호. "구현 대응"은 해석이 저장소의 어느 구현과 같은지다.

### a. 탐지기 Z와 R (정규화 방식, 참조 구간, m) — **모호함**

| 원문 (줄) | 내용 |
|---|---|
| L13 | "탐지기 Z(z-정규화 MP)에서만 … 탐지기 R(비정규화 MP)에서는 줄지 않는다." |
| L19 | "(게이트 1 P1과 같은 임계값 정의)" |
| 참조 구간 | **없음** |
| m의 정의 | **없음** (L13에 "창 길이 m"이라는 이름만 있음) |

| 해석 | 구현 대응 |
|---|---|
| Z-1: AB-join, T_A = x 전체, T_B = x[:train_end], ignore_trivial=False | 파일럿 `src/detectors/mp.py:44` (`MPScorer("Z")`), rev1 `MatrixProfileDetector` ab_join·`score_patch` (`src/detectors/matrixprofile.py:180`); D-P2-1 (DECISIONS L669) |
| Z-2: AB-join, T_A = x[train_end:] (테스트 구간만) | BRIEF3 P2의 문구(저장소 밖). 파일럿은 D-P2-1로 Z-1을 택함 |
| Z-3: self-join (게이트 1 `fit_on=full` 참조 조건) | `matrixprofile.py` self_join 모드 |
| R-1: `stumpy.aamp`, AB-join, T_A = x, T_B = x[:train_end], ignore_trivial=False, p = 2 | 파일럿 `src/detectors/mp.py:46` (`MPScorer("R")`) |
| R-2: `stumpy.aamp` self-join | 구현 없음 |
| m-1: 게이트 1 윈도우 (`find_length_rank`, rank = 1, 시리즈별) | 파일럿(`w2_pilot.py`가 `benchmark_window`로 재확인)과 rev1(`w`) 모두 이것 |
| m-2: 다른 정의 | 구현 없음 |

### b. 지우개 — **상수 계열 목록: 결정됨 / 개별 정의의 적용 공간: 모호함 / ②의 비교 집합: 없음**

| 원문 (줄) | 내용 |
|---|---|
| L13 | "상수 계열 지우개(Zero, SampleMean, OutOfDistHigh)" |
| L14 | "복원 지우개(ContextReconstruct)가 가장 작은 자국을 남긴다." |
| L28–32 | ContextReconstruct: "지운 구간 `[a, a+r)` 바로 앞 m점 `x[a-m:a]`를 쿼리로, 학습 구간에서 z-정규화 거리 최근접 위치 j를 찾음 (단 `j+m+r ≤ train_end`)" / "채우는 값: `x_train[j+m : j+m+r] + (x[a-1] - x_train[j+m-1])` (왼쪽 이음매 오프셋 정렬)" / "지울 구간 자체는 참조하지 않음" |
| L34–35 | "`NearestNeighborWindow`(Šimić 원 정의, 위치 기준 좌·우 이웃 반씩 이어 붙임)는 원본 정의 그대로 유지하며 보고서에서는 \"이웃 붙이기\"로 표기한다." |
| L45 | "(그때 쓴 채움 값은 원시 스케일이었고, 확정된 표준화 공간 정의와 다르다)" — 표준화 공간을 언급하지만 정의는 없음 |
| ②에서 CR과 비교할 지우개 집합 | **없음** |

| 해석 | 구현 대응 |
|---|---|
| 공간-1: 학습 구간 μ/σ로 표준화한 공간에서 Šimić 정의 적용 후 역변환 | D-P3-1 (DECISIONS L681), 파일럿 `operators_simic.erase(..., Space(True, "train"))` (`src/perturb/operators_simic.py:223`) |
| 공간-2: 원시 스케일 | PREREG L42–45의 scratch 실측이 쓴 방식. `Space(False, ...)` |
| 공간-3: 시리즈 전체 μ/σ | `Space(True, "full")` |
| 비교 집합-1: 파일럿 지우개 8종 (Šimić 7 + CR) | `operators_simic.OPERATORS` (L195), `configs/w2_pilot.yaml` L15–16 |
| 비교 집합-2: CR과 상수 계열 3종만 | 구현 부분 집합 |
| 비교 집합-3: 비교 집합-1 + rev1 연산자(B1–B6, recon_test, recon_train) | `src/perturb/operators.py` (별도 모듈, 인터페이스 다름) |
| CR 세부: j 동률, 매치 불가(`train_end < m + r + 1`) | PREREG에 없음. D-P3-2 (L708)는 동률 → 가장 작은 j, 매치 불가 → 예외 |

### c. 지우는 길이 r — **없음**

| 원문 (줄) | 내용 |
|---|---|
| L13 | "지운 길이 r이 창 길이 m보다 커질수록" — r의 값 목록 없음 |
| L42–43 | 공개 사항(파일럿 전 scratch)에 "r ∈ {13, 26, 52, 104}"가 나오지만 #066(m = 26) 한 시리즈의 실측 기록이지 정의가 아님 |

| 해석 | 구현 대응 |
|---|---|
| r-1: {⌊0.5m⌋, m, 2m, 4m} | 파일럿 `configs/w2_pilot.yaml` L24 `r_over_m: [0.5, 1, 2, 4]`, `w2_pilot.py:208` (floor) |
| r-2: rev1 격자 {0.25, 0.5, 1.0}w(반올림, 최소 1)에 {2w, 4w} 추가 | rev1 `src/w2.length_for`; 2w·4w는 rev1에 없음 |
| r-3: {m, 2m, 4m}만 (r < m이면 내부 창이 없음) | 부분 집합 |

### d. 주 지표 "가짜 경보" — **임계값: 결정됨 / 영향받은 창의 범위·탐지기별 임계값 적용·"율"의 집계: 모호함**

| 원문 (줄) | 내용 |
|---|---|
| L19 | "주 지표: 가짜 경보 여부 = 지운 뒤 영향받은 창의 최대 점수 > 원래 정상 테스트 점수의 99 백분위수 (게이트 1 P1과 같은 임계값 정의)." |
| L20 | "보조 지표: ΔS(영향받은 창 점수 평균 변화, 내부/경계 창 분리)." |
| "율"(rate) | **없음** — L19는 한 번의 지우기에 대한 이진값("여부")만 정의 |

임계값은 "게이트 1 P1과 같은 임계값 정의"로 정해진다고 읽을 수 있다: 게이트 1 판정은 buffer = w를 쓴다(D-C2-1, DECISIONS L443;
`src/gate1.py` normal_test_mask, `PERCENTILE = 99.0` L29). 그러나 아래는 문구로 정해지지 않는다.

| 해석 | 구현 대응 |
|---|---|
| 창-1: 영향받은 창 = 시작점 [a−m+1, a+r−1] (내부 + 경계), 그 최대 | 파일럿 `mp.affected_range` (`src/detectors/mp.py:68`), `w2_pilot.py:239`; rev1 J(R)와 같은 집합 |
| 창-2: 내부 창만 | 파일럿 `classify_windows` (L75)의 interior |
| 임계값-1: 탐지기마다 자기 무교란 점수(패딩 포함)의 P99, buffer = m | 파일럿 `p99_threshold` (`w2_pilot.py:201`, :533) |
| 임계값-2: buffer = 0 | 게이트 1 `thr_b0` (보고용 변형) |
| 율-1: 시리즈별 (위치 × r) 중 경보 비율 → 그룹 집계 | 구현 없음 |
| 율-2: 전체 위치를 합쳐 비율 | 구현 없음 |
| 사전 경보 처리: 지우기 전 이미 임계값을 넘는 위치를 제외 / 포함 | 파일럿은 `false_alarm_before`를 따로 기록 (`w2_pilot.py:240`) |

### e. 예측 ①의 판정 규칙 — **모호함 (규칙 자체는 없음)**

| 원문 (줄) | 내용 |
|---|---|
| L13 | "지운 길이 r이 창 길이 m보다 커질수록, 탐지기 Z(z-정규화 MP)에서만 상수 계열 지우개(Zero, SampleMean, OutOfDistHigh) 간 자국 차이가 줄어든다. 탐지기 R(비정규화 MP)에서는 줄지 않는다." |
| L24 | "content_group 89, Wilcoxon + Holm." |
| "자국 차이"의 지표, 비교 대상, 임계값, 유의수준 | **없음** |

| 해석 | 구현 대응 |
|---|---|
| 지표-1: 세 지우개의 **내부 창 점수** 차이(범위 또는 쌍별 차) | 파일럿 열 `interior_min/max/n_unique`, S2·S3 검사 (`w2_pilot.py:258` `sanity`); rev1 D12-a와 같은 기제 |
| 지표-2: 세 지우개의 **ΔS**(내부/경계) 쌍별 차 | 파일럿 열 `dS_interior_mean`, `dS_boundary_mean` |
| 지표-3: 세 지우개 사이 **가짜 경보 불일치** 비율 | 파일럿 열 `false_alarm` |
| 비교-1: r ≤ m 대 r > m (예: 0.5m 대 4m) | 구현 없음 |
| 비교-2: r에 따른 단조 추세 | 구현 없음 |
| 대비: Z와 R 각각 판정 후 "Z에서만 감소 ∧ R에서 비감소"를 모두 요구 | 구현 없음 |
| 검정: Wilcoxon signed-rank(그룹 단위, L24), Holm; α와 "줄지 않는다"의 판정(비유의? 동등성?)은 없음 | rev1 `w2_e3_report.holm` 등 도구만 있음 |

### f. 예측 ②의 판정 규칙 — **모호함 (규칙 자체는 없음)**

| 원문 (줄) | 내용 |
|---|---|
| L14 | "복원 지우개(ContextReconstruct)가 가장 작은 자국을 남긴다." |
| 지표·집계·동률·탐지기 | **없음** (L19–20의 주/보조 지표 중 어느 것인지도 명시 없음) |

| 해석 | 구현 대응 |
|---|---|
| 지표-1: 가짜 경보율(주 지표)이 가장 낮음 | 파일럿 `false_alarm` |
| 지표-2: \|ΔS\|(크기)가 가장 작음 | 파일럿 `dS_*`의 절댓값 |
| 지표-3: 부호 있는 ΔS가 가장 작음 | 파일럿 `dS_*` |
| 지표-4: `max_after − p99_normal`이 가장 작음 | 파일럿 열 `max_after`, `p99_normal` |
| 집계: 그룹 평균 / 그룹 중앙값 | 구현 없음 |
| 판정: CR 대 다른 지우개 각각 Wilcoxon + Holm에서 모두 유의하게 작음 / 순위 1위이기만 하면 됨 | 구현 없음 |
| 동률: 공동 1위 인정 / 불인정 / 보조 지표로 깸 | 구현 없음 |
| 탐지기: Z와 R 둘 다 / 각각 따로 판정 | 구현 없음 |

### g. 표본 — **모호함**

| 원문 (줄) | 내용 |
|---|---|
| L24 | "content_group 89" |

| 해석 | 구현 대응 |
|---|---|
| 표본-1: rev1 표본 (그룹당 1개, 89개 시리즈) | `configs/w2_sample.yaml` (`scripts/w2_sample.py`, D-S-1) |
| 표본-2: 250개 전체를 89그룹으로 집계 | 구현 없음 (rev1 §5는 250 전수를 W3 이후로 둠) |
| 표본-3: 파일럿식 규칙 선택 부분집합 | `w2_pilot.select_series` (주기 도메인·정상 구간 ≥ 8m) |

### h. 구간 위치 규칙과 시드 — **없음**

| 해석 | 구현 대응 |
|---|---|
| 위치-1: 파일럿 규칙 — 확실한 정상 조각 하나 안에 "영향 창 + 모든 지우개의 읽기 범위"(footprint)가 들어가고 위치끼리 겹치지 않음, r 전체가 같은 a 공유 | D-P5-1 (DECISIONS L731), `w2_pilot.draw_positions` (L176), `READ_EXTENT` (`operators_simic.py:209`); 파일럿 시드 20260923, 위치 3개 |
| 위치-2: rev1 E2a 규칙 — I(R) ⊂ 테스트 구간, I(R) ∩ E = ∅, 길이별 20개 비겹침, D-E2a-1 부족 규칙 | `src/w2.admissible_starts` (L80), `draw_disjoint` (L136), `scripts/w2_e2.regions_for` (L61); 시드 20260924 |
| 위치-3: 위치-2에 지우개 읽기 범위(NNW ⌈r/2⌉, CR m) 조건을 더함 | 부분 구현(위 두 함수 조합) |
| 위치 수: 3(파일럿) / 10(BRIEF3 W2 외삽의 K) / 20(rev1) | `w2_pilot.yaml` L25·L36, `w2.yaml` e2_grid |

### i. 통계 단위와 집계 — **단위: 결정됨 / 집계와 Holm 범위: 모호함**

| 원문 (줄) | 내용 |
|---|---|
| L24 | "content_group 89, Wilcoxon + Holm." |

| 해석 | 구현 대응 |
|---|---|
| 그룹 내 집계: 위치의 평균 / 중앙값 | rev1 E2 보고는 "시리즈별 중앙값의 시리즈 간 중앙값" (D-E2-2) |
| Holm 범위: 예측별 / 탐지기별 / 예측 전체 한 묶음 | 구현 없음 |

### j. STOP·판정 불가 조건 — **없음**

| 해석 | 구현 대응 |
|---|---|
| STOP-1: 파일럿 sanity S1–S4를 전 표본에서 검사 (S4는 rev1 허용오차 1e-6) | `w2_pilot.sanity`; D-P4-1, D-E0-3 |
| STOP-2: rev1 §11식 조건 — CR 매치 불가 비율, 위치 부족, scale 제외 비율 | rev1 `w2_e2.py`의 STOP 구조 |
| STOP-3: 정하지 않음(보고만) | — |
| 증분 계산 정확도(R): 허용오차 미정 (파일럿 S4는 Z·R 1e-8, rev1 E0는 Z만 1e-6) | `tests/test_mp_incremental.py` (Z·R 모두 1e-6) |

---

## 4. 구현 재사용 점검

| 구성 요소 | 이미 있음 (위치) | 새로 필요 |
|---|---|---|
| 탐지기 Z 전체 계산 | `mp.MPScorer("Z").profile` (`src/detectors/mp.py:98`); 게이트 1 저장 점수 재사용 가능 (`src/w2.gate1_score`) | — |
| 탐지기 R 전체 계산 | `mp.MPScorer("R").profile` | **R의 무교란 전체 프로파일과 P99 임계값을 시리즈마다 계산** (게이트 1에는 Z 점수만 저장됨). 긴 시리즈의 계산 시간 예산 |
| Z·R 증분 계산 | `mp.MPScorer.incremental` (L109) — Z·R 공용, 합성 데이터에서 8개 지우개 × Z/R 증분 == 전체 (`tests/test_mp_incremental.py`, atol 1e-6) | **실데이터에서 R 증분 정확도 점검**(rev1 E0는 Z만) |
| 영향 창·내부/경계 분류 | `mp.affected_range` (L68), `classify_windows` (L75) | — |
| Šimić 지우개 7종 + CR | `src/perturb/operators_simic.py` (원본과 비트 일치 테스트 `tests/test_simic_operators.py`) | CR 매치 불가 시 처리 규칙(E5-P15) |
| r = 2m, 4m | 코드상 제한 없음(파일럿이 4m까지 실행). CR은 `train_end ≥ m + r + 1` 필요 | 89개 시리즈에서 4m 위치 확보·CR 가능 여부 점검 |
| 위치 규칙 | 파일럿 `draw_positions`(단일 시리즈), rev1 `w2.admissible_starts`/`draw_disjoint` | E5-P12 결정에 맞춘 다중 시리즈 위치 생성 |
| 지표 계산 | 파일럿 `run_grid` (L205–256): false_alarm, ΔS 내부/경계, max_after, p99 | 가짜 경보"율" 집계(E5-P8), ①·② 판정 통계(E5-P9, P10) |
| 실행기 | 파일럿은 단일 시리즈; rev1 `w2_e2.py`의 4워커 병렬 구조, `check_decisions` | 다중 시리즈 실행 스크립트, 설정 파일, 보고서 생성기 |

**탐지기 R의 stumpy 함수 (설치본 1.14.1 기준):**

| 항목 | 소스 |
|---|---|
| 함수 | `stumpy.aamp(T_A, m, T_B=None, ignore_trivial=True, p=2.0, k=1)` — `stumpy/aamp.py:334` |
| 파일럿 호출 | `stumpy.aamp(t_a, m, T_B=t_b, ignore_trivial=False)` — `src/detectors/mp.py:46` (p = 2 기본값 → 유클리드) |
| AB-join 분기 | `aamp.py:421-422` (`diags = np.arange(-(n_A - m + 1) + 1, n_B - m + 1)`), 제외 구역 없음 |
| 거리 누적 | 대각선 첫 칸은 직접 `np.linalg.norm(...)**p` (`aamp.py:122-130`), 이후 칸은 점화식 가감 (`:131-141`) — `_compute_diagonal`, `fastmath` (`:16`) |
| 0 근처 처리 | `p_norm < config.STUMPY_P_NORM_THRESHOLD` (1e-14, `config.py:15`)이면 0 (`aamp.py:143-144`) |
| 최종 거리 | `np.power(P, 1.0 / p)` (`aamp.py:325`) |
| 상수 창 | 별도 분기 없음 (Z의 `stump.py:201-204`와 다름) |
| 전처리 | `core.preprocess_non_normalized` (`aamp.py:402-403`) — z-정규화 없음 |

---

## 5. 파일럿 노출 기록 (목록만; 파일을 다시 열지 않음)

| 파일 | 조건 | ①·②와의 관련 |
|---|---|---|
| `reports/w2_pilot.csv` | #066, 위치 3개, r ∈ {⌊0.5m⌋, m, 2m, 4m}, 지우개 8종(상수 계열 3종·CR 포함) × Z/R; 열 `false_alarm`, `false_alarm_before`, `dS_interior_mean`, `dS_boundary_mean`, `max_after`, `p99_normal`, `interior_min/max/n_unique` | ①·② 모두 (PREREG L19–20과 같은 지표군) |
| `reports/w2_pilot.md` | §3 S2(Z 상수 계열 내부 창), S3(R 상수 계열 내부 창), S5; §4 지우개 × r × 탐지기 조건별 요약표 | ①·② 모두 |
| `reports/figures/w2_pilot/F2_*` | #066, 위치 #0, r = 2m, Zero와 CR, Z·R 점수 곡선 | ② |
| `docs/PREREG_W2.md` L39–46 | 커밋 전 scratch(#066, 원시 스케일 채움) | ① (공개 사항으로 이미 기록됨) |
| `docs/W2_REPORT.md` §2 | F2 설명에 파일럿 Z 점수 서술 포함 (`621afd4`부터, 원격에 push됨) | ② |
| rev1 E2 (`results/w2/ai.csv.gz`, `reports/w2_artifact.md`) | PREREG의 지우개가 아님(B1–B6, recon). Z만. 상수 계열 B1–B3·D12-a는 ①의 Z 쪽 기제와 같은 성질 | ① (간접), ② (간접) |

- **Claude Code:** 위 파일럿 값을 모두 생성하고 읽었다(D-PRE-1).
- **연구자:** D-PRE-1(2026-09-24)에 "[보지 않음]"으로 기록됨. 그 뒤 `docs/W2_REPORT.md` §2(랩미팅용, push됨)에 파일럿 수치 일부가
  들어갔으므로, 현재 열람 여부는 연구자가 다시 확인해 기록할 필요가 있다.

---

## 6. PENDING으로 옮긴 선택지

`docs/DECISIONS.md` "PENDING — E5 (PREREG_W2 예측 ①·② 판정 정의)"의 E5-P1 … E5-P15. 승인 표기 없음.
