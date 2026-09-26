# W2 수행 보고서 — 랩미팅용

> **작성:** 2026-09-26 (E3 정렬 규칙 확정·E4 반영 갱신) · **대상 기간:** 2026-09-23 ~ 09-26
> **범위:** BRIEF3(W2-0 단일 시리즈 파일럿) + `docs/briefs/BRIEF_W2_perturbation-experiments_rev1.md`
> (세션 1: E0·표본·E1·게이트 2 / 세션 2: 결정 커밋·E2·E3 / D-E3-5 정렬 규칙 확정 후 E3 곡선 재계산·E4).
> **저장소:** `github.com/hyun-28/XAD` — 수치의 출처 커밋은 각 보고서 머리글에 있다.
> **E3 버전:** 본문은 **무작위 동률 버전(D-E3-5, 주 결과)**이다. 인덱스 동률 버전은 규칙 확정 전에 생성된 기록으로
> 부록 A에 둔다.
>
> **이 문서의 지위:** 미팅용 서술 요약이다. **모든 수치는 스크립트 산출물에서 옮겨 적었고, 출처 파일을
> 괄호로 표시했다.** 수치의 근거는 이 문서가 아니라 그 파일이다(CLAUDE.md §1-4). 그래서 `reports/`가 아니라
> `docs/`에 둔다. 해석은 하지 않고, 논의가 필요한 지점은 질문으로 적었다(§7).

---

## 0. 한 장 요약

| 단계 | 결과 |
|---|---|
| W2-0 파일럿 (#066, Šimić 지우개) | 그림 F2 생성. S1–S3 통과, **S4 불일치(4.0e-7 > 1e-8)로 STOP** → rev1에서 기준 1e-6으로 변경 |
| E0 — 증분 점수 `score_patch` | **통과.** 400구간·70,652창에서 max\|Δ\| = 1.0e-11 (기준 1e-6), 항등 연산자 AI 400/400 정확히 0 |
| 게이트 2 | **통과.** E2 예상 0.20 h(89그룹) ≤ 2 h → rev1 §5에 따라 **89그룹 전수** |
| 결정 커밋 | `f29eddf` (2026-09-24 20:30) — D10–D13·D-E3-1(rev) 승인, D12 tol, 89그룹. 이후에만 E2·E3 실행 |
| E2 (C1: Artifact Index, SAR) | STOP 없음. **D12 예측 4종 전부 100% 일치.** B1–B6는 정상 구간에서 AI 중앙값 0.1–2.5, 재구성형은 0.002–0.023 |
| E3 (C2: AM 순위) | 89/89 시리즈. 정렬 규칙 D-E3-5(무작위 동률) 확정 후 곡선 재계산. **6개 평가 연산자 간 Kendall's W = 0.233** (95% CI [0.044, 0.556]) |
| 결정 커밋 2 | `b8b783c` (2026-09-26 23:03) — D-E3-5(정렬·동률), D-E4-1/2/3, 노출 기록 D-PRE-2 |
| E4 (C3 축소: plausibility) | 주 지표 정규화 RRA로 AM 순위: KernelSHAP 1 · MPNative 2 · FeatureAblation 3 · Random 4. E3 연산자별 순위와 Kendall τ −0.548 ~ 0.667 |

---

## 1. W2의 흐름

1. **W2-0 파일럿** (BRIEF3, 09-23) — 시리즈 1개에서 그림 F2, 탐지기 Z/R, 증분 재계산을 점검.
2. **rev1 세션 1** (09-24) — `score_patch`(E0), 연산자 9종(E1), 89그룹 표본, 게이트 2.
3. **결정 전용 커밋** `f29eddf` — 연구자 승인 사항만 담았다. E2·E3 스크립트는 이 커밋이 HEAD의 조상인지
   확인한 뒤에만 실행된다.
4. **E2** (09-24, 12분) → **E3** (09-25~26, 4.18 h; attribution 계산 포함).
5. **결정 커밋 2** `b8b783c` (09-26) — 충실성 정렬 규칙 D-E3-5와 E4 정의 D-E4-1/2/3. 이 규칙은 인덱스 동률의
   위치 편향 때문에 도입됐고, 인덱스 버전 E3 값을 Claude Code가 이미 본 뒤였다는 사실을 D-PRE-2에 기록했다
   (연구자 열람 여부: 보지 않음).
6. **E3 곡선 재계산** (16.8분; attribution은 재사용) → **E4** (plausibility).

---

## 2. W2-0 파일럿과 그림 F2 (`reports/w2_pilot.md`)

- 시리즈 선택은 **규칙으로** 했다(게이트 1 통과 · 주기 도메인 · 정상 구간 ≥ 8m · 가장 짧은 것) →
  **#066 DISTORTEDinsectEPG2** (n = 9,998, m = 26). 규칙 3을 네 가지로 해석해도 같은 시리즈가 뽑혔다.
- **F2** (`reports/figures/w2_pilot/F2_zoom.png`): 정상 구간 [4126, 4178)을 Zero와 복원(ContextReconstruct)으로 지운 뒤
  탐지기 Z·R 점수. Zero는 Z에서 √26 = 5.099까지 튄다(임계값 0.80). 복원은 Z 점수를 0 근처로 떨어뜨린다
  (복원된 창이 학습 조각의 오프셋 복사본이라 z-정규화 거리가 수학적으로 0).
- sanity check: S1(영향 없는 창) · S2(Z에서 상수 지우개 3종이 √m으로 동일) · S3 통과,
  **S4(증분 == 전체, atol 1e-8) 5/192건 불일치 → STOP.** 원인: 참 거리 0 근처에서 √(2m(1−ρ))가 반올림을 증폭
  (D² 기준 차이는 1.6e-13). rev1이 E0 기준을 1e-6으로 바꾸면서 해소.
- 파일럿 수치는 연구 결과로 쓰지 않는다(BRIEF3 §0). 이 파일럿을 Claude Code가 봤다는 사실은 DECISIONS D-PRE-1에 기록
  (연구자 열람 여부: 보지 않음).

---

## 3. E0 · 게이트 2 (`reports/w2_gate2.md`)

- **`score_patch`:** 구간 R이 닿는 창 J(R)만 학습 구간에 대해 다시 계산. 무작위 시리즈 20개 × 구간 20개에
  가우시안 잡음을 넣고 전체 재계산과 비교 → **max\|Δ\| = 1.01e-11**, 중앙값 1.4e-14, 9.6%는 정확히 0.
- **항등 연산자:** 400/400 비트 일치, AI 정확히 0.0.
- **stumpy 소스 추적** (1.14.1, 실행 시 설치본과 줄 대조):
  (a) 상수 판정은 `ptp == 0` 정확 비교, 허용오차 없음 (`core.py:2614`)
  (b) 상관 ≈ 1을 거리 0으로 맞추는 임계값 없음, `min(1.0, pearson)`만 (`stump.py:207`)
  (c) 한쪽 상수 → ρ = 0.5 대입 → D = √m, 양쪽 상수 → D = 0 (`stump.py:201-204`)
- **D12 tol (승인):** a/a′ = 1e-12, b = 1e-3 (근거: 게이트 1 자기일치 잔차 최대 3.3e-4, #249).
- **E2 예상 시간:** 60그룹 0.14 h, 89그룹 0.20 h → 89그룹 전수로 결정.

---

## 4. 표본 (`configs/w2_sample.yaml`, `reports/w2_gate2.md` §4)

- 단위는 `content_group` 89개, 그룹당 1개 시리즈(시드 고정 추출). 결과를 보기 전에 커밋.
- 도메인: ECG 51 · ABP 14 · Gait 10 · EPG 5 · Power Demand 3 · Acceleration 2 · RESP 2 · Air Temperature 1 · NASA 1.
- 게이트 1 탐지 성공(`detected`) 65, 실패 24 — 실패 시리즈도 포함. 학습 구간에 상수 창이 있는 시리즈 5.

---

## 5. E2 결과 — C1: 마스킹이 정상 구간에서 점수를 얼마나 움직이나 (`reports/w2_artifact.md`)

**정의:** AI = (resp(x′, R) − resp(x, R)) / scale(x), resp = R에 닿는 창의 최대 점수, 부호 유지.
64,170행 (89시리즈 × 구간 ~81 × 연산자 9).

### 5.1 정상 구간 AI (시리즈별 중앙값의 시리즈 간 중앙값)과 SAR

| 연산자 | 0.25w | 0.5w | 1.0w | 이상 구간 reversal | SAR 중앙값 | SAR_abs 중앙값 |
|---|---|---|---|---|---|---|
| B1_zero | 0.820 | 1.420 | 2.282 | 83.1% | 0.000 | 0.403 |
| B2_global_mean | 0.495 | 1.239 | 2.292 | 77.5% | 0.000 | 0.338 |
| B3_local_mean | 0.438 | 1.101 | 2.291 | 77.5% | 0.000 | 0.337 |
| B4_linear_interp | 0.118 | 0.878 | 1.569 | 56.2% | 0.000 | 0.362 |
| B5_gaussian | 1.240 | 2.055 | 2.515 | 88.8% | 0.000 | 0.428 |
| B6_shuffle | 0.292 | 1.429 | 2.394 | 87.6% | 0.000 | 0.378 |
| recon_test | 0.002 | 0.006 | 0.023 | 19.1% | 7.536 | 7.765 |
| recon_train | 0.003 | 0.005 | 0.012 | 15.7% | 4.948 | 5.305 |
| identity | 0 | 0 | 0 | 0% | — | — |

- reversal = 이상 구간을 지웠더니 점수가 **올라간** 비율. SAR(D10)은 이 경우 제거 반응을 0으로 두므로
  B1–B6의 SAR 중앙값이 0이 되고, 보조 지표 SAR_abs와 갈린다.
- 플래그 시리즈 10개를 뺀 같은 표는 `reports/w2_artifact.md` §2, §4에 있다.

### 5.2 사전 등록 예측 D12 (전부 일치)

| 예측 | 사례 | 일치율 |
|---|---|---|
| D12-a (i) 상수 지우개, \|R\| = w → s(j*) = √w | 4,902 | 100% |
| D12-a (ii) resp_after ≥ √w | 4,902 | 100% |
| D12-a′ 학습 구간에 상수 창 있음 → s(j*) = 0 | 300 | 100% |
| D12-b recon_train, \|R\| = w → s(j*) ≤ 1e-3 | 1,734 | 100% (실제 잔차 최대 1.75e-5) |

### 5.3 recon_test 대 recon_train (순환성 대조)

정상 구간 AI 중앙값은 두 연산자가 비슷하고(1.0w: 0.020 대 0.012), 같은 위치 짝 비교에서 recon_train이 더 작은
비율은 48–50%. 이상 구간 AI 중앙값 −0.821 대 −0.781, SAR 중앙값 7.536 대 4.948.

---

## 6. E3 결과 — C2: 평가 연산자를 바꾸면 설명기 순위가 바뀌나 (`reports/w2_ranking.md`, 무작위 동률 버전)

**설정:** 설명용 연산자 recon_test, 평가용 B1–B6(identity 제외, recon_train은 별도), AM 4종
(Random · FeatureAblation · KernelSHAP[S = 2K+2048] · MPNative). 지표는 Šimić 저장소를 소스 추적한 그대로
**CMI(평균 DDS, PES)**로 순위. 분류에서는 Šimić et al. 2025가 이미 보인 현상이며, 새로움은 주장하지 않는다.
**정렬(D-E3-5):** 부호 유지 내림차순, 동률은 (시리즈, AM) seed 무작위 순열, LeRF = MoRF의 정확한 역순.
attribution은 E3 저장본을 그대로 쓰고 곡선만 다시 계산했다. 동률 없는 AM(Random, KernelSHAP)의 곡선은
재계산 전후가 비트 단위로 같음을 실행 중 검사했다.

### 6.1 평가 연산자별 AM 순위 (1 = 최고, CMI 기준)

| 평가 연산자 | Random | FeatureAblation | KernelSHAP | MPNative |
|---|---|---|---|---|
| B1_zero | 1 | 3.5 | 3.5 | 2 |
| B2_global_mean | 4 | 2 | 3 | 1 |
| B3_local_mean | 4 | 1 | 3 | 2 |
| B4_linear_interp | 4 | 2 | 1 | 3 |
| B5_gaussian | 2 | 3.5 | 3.5 | 1 |
| B6_shuffle | 4 | 3 | 2 | 1 |

(그림: `reports/figures/w2/rank_bump.png`)

### 6.2 일치도

- **Kendall's W = 0.233** (content_group 부트스트랩 95% CI [0.044, 0.556], B = 1000).
- 연산자 쌍 Spearman ρ: −0.949 ~ 0.800 (B1–B4 = −0.949, B1–B5 = 0.778, B2–B3 = B2–B6 = 0.800). CI는 대부분 매우 넓다.
  AM이 4개라 ρ는 11개 값만 가질 수 있다(해상도 한계).
- 보조: 시리즈별 W 중앙값 0.411 (IQR [0.222, 0.578]).
- **자기 일치 민감도** (설명용 = 평가용으로 FA·KernelSHAP 재계산): 주 순위와의 ρ는
  B1 −0.738 · B2 0.2 · B3 0.4 · B4 1.0 · B5 −0.738 · B6 0.4.
- **|R| 정렬 민감도** (FA·KernelSHAP): 부호 유지 순위와의 ρ는 B1 1.0 · B2 0.8 · B3 0.8 · B4 1.0 · B5 0.316 · B6 0.8.
- **동률 진단** (동률 세그먼트 수 / K): FeatureAblation 중앙값 0.524 (동률 있는 설명 96.6%), MPNative 0.667 (100%),
  KernelSHAP·Random 0.
- 인덱스 동률 버전과의 대조표(연산자별 순위, CMI, ρ, W)는 `reports/w2_ranking.md` §6과 부록 A.

---

## 7. E4 결과 — C3 축소: plausibility (`reports/w2_plausibility.md`)

**정의(D-E4-1/2/3):** E3 주 조건 attribution을 세그먼트에서 ReLU(민감도 |R|)한 뒤 점으로 균등 투영(질량 보존).
Arras et al.(arXiv:2003.07258 §3.5)의 Relevance Mass/Rank Accuracy를 Ω 위에서 계산하고,
(값 − 우연) / (oracle − 우연)으로 정규화. 우연 = |GT|/|Ω|, oracle = 각 세그먼트에 GT 겹침 비율을 준 설명.
주 지표는 정규화 RRA. faithfulness와 구분해 **plausibility**로 부르며 검정하지 않는다.

| AM | 정규화 RRA 평균 (주) | 순위 | 정규화 RMA 평균 (민감도) | 순위 | 정규화 RRA, \|R\| (민감도) | 순위 |
|---|---|---|---|---|---|---|
| Random | −0.007 | 4 | −0.030 | 4 | −0.007 | 4 |
| FeatureAblation | 0.225 | 3 | 0.530 | 1 | 0.354 | 1 |
| KernelSHAP | 0.344 | 1 | 0.453 | 3 | 0.342 | 2 |
| MPNative | 0.273 | 2 | 0.498 | 2 | 0.273 | 3 |

- 정규화 값 NaN으로 제외된 시리즈: AM마다 4개(oracle − 우연 < 1e-6), 정규화 RMA는 0개.
- **E4 주 순위와 E3 순위의 Kendall τ:** B1 −0.548 · B2 0.333 · B3 0.000 · B4 0.667 · B5 −0.183 · B6 0.667.
  AM이 4개라 τ는 동률이 없을 때 7개 값만 가진다.
- **그룹 단위 τ** (시리즈별 E3 DDS 순위 vs E4 순위): 연산자별 중앙값 −0.183 ~ 0.183.
- **sanity:** Random의 정규화 RRA 중앙값 −0.043, IQR [−0.140, 0.173].
- 진단: 음수 질량 비율 Σ|R⁻|/Σ|R| 중앙값 FeatureAblation 0.296, KernelSHAP 0.120 (Random·MPNative 0).
  GT가 세그먼트 길이 g보다 짧은 시리즈 8개.

---

## 8. 논의가 필요한 지점 (사실만, 해석하지 않음)

1. **reversal이 많다.** B1–B6로 이상 구간을 지우면 56–89% 시리즈에서 점수가 오히려 오른다. 그래서 D10 SAR은
   중앙값 0, SAR_abs는 0.34–0.43. **질문:** 논문에서 C1의 주 지표를 SAR로 둘지, SAR_abs를 병기할지.
2. **B1에서 FA·KernelSHAP의 CMI가 정확히 0** (평균 DDS 음수, PES 양수 → CMI 정의상 0) → 3.5위 동률.
3. **CMI가 1을 넘는 경우** (recon_train 평가 시 KernelSHAP 1.298) — DDS가 이상 점수 척도라 [−1, 1]에 묶이지 않음(D8).
4. **B3의 50점 문맥이 다른 마스킹 run과 겹친 비율 64.0%** (E3 곡선 재계산, 45,404건 중). 승인 조건대로 한계로 보고.
5. **sar_den_zero가 #239·#240에 몰림** — 학습 구간 상수 창이 13.6만·17.2만 개인 시리즈(taichi).
6. **탐지 성공 시리즈의 AI가 더 크다** (예: B1 2.13 대 0.76; `reports/w2_artifact.md` §5).
7. **동률 처리 규칙만 바꿨는데 E3 순위가 크게 달라졌다** — W 0.069(인덱스) → 0.233(무작위), MPNative의 CMI가
   B2에서 0.055 → 0.373 등. CMI 값이 바뀐 AM은 동률이 많은 FeatureAblation·MPNative뿐이고, Random·KernelSHAP은
   CMI가 같고 순위만 움직였다(`reports/w2_ranking.md` §6, 부록 A).
8. **정규화 RMA가 1을 넘는 시리즈가 있다** (FeatureAblation IQR 상단 1.037) — oracle은 겹침 비율을 준 설명이지
   가능한 최댓값이 아니다.
9. **FeatureAblation의 반올림 수준 양수** — #202는 양수 세그먼트 3개의 최댓값이 1.47e-14라 ReLU 질량이 0이 아니고
   RMA = 0.0으로 계산된다(NaN 아님). 미세값을 0으로 볼 임계값은 정하지 않았다.
10. **E4 순위는 지표에 따라 바뀐다** — 정규화 RRA에서는 KernelSHAP 1위, 정규화 RMA와 |R| RRA에서는 FeatureAblation 1위.

---

## 9. 작업 중 있었던 일

| 일 | 처리 |
|---|---|
| 파일럿 STOP 2건 (S1 "정확히 0" 불성립, Šimić 지우개의 "샘플" 정의) | 연구자 결정: S1 atol 1e-8, 학습 구간 μ/σ 표준화 공간에서 지우개 적용 |
| 파일럿 S4 불일치 (Z × 복원, 4.0e-7) | rev1 E0 기준 1e-6으로 해소 (DECISIONS D-E0-3) |
| 참조 env에 parquet 엔진 없음 | 결과를 `.csv.gz`로 저장 (D-E2-1) |
| E3 첫 실행 메모리 폭주 (16 GB 중 15 GB, 스왑) | 6분 만에 중단, donor 캐시를 Ω 띠로 제한 → 고르는 donor는 동일, 워커 최대 0.74 GB (D-E3-4) |
| E3 소요 | 예산 추정 3.65 h, 실제 4.18 h (8 h 한도 내) |
| E3 동률 규칙 변경 (인덱스 → 무작위) | 결정 커밋 `b8b783c` 후 곡선만 재계산(16.8분). 인덱스 버전은 보존(부록 A). 노출 기록 D-PRE-2 |
| 결정 ID 충돌 (연구자 지시 "D-E3-2"가 기존 항목과 겹침) | D-E3-5로 기록 |
| DECISIONS에 D-C5-4·D-C5-5 항목 누락 | W1 결정으로 사후 이관 (근거 커밋 `6b5d7c1`) |

---

## 10. 다음 단계와 결정 대기

- **C1 주 지표** — SAR 대 SAR_abs (§8-1).
- **미세 attribution 값 처리** — 반올림 수준 값을 0으로 볼지(§8-9).
- **서버 실행** — 아직 서버에서 클린 재현을 한 적이 없다(`docs/SETUP.md`). 250개 전수·더 큰 S로 가려면 필요.
- **D-C5-3**도 코드·보고서에서 인용되지만 DECISIONS에 항목이 없다(이번에 이관하지 않음).

---

## 부록 A. E3 규칙 확정 전 버전 (인덱스 동률) — `reports/w2_ranking_v1_index_ties.md`

**규칙 확정 전 버전이다.** 동률을 세그먼트 번호가 작은 쪽부터 깼고, LeRF는 오름차순으로 따로 정렬했다
(`faithfulness_ad.order`). E3 실행(2026-09-25, 커밋 `6c83779`)에서 생성됐고 Claude Code가 값을 봤다(D-PRE-2).
주 결과는 본문 §6의 무작위 동률 버전이다.

| 평가 연산자 | Random | FeatureAblation | KernelSHAP | MPNative |
|---|---|---|---|---|
| B1_zero | 1 | 3.5 | 3.5 | 2 |
| B2_global_mean | 3 | 1 | 2 | 4 |
| B3_local_mean | 3 | 1 | 2 | 4 |
| B4_linear_interp | 4 | 3 | 1 | 2 |
| B5_gaussian | 3 | 2 | 4 | 1 |
| B6_shuffle | 4 | 3 | 2 | 1 |

- Kendall's W = 0.069 (95% CI [0.019, 0.478]); 연산자 쌍 ρ −0.738 ~ 1.000; 시리즈별 W 중앙값 0.378.
- 자기 일치 민감도 ρ: B1 −0.738 · B2 0.8 · B3 0.8 · B4 0.8 · B5 −0.4 · B6 0.4.
- 그림: `reports/figures/w2/rank_bump_v1_index_ties.png`.

---

## 부록 B. 주요 산출물 위치

| 파일 | 내용 |
|---|---|
| `docs/PREREG_W2.md` | 사전 등록 (`0d041d7`, 파일럿 전) |
| `reports/w2_pilot.md`, `reports/figures/w2_pilot/F2_*.{png,pdf}` | 파일럿, 그림 F2 |
| `reports/w2_gate2.md` | E0, stumpy 추적, 게이트 2, 표본, 연산자 표 |
| `reports/w2_artifact.md`, `reports/figures/w2/ai_by_operator.png` | E2 |
| `reports/w2_ranking.md`, `reports/figures/w2/rank_bump.png` | E3 (무작위 동률, 주 결과) |
| `reports/w2_ranking_v1_index_ties.md`, `reports/figures/w2/rank_bump_v1_index_ties.png` | E3 규칙 확정 전 버전 |
| `reports/w2_plausibility.md` | E4 |
| `results/w2/` | 원자료 (ai / sar / faithfulness(v1) / faithfulness_v2 `.csv.gz`, plausibility.csv, e0, 예산·실행 기록) |
| `docs/DECISIONS.md` | 결정 기록 (승인 섹션 `f29eddf`, 정렬·E4 결정 `b8b783c`) |
