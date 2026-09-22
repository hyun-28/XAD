# VERIFICATION — Task A (UCR 데이터 계보 감사) 검토 패키지

`BRIEF.md` §3 Task A의 산출물을 검토용으로 한곳에 모은 폴더입니다.
**여기 있는 파일은 전부 `tsad-xai/`에서 복사한 것**이며, 원본과 재생성 스크립트는 `tsad-xai/`에 있습니다.
숫자는 손으로 적지 않았습니다 — 모든 수치는 `01_data_audit.md`(스크립트 생성물)를 보세요.

## 무엇을 검토하면 되는가

| 파일 | 검토 포인트 | BRIEF |
|---|---|---|
| `01_data_audit.md` | A1 체크섬 대조표, A3 위반 수, 도메인×DISTORTED 교차표, 비의료 서브셋 크기, 이상 길이 분포, GT 불일치, **§6 인덱스 규약 근거** | A6 |
| `02_ucr_manifest.csv` | 250행. `start0/stop0`는 `09_conventions.yaml`의 규약으로 변환된 값. A4 컬럼은 `NA`(미실행) | A6 |
| `03_CHECKSUMS.sha256` | 공식 zip + 파일별 SHA256 | A1 |
| `04_audit_a1.json` | 로컬 사본 vs 공식본 비교 원자료 | A1 |
| `05_index_convention_evidence.csv` + `figures/` | 1~2점짜리 이상 20개에서 두 인덱스 해석(H1 1-based / H0 0-based)의 대비 | A2 |
| `06_domain_unmatched.csv` | 공식 출처 분류와 Goswami 분류가 다르거나 NA인 시리즈 | A5 |
| `07_DECISIONS.md` | 작업 중 내린 모든 선택 — 무엇/왜/대안/되돌리는 법 | §1.5 |
| `08_DATA_LICENSE.md` | 아카이브가 라이선스에 대해 (안) 말하는 것 | A1 |
| `09_conventions.yaml` | 현재 적용 중인 인덱스 규약과 그 근거 (D3) | A2, §8 |
| `10_pytest_result.txt` | 테스트 결과 요약 | §9 |
| `11_recording_groups.csv` | F2: 비교한 1,037쌍 전부 — 상관값과 판정 (`01_data_audit.md` §7의 원자료) | 후속 F2 |
| `12_stats.yaml` / `13_subsets.yaml` / `14_sentinels.yaml` | 분석 단위·서브셋·sentinel 설정값 | 후속 F2/F4/F3 |
| `15_BRIEF_A-followup.md` | 후속 브리프 원문 | — |
| `figures/sentinels/` | −999 위치: plain vs DISTORTED/NOISE 쌍둥이 (123, 184, 185) | 후속 F3 |
| `16_TSBAD_INTERNALS.md` | TSB-AD 1.5 IForest 경로 소스 추적(줄 번호·실측), 벤치마크 HP 추적 | B1 |
| `17_detectors.yaml` | 탐지기 기본값(벤치마크 추적값)과 연구자 변형(max_features, fit_on) | B2 |
| `18_gate1_IForest.md` | 게이트 1 IForest 판정(미달) — 조건별 성공률, 도메인/변형/서브셋 분해, seed 분산, 짝 비교 | C3–C4 |
| `19_gate1_IForest_detection.csv` | 시리즈 × 조건(12) × P1/P2 원자료 3,000행 | C4 |
| `22_gate1_MatrixProfile.md` | **게이트 1 MatrixProfile 판정** (교체 탐지기, D-C5-1/2/3) | C3–C4 |
| `23_gate1_MatrixProfile_detection.csv` | 시리즈 × 조건(2) × P1/P2 + 학습 구간 sanity 원자료 | C4, D-C5-4 |
| `24_wafer_zero_class.csv` | Wafer 25개 모델의 zero class + 테스트 정확도 재현 | BRIEF2 D3 |
| `21_SIMIC_PORT.md` §4 | 논문 Eq. 1–6 vs 코드 대조 (PDF 확보 완료) | BRIEF2 D1-3 |
| `20_SETUP.md` | Mac/서버 환경 절차, 새 env 검증 기록 | §2 |

## 게이트 1 (2026-09-22) — **미달, STOP (BRIEF §7-6)**

`18_gate1.md` §1–§2: 사전 등록 조건(`fit_on=train_prefix`, `max_features=1`, buffer=w) P1 = **28.4%**(seed 중앙값) < 50%.
네 조건 전부 50% 미만(최고 `full`/`max_features=1.0` 39.2%). 탐지기 교체·기준 변경은 연구자 결정.

## 후속 조치(2026-09-22) — `01_data_audit.md` §7–§9, 결정 반영 완료

- **STOP-2 → 해제 (D-F2-2):** `content_group`(89)이 주 분석 단위, `name_group`(100)은 민감도 분석.
  전체 250이 검정 대상; `non_medical`(22그룹)·`physical`(7그룹)은 기술통계.
- **STOP-3 / F3 → 해제 (D-F3-4):** −999는 결측이 아닌 정상 측정값. 주 분석에서 제외하지 않음. 123 ECG4 조치 없음.
  `no_sentinel` 서브셋과 `data/derived/sentinels/`는 민감도 분석용으로 유지.
- 관련 플래그 3종은 INFO로 내려 `02_audit.py`는 exit 0.

## 연구자 결정 (BRIEF §8)

- **D3 인덱스 규약** — 2026-09-22 승인 (`07_DECISIONS.md` D-A2-3, `09_conventions.yaml` `approved`).
- **is_medical에서 Gait/EPG를 비의료로 둔 기본값** — 유지, `physical` 서브셋 별도 정의 (D-A5-3, D-A5-4).
- **라이선스** — 확정 (D-A1-4, `08_DATA_LICENSE.md`).

## 직접 재현하기

```bash
cd tsad-xai
conda env create -f environment.yml && conda activate tsadxai
python scripts/01_download.py && python scripts/02_audit.py
python -m pytest tests -q
bash ../VERIFICATION/sync.sh      # 이 폴더를 다시 채움
```

`01_download.py`는 공식 zip(184 MB)을 `tsad-xai/data/raw/ucr/official/`에 받아 로컬 사본과 대조합니다.
로컬 사본 경로는 `tsad-xai/configs/paths.yaml`에 있습니다.
