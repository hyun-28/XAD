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

## 연구자 결정 대기 (BRIEF §8)

- **D3 인덱스 규약** — `01_data_audit.md` §6과 `figures/`를 보고 `matlab_1based_inclusive`를 승인하거나
  `tsad-xai/configs/conventions.yaml`에서 바꾸고 `02_audit.py`를 다시 실행.
- **is_medical에서 Gait/EPG를 비의료로 둔 기본값** — `07_DECISIONS.md` D-A5-3.
- **라이선스** — CC BY-NC 4.0 표기의 출처 확인 (`08_DATA_LICENSE.md`).

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
