# 구현 브리프 — W1: UCR 데이터 계보 감사 + TSB-AD 안전 래퍼 + 게이트 1

> **대상:** Claude Code
> **작성:** 2026-09-18 · 연구 파트너(claude.ai) → 구현(Claude Code) 인계
> **프로젝트:** TSAD-XAI 평가 프로토콜 (KIISE 포스터 논문, 마감 2026-10-12)
> **이 문서의 지위:** 요구사항 명세. 여기 적힌 "STOP" 조건에 걸리면 **스스로 결정하지 말고 멈추고 보고**할 것.

---

## 0. 왜 이 작업이 게이트인가

이 연구는 마스킹 연산자가 **정상 구간에서** 이상 점수를 얼마나 움직이는지(Artifact Index, AI)를 잰다.
그러려면 세 가지가 흔들리면 안 된다.

1. **어느 시리즈를 쓰는가** — 원본 UCR 250개인가, TSB-AD가 큐레이션한 부분집합인가.
2. **이상 구간(GT)이 정확히 어디인가** — 인덱스 기준(0/1-based), 끝점 포함 여부, 출처(파일명 vs Label 열).
3. **탐지기 점수가 재현 가능한가** — 같은 입력에 같은 점수, 조용한 실패 없음, 섭동 전후 동일 모델.

셋 중 하나라도 틀리면 이후 모든 수치가 오염되고, **오염은 결과 표에서 보이지 않는다.**
그래서 실험 코드보다 이걸 먼저 한다.

---

## 1. 비협상 원칙 (Non-negotiables)

1. **Fail loud.** 예상과 다른 모든 상황은 예외를 던진다. 경고 로그 후 계속 진행 금지.
   유일한 예외는 감사(audit) 스크립트: 위반을 **수집해 보고서에 기록**하고 종료 코드 ≠ 0으로 끝낸다.
2. **추측 금지.** 인덱스 규약, 파일 포맷, API 동작을 "아마 이럴 것"으로 코딩하지 말 것.
   문서·소스·실측으로 확인하고, 확인 근거(파일 경로 + 줄 번호, 또는 실측 테스트)를 코드 주석과 보고서에 남긴다.
3. **원본 불변.** 다운로드한 원본 파일은 읽기 전용. 모든 파생물은 별도 디렉터리에, 생성 스크립트와 함께.
4. **모든 수치는 스크립트가 만든다.** 보고서의 숫자를 손으로 적지 않는다. `reports/*.md`는 스크립트 출력물.
5. **결정은 기록한다.** 선택지가 있었던 모든 지점을 `docs/DECISIONS.md`에 "무엇을 / 왜 / 대안 / 되돌리는 법"으로 남긴다.
6. **연구자 결정 사항은 대신 정하지 않는다.** §8의 항목은 기본값으로 구현하되 **설정값으로 노출**하고, 보고서에 "미결정"으로 표시.

---

## 2. 환경

### 2.1 머신
- **주 개발:** MacBook (Apple Silicon, arm64), conda(Miniforge).
- **실행:** 원격 Linux GPU 서버 (연구실). WSL2 경로는 **폐기됨** — 관련 설정 만들지 말 것.
- 코드는 **양쪽에서 동일하게 동작**해야 한다. OS 분기 코드 금지. 경로는 `pathlib` + 설정 파일.

### 2.2 패키지 고정
- Python **3.11** (TSB-AD 권장 범위 3.8–3.12 안).
- **`scikit-learn < 1.6` 고정.** 1.6 이상에서 `IForest.decision_function()`이
  `check_is_fitted()` 경로에서 `AttributeError: 'IForest' object has no attribute '__sklearn_tags__'`로 깨진다.
- **먼저 확인할 것:** TSB-AD `requirements.txt`가 sklearn 버전을 자체 고정하는지. 충돌하면 **STOP 후 보고.**
- TSB-AD의 `pytorch-cuda` 설치 라인은 Mac에서 쓰지 않는다. 이번 작업은 IForest만 필요하므로
  **최소 의존성으로 설치 가능한지** 확인하고, 불가하면 서버 전용으로 분리한 방법을 `docs/SETUP.md`에 기록.
- 산출물: `environment.yml`(사람이 읽는 것) + `conda-lock` 또는 `pip freeze` 결과(정확한 재현용) 둘 다 커밋.
- 모든 실행 로그 첫 줄에 `python / numpy / sklearn / TSB-AD(git commit hash) / OS / hostname` 기록.

---

## 3. Task A — UCR 데이터 계보 감사 (가장 먼저)

### A0. 해결해야 할 모순

현재 연구 기록에는 두 전제가 **동시에** 적혀 있다.

- (가) "UCR Anomaly Archive 250 시리즈, 이상 구간 start/end가 파일명에 인코딩, `DISTORTED` 접두사로 합성 이상 구분"
- (나) "TSB-AD 큐레이션 경유로 접근"

그런데 TSB-AD 파일명은 형식이 다르다:
`<idx>_<Source>_id_<n>_<Domain>_tr_<train_end>_1st_<first_anom_start>.csv`
(예: `001_NAB_id_1_Facility_tr_1007_1st_2014.csv`). CSV 마지막 열이 `Label`.

이 형식에는 **원본 UCR 이름(예: `DISTORTED1sddb40`)이 없고, 이상 끝점도 없다.** 또 TSB-AD는
큐레이션 과정에서 **라벨 오류·편향이 있는 시리즈를 제거**했다고 밝히므로, UCR 250개가 전부 들어있다는 보장이 없다.

→ **해소 방향 (2026-09-18 갱신):** 데이터는 **Hexagon ML/UCR Time Series Anomaly Archive 2021 완전판**
(2021-08-14 공개, 레이블·출처 포함, 250개)을 **직접** 쓴다. TSB-AD는 **탐지기 구현(IForest)만** 가져온다.
TSB-AD의 UCR 부분집합·Label 열은 1차 데이터가 아니다 → A4는 필수 게이트에서 **선택적 교차검증**으로 격하.

**주의 — 버전 혼동 위험:** 같은 아카이브에 두 판이 있다.
- SIGKDD 2021 공모전판: 200개 시리즈, 학습/테스트 단계 분리 (테스트 레이블 비공개였음)
- 2021 완전판: 250개, 파일명에 `train_end_begin_end` 인코딩
미러·재배포본(Kaggle, 타 저장소 등)은 둘 중 무엇인지, 수정됐는지 보장이 없다. **A1의 체크섬 대조가 이걸 판정한다.**

→ **Task A의 목적은 우리가 가진 데이터가 공식 2021 완전판과 비트 단위로 같은지 확정하고, GT를 흔들림 없이 정의하는 것이다.**

### A1. 2021 완전판 확보 및 동일성 검증

- 공식 URL (Wu & Keogh 지원 페이지 명시):
  `https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip`
- 압축 해제 후 경로: `AnomalyDatasets_2021/UCR_TimeSeriesAnomalyDatasets2021/FilesAreInHere/UCR_Anomaly_FullData/`
- **연구자가 이미 로컬 사본을 가지고 있다.** 절차:
  1. 로컬 사본 경로를 설정 파일에서 받는다 (하드코딩 금지).
  2. 공식 zip을 **별도로** 받아 풀고, 파일별 SHA256을 로컬 사본과 대조.
  3. 결과를 `reports/data_audit.md` 맨 위에 표로: 일치 / 불일치 / 로컬에만 있음 / 공식본에만 있음.
  4. **한 파일이라도 다르면 STOP.** 어느 쪽을 쓸지는 연구자 결정. 이후 모든 작업은 **공식본 기준**으로 한다.
- zip과 각 파일의 SHA256을 `data/raw/ucr/CHECKSUMS.sha256`에 기록.
- **assert: `.txt` 파일 정확히 250개.** 200개면 공모전판이다 → STOP.
- zip 안의 **부속 문서를 전부 보존**하고 목록을 보고서에 적는다. 특히 **`UCR_AnomalyDataSets.pptx`**
  (아카이브 보충 자료로 인용되는 슬라이드, 주입된 이상 설명 포함)를 찾을 것. 저자들은 완전판을
  "레이블과 출처(provenance)"와 함께 공개한다고 밝혔으므로, **A2 인덱스 규약과 A5 도메인 분류의 1차 근거가 여기 있을 가능성이 높다.**
  pptx 텍스트를 `python-pptx`로 추출해 `docs/ucr_supplement_text.md`로 저장하고, 규약·출처 관련 문장을 인용 위치와 함께 정리.
- **라이선스:** 아카이브 자체의 이용 조건을 부속 문서에서 찾아 `docs/DATA_LICENSE.md`에 기록.
  (지원 페이지의 CC BY-NC 4.0 표기는 웹사이트 콘텐츠에 대한 것일 수 있으므로 아카이브 라이선스로 간주하지 말 것 — 확인 필요)

### A2. 원본 파일명 파서

형식: `<num>_UCR_Anomaly_<name>_<train_end>_<anom_begin>_<anom_end>.txt`
(예: `012_UCR_Anomaly_tiltAPB1_100000_114283_114350.txt`)

- `str.split("_")` 금지 — `<name>`에 밑줄이 있으면 깨진다. 끝에서부터 숫자 3개를 떼는 정규식을 쓸 것:
  ```python
  UCR_RE = re.compile(r"^(?P<num>\d{3})_UCR_Anomaly_(?P<name>.+)_(?P<train_end>\d+)_(?P<begin>\d+)_(?P<end>\d+)\.txt$")
  ```
- **assert: 250개 전부 매칭.** 불일치 파일은 목록으로 출력 후 STOP.
- `num`이 001–250 연속이고 중복 없는지 확인.

#### ⚠ 인덱스 규약 — 가장 흔한 조용한 버그
공개 튜토리얼들은 `train_end`를 "1부터 X까지가 학습 데이터"로 설명한다. 즉 **1-based 서술**일 가능성이 있다.
`begin`/`end`가 0-based인지 1-based인지, `end`가 포함(inclusive)인지는 **문서로 확인해야 한다.**

- A1에서 보존한 부속 문서에서 규약 근거를 찾는다.
- 근거가 없으면 **실측 검증**: 몇 개 시리즈(특히 이상이 눈에 띄는 것)에서 `x[begin-1]`, `x[begin]`, `x[end]`, `x[end+1]` 주변을
  플롯해 `reports/figures/index_convention/`에 저장하고, 어느 규약이 시각적으로 맞는지 보고.
- 최종 규약은 `src/data/conventions.py`의 **단일 상수**로 정의하고, 모든 코드는 이 상수를 거쳐 0-based 반열린구간 `[start, stop)`으로 변환한다.
- **확정 근거가 없으면 STOP — §8 결정 항목.** 기본값을 조용히 고르지 말 것.

### A3. 로딩 무결성

250개 전부 로드하고 다음을 **감사**한다(위반은 수집, 마지막에 일괄 보고):

| 검사 | 조건 |
|---|---|
| 형태 | 로드 결과가 1-D `float64`. (한 줄에 공백 구분으로 저장된 파일 등 **포맷 변종이 있을 수 있다** — 로더가 모두 처리하는지 확인) |
| 결측 | NaN / inf 없음 |
| 길이 | `len(x) > end` |
| 순서 | `train_end < begin <= end` (UCR 설계상 이상은 학습 구간 이후) |
| 이상 길이 | `end - begin + 1 ≥ 1`, 분포 요약(min/median/max) |
| 상수 구간 | 학습 구간이 전부 동일값인 시리즈 여부 (z-정규화 시 0 나눗셈) |

결과를 `data/manifest/ucr_manifest.csv`에 기록(컬럼은 A6).

### A4. TSB-AD-U의 UCR 부분집합과 대조 — **선택 (게이트 아님)**

> 2021 완전판을 1차 데이터로 쓰기로 했으므로 이 단계는 **게이트가 아니다.** A1–A3, A5, Task B, Task C를 먼저 끝내고
> 시간이 남으면 수행. 가치: TSB-AD가 어떤 UCR 시리즈를 왜 뺐는지 알면 논문 Limitations에 "우리는 전체 250을 썼고,
> TSB-AD 큐레이션에서 제외된 k개에서도 결론이 유지되는가"를 민감도 분석으로 쓸 수 있다.
> 아래 항목 중 **5번(GT 불일치)은 STOP 조건에서 제외**하고 보고만 한다.

1. TSB-AD-U 다운로드 (`https://www.thedatum.org/datasets/TSB-AD-U.zip`), SHA256 기록.
2. 파일명에 `_UCR_`이 들어간 파일 목록과 **개수**를 센다. 이 숫자가 250인지 아닌지가 이 감사의 첫 번째 답이다.
3. TSB-AD 파일명 파서:
   ```python
   TSB_RE = re.compile(r"^(?P<idx>\d+)_(?P<src>[A-Za-z0-9]+)_id_(?P<n>\d+)_(?P<domain>[A-Za-z]+)_tr_(?P<train_end>\d+)_1st_(?P<first>\d+)\.csv$")
   ```
   정규식은 **추정**이다. 실제 UCR 파일 전부가 매칭되는지 assert하고, 안 되면 실제 파일명을 보고 수정하되 수정 근거를 DECISIONS에 기록.
4. **TSB-AD 파일 → 원본 UCR 파일 매핑.** `id_<n>`이 원본 `num`과 같다고 **가정하지 말 것.** 다음 순서로 매칭하고 각 매칭의 근거를 기록:
   - (1) 길이 동일 + `train_end` 동일 + 값 배열 `np.allclose`(TSB-AD가 정규화·절단했을 수 있으니 실패 시 z-정규화 후 비교, 그래도 실패 시 부분열 매칭)
   - (2) 매칭 방법(`exact` / `allclose` / `znorm` / `subseq` / `unmatched`)을 manifest에 기록
5. **GT 대조.** TSB-AD `Label` 열에서 이상 구간을 추출(연속 1 구간의 시작·끝, 구간 개수)하고 원본 파일명 GT와 비교.
   - 구간 개수 ≠ 1, 시작·끝 불일치(오프셋 포함) 전부 목록화.
   - **TSB-AD가 라벨을 교정했을 가능성**이 있다. 어느 쪽이 맞는지 Claude Code가 판정하지 말 것 → §8 결정 항목.
6. **TSB-AD에서 빠진 원본 UCR 시리즈**를 목록화하고, TSB-AD 문서·저장소에 제외 사유가 있으면 인용.

### A5. 도메인 매핑 — 두 가지 분류 체계

같은 시리즈에 **두 개의 도메인 라벨**을 붙인다. 섞지 말 것.

**(i) `domain_tsbad`** — TSB-AD 파일명의 `<Domain>` 필드 그대로. 매핑된 시리즈만.

**(ii) `domain_goswami`** — Goswami et al. (ICLR 2023, arXiv:2210.01078)의 9분할:
Acceleration / Air Temperature / ABP / EPG / ECG / Gait / NASA / Power Demand / RESP.
이 분할은 이후 여러 논문이 인용하는 관행이지만, **분류 규칙 자체는 논문 본문에 없다.**

- **0순위 (신규):** A1에서 추출한 아카이브 **공식 출처(provenance) 정보**. 원 저자가 각 시리즈의 출처를 밝혔다면
  이게 가장 권위 있는 분류다. 이 경우 컬럼 `domain_official`을 추가하고, Goswami 9분할과의 대응표를 만든다.
  공식 분류와 Goswami 분류가 어긋나는 시리즈는 목록으로 보고.
- **1순위:** 공식 코드 `https://github.com/mononitogoswami/tsad-model-selection` (Apache-2.0)에서
  UCR 데이터 로더·설정 파일을 찾아 **원본 매핑을 그대로** 가져온다. 커밋 해시와 파일 경로를 기록.
- **2순위(1순위 실패 시):** 원본 UCR `<name>` 문자열 기반 키워드 규칙을 **직접 작성**하되
  - 컬럼명을 `domain_goswami_reconstructed`로 바꾸고,
  - 매칭 안 된 이름·복수 매칭된 이름을 `reports/domain_unmatched.csv`로 출력하고,
  - 보고서에 "Goswami의 매핑이 아니라 우리의 재구성"이라고 명시.
  - **논문에 "Goswami 관행을 따랐다"고 쓸 수 있는지는 이 결과에 달려 있다.**

**(iii) `is_distorted`** — 원본 이름에 `DISTORTED` 접두사 여부.

**(iv) `is_medical`** — `domain_goswami`가 ABP/ECG/RESP면 True. (Gait, EPG 분류는 규칙을 명시하고 DECISIONS에 기록)

### A6. 산출물

**`data/manifest/ucr_manifest.csv`** — 원본 250행 기준, 컬럼:
```
num, name, filename, sha256, length, train_end_raw, begin_raw, end_raw,
start0, stop0,                      # conventions.py로 변환한 0-based [start, stop)
anom_len, is_distorted, domain_goswami(_reconstructed), is_medical,
in_tsbad, tsbad_filename, tsbad_domain, match_method,
tsbad_label_n_segments, tsbad_label_start0, tsbad_label_stop0, gt_agree,
audit_flags                          # A3 위반 코드들, ;로 구분
```

**`reports/data_audit.md`** (스크립트 생성) — 최소 다음 표를 포함:

1. 요약: 원본 개수 / TSB-AD UCR 개수 / 매핑 성공 / 미포함 / GT 불일치 / A3 위반 수
2. **도메인 × DISTORTED 교차표** (원본 250 기준, TSB-AD 부분집합 기준 두 벌)
3. **비의료 서브셋 크기** — 전체, DISTORTED 제외 시. **30 미만이면 굵게 경고.**
   (공개 요약에 따르면 원본 UCR은 의료 약 64%, 생물 22%, 산업 9%, 기온 5% 구성이라 산업 계열은 20여 개 수준일 가능성이 있다 — 실측으로 확인할 것)
4. 이상 길이 분포 (도메인별)
5. GT 불일치 목록 전체
6. 인덱스 규약 결정 상태와 근거

---

## 4. Task B — TSB-AD 안전 래퍼

### B0. 이 래퍼가 필요한 진짜 이유

단순 에러 처리 문제가 아니다. **실험 설계상 필수**다.

- `run_Unsupervise_AD()`는 **호출할 때마다 재학습**한다.
- Artifact Index는 "같은 모델에 원본을 넣은 점수 vs 섭동본을 넣은 점수"의 차이다.
  섭동본을 넣을 때 모델이 다시 학습되면, 점수 변화에 **섭동 효과 + 재학습 분산**이 섞인다. 측정이 성립하지 않는다.
- 따라서 **fit 한 번 → score 여러 번** 인터페이스가 반드시 필요하다.

### B1. 먼저 소스를 읽어라

코딩 전에 `TSB_AD/model_wrapper.py`에서 IForest 실행 경로를 끝까지 추적하고 `docs/TSBAD_INTERNALS.md`에 정리:
- IForest 객체 생성 인자 (slidingWindow, n_estimators, max_features, `normalize` 등)와 기본값
- 슬라이딩 윈도우 크기 결정 방식 (자동 주기 추정인지 고정값인지)
- **윈도우 점수 → 타임스텝 점수 변환 방식** (패딩: 앞/뒤/양쪽, 값: 0/edge/평균). 이건 AI와 localization 둘 다에 영향을 준다.
- 점수 후처리 (MinMax 스케일링 여부 등)
- 에러 발생 시 반환 형태 (문자열을 반환하는 정확한 코드 위치)
- `random_state` 전달 경로

### B2. 인터페이스

```python
# src/detectors/base.py
class DetectorError(RuntimeError): ...

class FittedDetector(Protocol):
    name: str
    config: dict           # 재현에 필요한 모든 하이퍼파라미터 (기본값 포함, 명시적으로)
    def score(self, x: np.ndarray) -> np.ndarray: ...   # 재학습 없음

def fit_detector(name: str, x_fit: np.ndarray, *, seed: int, **hp) -> FittedDetector: ...
```

- `fit_on` 설정: `"train_prefix"`(= `x[:train_end]`로 학습) | `"full"`(= TSB-AD 기본 동작, 전체로 학습).
  **기본값 `train_prefix`**, 단 게이트 1에서는 **두 방식 모두 실행**해 비교 (§8 결정 항목).
- IForest는 **`normalize=False` 강제.** `normalize=True`면 `fit()`과 `decision_function()`의 전처리가 어긋나 점수가 불일치한다.
  이 인자를 사용자가 True로 넘기면 예외.

### B3. 반환값 검증 (모든 `score()` 호출마다)

| 검사 | 실패 시 |
|---|---|
| `isinstance(out, str)` | `DetectorError(out)` — TSB-AD의 조용한 에러 문자열을 여기서 잡는다 |
| `np.ndarray`로 변환 가능 | `DetectorError` |
| `(n,1)`이면 squeeze, 그 외 2-D | `DetectorError` |
| `len(out) == len(x)` | `DetectorError` (패딩 규칙 위반) |
| 전부 유한값 | `DetectorError` |
| 분산 > 0 | `DetectorError` 또는 설정에 따라 flag (상수 점수는 AI 계산 불가) |
| dtype | `float64`로 변환 |

### B4. 재현성

- 모든 확률적 요소에 seed 전달. IForest `random_state` 경로를 B1에서 확인한 대로.
- 전역 `np.random` 상태에 의존하는 코드가 TSB-AD 내부에 있으면 기록하고 래퍼에서 격리.

### B5. 로깅

모든 fit/score 호출을 `runs/<run_id>/calls.jsonl`에 한 줄씩:
`series_num, detector, config, seed, fit_on, phase(fit|score), n, runtime_s, ok, error`.
실패는 `runs/<run_id>/failures.jsonl`에 추가로. **실패한 시리즈를 조용히 건너뛰지 않는다** —
루프는 계속 돌되 끝에서 실패율을 계산하고, 설정한 허용치(기본 0%)를 넘으면 종료 코드 ≠ 0.

### B6. 테스트 (pytest, 모두 CI처럼 로컬에서 통과해야 완료)

```
tests/test_conventions.py
  - 합성 파일명으로 0/1-based 변환 왕복 검증
  - 알려진 시리즈 1개의 GT 경계값 스냅샷 테스트
tests/test_ucr_parser.py
  - 이름에 밑줄 포함 케이스 / 잘못된 파일명 거부
tests/test_loader.py
  - 포맷 변종 파일 로드 (A3에서 발견된 실제 변종으로)
tests/test_wrapper_errors.py
  - TSB-AD가 문자열 반환하도록 모킹 → DetectorError
  - 길이 불일치 / NaN / 상수 출력 모킹 → DetectorError
  - normalize=True 전달 → 예외
tests/test_wrapper_determinism.py
  - 같은 seed 두 번 → 점수 bitwise 동일
  - fit 1회 후 score 2회 → 동일 (재학습 없음 확인)
tests/test_fit_score_consistency.py
  - normalize=False에서 TSB-AD 원래 경로와 래퍼 경로의 점수가 일치 (허용오차 명시)
  - normalize=True에서 불일치가 재현됨 (함정이 실제로 존재함을 문서화하는 테스트)
tests/test_padding.py
  - 윈도우→타임스텝 변환 규칙이 B1에서 문서화한 대로인지
```

---

## 5. Task C — 게이트 1 실행

### C1. 실행
- 대상: **(a) 원본 UCR 250 전체**, **(b) TSB-AD UCR 부분집합** — 둘 다.
- 탐지기: IForest, `normalize=False`, seed 3개(0, 1, 2), `fit_on` 두 방식.
- 실행 위치: 서버. 단 Mac에서 시리즈 5개로 스모크 테스트 먼저.

### C2. 탐지 성공 기준 — 사전 등록 (결과 보기 전에 고정)

**주 기준 (P1):** 테스트 구간(`train_end` 이후)에서
`max(score[start0:stop0]) > percentile(score[normal_test], 99)`
여기서 `normal_test` = 테스트 구간에서 이상 구간 ± buffer를 제외한 부분.
buffer = 0과 buffer = 윈도우 길이 두 조건 모두 보고.

**보조 기준 (P2):** 테스트 구간 점수의 argmax 위치가 `[start0 − 100, stop0 + 100)` 안에 있는가.
UCR 공모전(KDD Cup 2021) 채점 방식에서 유래한 관행으로 알려져 있으나 **정확한 허용 폭 규칙은 확인 필요** —
원본 zip 부속 문서에서 근거를 찾아 인용하고, 못 찾으면 보고서에 "근거 미확인"으로 표기.

### C3. 게이트 판정
- **통과 조건:** (a) 원본 250 기준, P1 성공률 ≥ 50% (seed 3개 중앙값, `fit_on=train_prefix`).
- 미달이면 **STOP** — 탐지기 교체 여부는 연구자가 결정.
- 성공률은 **도메인별 / DISTORTED별 / 비의료 서브셋별**로도 쪼개 보고.

### C4. 산출물
- `results/gate1/scores/<num>_<fit_on>_seed<k>.npy` (원점수 전부 보존 — 이후 AI 계산에 재사용)
- `results/gate1/detection.csv` (시리즈 × 조건 × P1/P2 결과)
- `reports/gate1.md` (스크립트 생성): 성공률 표, seed 간 분산, `fit_on` 두 방식 차이, 실패 목록, 실행 환경 헤더

---

## 6. 저장소 구조

```
tsad-xai/
├── configs/            # 경로, seed, 탐지기 hp, 인덱스 규약 선택
├── data/
│   ├── raw/            # 원본 (읽기 전용, git 제외, CHECKSUMS만 커밋)
│   └── manifest/       # ucr_manifest.csv (커밋)
├── docs/
│   ├── SETUP.md        # Mac / 서버 설치 절차
│   ├── DECISIONS.md    # 결정 로그
│   └── TSBAD_INTERNALS.md
├── src/
│   ├── data/           # conventions.py, ucr.py, tsbad.py, domains.py
│   └── detectors/      # base.py, iforest.py
├── scripts/            # 01_download.py, 02_audit.py, 03_gate1.py, 04_report.py
├── tests/
├── results/            # git 제외, 요약 csv만 커밋
├── reports/            # 스크립트 생성 md + figures (커밋)
├── environment.yml
└── README.md           # 재현 명령 3줄
```

재현 명령은 README에 이 3줄로 끝나야 한다:
```bash
conda env create -f environment.yml && conda activate tsad-xai
python scripts/01_download.py && python scripts/02_audit.py
python scripts/03_gate1.py && python scripts/04_report.py
```

---

## 7. STOP 조건 — 멈추고 보고할 것

1. 원본 UCR 파일 수 ≠ 250 (200이면 공모전판), 파일명 파싱 실패, 또는 **로컬 사본과 공식 2021 완전판의 체크섬 불일치.**
2. 인덱스 규약(0/1-based, 끝점 포함 여부)을 문서·실측으로 확정할 수 없음.
3. TSB-AD 요구 sklearn 버전이 `< 1.6` 고정과 충돌.
4. TSB-AD 소스에서 fit/score 분리가 불가능한 구조 (IForest를 직접 구성해야 하는 경우 포함 — 그 경우 TSB-AD 경로와 점수 일치를 테스트로 입증하는 방안까지 제시 후 대기).
5. ~~GT 불일치~~ → A4가 선택 단계로 격하되어 STOP 아님. 단 A3에서 **완전판 파일명 GT 자체의 모순**
   (예: `begin > end`, `end ≥ len(x)`, `begin ≤ train_end`)이 나오면 STOP.
6. 게이트 1 미달.

보고 형식: 무엇을 봤는가 / 근거(파일·줄·수치) / 가능한 선택지 2~3개와 각각의 영향.

---

## 8. 연구자가 결정할 항목 (Claude Code는 기본값 구현 + 설정 노출 + 보고만)

| # | 결정 | 기본값 | 왜 중요한가 |
|---|---|---|---|
| D1 | ~~사용 시리즈~~ | **결정됨: 2021 완전판 250** | TSB-AD 부분집합은 선택적 민감도 분석용 |
| D2 | ~~GT 출처~~ | **결정됨: 완전판 파일명** | 단 A1 체크섬 일치가 전제. TSB-AD Label과의 차이는 보고만 |
| D2' | 로컬 사본 ≠ 공식본일 때 어느 쪽을 쓸지 | 공식본 | 로컬 사본으로 이미 돌린 dry-run 결과의 유효성이 걸림 |
| D3 | 인덱스 규약 | 근거 확정 전 미정 | 경계 한 칸 차이가 짧은 이상에선 결과를 뒤집는다 |
| D4 | `fit_on`: train_prefix vs full | train_prefix | AI 측정의 기준 모델 정의 |
| D5 | 도메인 분류: Goswami 원본 vs 재구성 | 원본 확보 시 원본 | 논문의 인용 가능 여부 |
| D6 | 탐지 성공 필터를 이후 실험에 적용할지 | 미적용(전/후 모두 보고) | 필터가 결과를 만드는 것처럼 보이면 안 됨 |

---

## 9. 완료 정의 (Definition of Done)

- [ ] `pytest` 전부 통과 (Mac, 서버 둘 다)
- [ ] `ucr_manifest.csv` 250행, 모든 컬럼 채워짐 (미확정 값은 명시적 `NA` + 사유)
- [ ] `reports/data_audit.md` 생성, §3 A6의 6개 항목 포함
- [ ] `reports/gate1.md` 생성, 통과/미달 명시
- [ ] `docs/DECISIONS.md`에 작업 중 내린 모든 선택 기록
- [ ] `docs/TSBAD_INTERNALS.md`에 B1 항목 전부 기록 (소스 위치 포함)
- [ ] README 3줄로 클린 환경에서 재현 확인 (서버에서 새 conda env로 1회)
- [ ] §7 STOP 조건에 걸린 항목이 있으면 그 상태로 멈추고 보고 — 우회 금지

---

## 부록. 참고 출처

- Wu, R. & Keogh, E. *IEEE TKDE* 35(3):2421–2429 (2023), arXiv:2009.13807 — UCR Anomaly Archive
- Liu, Q. & Paparrizos, J. NeurIPS 2024 D&B — TSB-AD. 코드: `github.com/TheDatumOrg/TSB-AD`
- Goswami, M. et al. ICLR 2023, arXiv:2210.01078 — UCR 9-도메인 분할. 코드: `github.com/mononitogoswami/tsad-model-selection`
