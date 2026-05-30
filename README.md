# job-segmenter

자유 텍스트 구인글을 추천·푸시 실험에서 재사용할 수 있는 세그먼트 단위로 바꾸는 백엔드·데이터 파이프라인입니다.
실제 서비스 데이터에서 최신성 기준으로 추린 10,450건 중 구인글 9,941건을 48개 운영 세그먼트와 정보부족형 115건으로 정리했고,
신규 글을 같은 체계에 자동 배정하는 서비스 흐름까지 구현했습니다.

> 프로젝트의 배경·역할·핵심 의사결정·결과 요약은 같은 디렉터리의 [`포트폴리오.md`](./포트폴리오.md)에 있습니다.
> 이 README는 그 평가 문서가 다루지 않는 코드 구조와 구현 디테일을 담은 기술 참고 문서입니다.

> ### ⚠ 포트폴리오용 부분 공개 저장소
>
> 이 저장소는 코드 구조와 설계 의사결정을 보여주기 위한 공개용 발췌본입니다.
> 보안과 비즈니스 자산 보호를 위해 다음은 포함하지 않았습니다.
>
> - 운영 DB 접속정보 (`local_config.py`) 및 실데이터 (`*.csv`, `*.jsonl`)
> - 학습된 모델 산출물 (`*.pkl`, RF·인코더·RoBERTa 가중치)
> - 설계·구현 기록 사내 문서 (`docs/`), 마케터 검토 리포트, 산출물 디렉터리
> - 실제 테이블명·내부 운영 스크립트·회사 내부 커뮤니케이션
>
> 이 저장소만으로 운영 파이프라인을 그대로 실행할 수는 없습니다. README의 DB·컬럼명은 공개용으로 일부 익명화했습니다.

---

## 1. 프로젝트 개요

대상 서비스는 모델·포토그래퍼·뷰티 전문가를 연결하는 촬영 업계 매칭 플랫폼입니다.
플랫폼에는 브랜드·스튜디오·개인 작가·일반 사용자 등 다양한 주체가
자유 텍스트로 구인글을 등록하며, 약 23,000건의 구인글이 누적되어 있습니다.

이 프로젝트는 구인글을 **작성자가 아니라 글 자체의 스타일** 기준으로
자동 세그먼테이션하는 파이프라인입니다.
LLM으로 구조화된 속성을 추출하고, 비지도 클러스터링으로 세그먼트 체계를 발견한 뒤,
RandomForest 분류기로 신규 구인글을 자동 배정합니다.

| 항목 | 내용 |
|---|---|
| **도메인** | 촬영 업계 매칭 플랫폼 구인글 자동 세그먼테이션 |
| **목표** | 구인글 텍스트 → 구조화 속성 추출 → 세그먼트 체계 확정 → 신규 글 자동 배정 |
| **데이터** | 공개용 명칭 기준 MySQL `job_postings` 약 23,000건 (제목·본문·카테고리·payment 등) |
| **LLM** | Ollama + Qwen2.5:14b (self-hosted) |
| **현재 상태** | v2.1-tune — 세그먼트 확정, RF 배정기 구현, CLI 배치·단건 검증 + REST API 서비스 완료 |

## 2. 내 구현 범위와 검증 상태

| 구분 | 내용 |
|---|---|
| **내 구현 범위** | 파이프라인 구조 설계, LLM 속성 추출, Gower 거리 기반 클러스터링, 셀별 RF 배정기, DB 저장 흐름, 실패 로그·상태 분기, FastAPI REST API |
| **도메인 검토** | 마케터가 분류 기준과 세그먼트 이름을 검토했고, 긴급도 제거·시술 종류 통합·기존 DB 필드 활용 같은 수정으로 반영 |
| **완료된 검증** | 48개 운영 세그먼트 확정, 정보부족형 115건 확인, 단건·배치 배정 서비스 동작 검증, RF hold-out 라벨 재현 99~100% 확인 |
| **아직 안 한 검증** | 세그먼트별 고객 반응률 차이, 장기 드리프트, 회귀 테스트용 Golden Set |

## 3. 풀려는 문제

### 3.1 배경 — 작성 주체가 다양한 자유 텍스트 구인글

플랫폼에는 다양한 주체가 구인글을 올립니다.
브랜드, 스튜디오, 개인 작가, 일반 사용자가 같은 플랫폼을 쓰지만
표현 방식과 정보량이 제각각입니다.

같은 "모델 모집"이라는 제목이라도 실제 성격은 크게 달라질 수 있습니다.

- 브랜드의 상업 광고 촬영
- 작가의 개인 포트폴리오 촬영
- 급하게 인원을 구하는 단기 촬영
- 초보도 가능한 가벼운 촬영
- 전문 모델을 원하는 상업 촬영

반대로 같은 의미가 작성자마다 전혀 다른 단어로 표현되기도 합니다.
결과적으로 구인글은 단순 카테고리(모델/포토/뷰티)만으로 구분되지 않으며,
**구인글끼리 비교할 공통 기준 자체가 없는 상태**입니다.

### 3.2 문제 — 일관성·확장성·데이터 활용이 동시에 막히는 구조

이 상태에서는 여러 한계가 함께 나타납니다.

- **모델 사용자 측면** — 같은 카테고리 안에서도 자신과 무관한 글이 섞여 반복 노출됩니다.
  관련 없는 글을 반복해서 보게 되면 서비스 이탈로 이어질 수 있습니다.
- **운영 측면** — 누구에게 어떤 글을 우선 노출해야 할지 판단할 근거가 없습니다.
  시간순이나 카테고리순으로 노출하면 적합도와 무관한 글이 섞일 수밖에 없습니다.
- **데이터 자산 측면** — 제목·본문이라는 풍부한 텍스트가 쌓여 있지만,
  자유 텍스트 상태로는 분석 단위를 잡을 수 없어 거의 활용되지 못합니다.

대안으로 마케터가 수동으로 분류하는 방식이 있었지만,
구인글이 늘어날수록 인력 부담이 커지고, 사람마다 기준이 달라지며,
같은 사람의 기준도 시간이 지나면 흔들립니다.
결국 수동 분류는 **확장성과 일관성 양쪽 모두 한계**가 있습니다.

### 3.3 세그먼테이션의 역할 — 추천이 아니라 해석을 위한 중간 계층

이 프로젝트의 핵심은 단순히 "구인글을 묶는 것"이 아니라,
**고객 행동을 해석할 수 있는 중간 계층**을 만드는 것입니다.

개별 구인글은 고유해서 같은 글이 반복되지 않기 때문에,
"어떤 고객이 어떤 글에 반응했는가"만으로는 선호를 일반화하기 어렵습니다.
세그먼트라는 중간 단위가 있어야
"고객 A는 상업 촬영형 글에 반응이 높다"처럼 유형 수준에서 해석할 수 있습니다.

따라서 이 프로젝트는 추천 모델 자체가 아니라,
매칭이나 푸시 알고리즘이 작동하기 위한 **기반 계층**입니다.
이 계층이 없으면 매칭·푸시·반응률 분석 모두 신뢰할 수 있는 단위 없이 돌아가게 됩니다.

### 3.4 왜 AI인가 — 규칙으로 잘리지 않는 경계

세그먼트 기준이 정량 조건(`지역=서울`, `금액=10만원 이하`)으로 깔끔하게 잘린다면
규칙 기반이 AI보다 단순하고 설명하기 쉽습니다.
모든 곳에 AI를 쓰는 것이 정답은 아닙니다.

그러나 구인글 스타일은 한두 조건으로 잘리지 않습니다.
같은 `서울 / 야외 / 10만원` 조건이라도 브랜드 광고인지 개인 포트폴리오인지에 따라
모델 입장에서는 전혀 다른 글입니다.
표현이 다양하고 경계가 흐린 영역에서 규칙 기반은
새 표현 추가, 규칙 충돌, 우선순위 모호성, 누락 케이스 같은 관리 비용이 계속 쌓입니다.

이 프로젝트에서 규칙과 AI는 서로 대체하는 관계가 아니라 **역할을 나눈 구조**입니다.
명확한 정량 값(촬영 유형, payment 그룹)은 규칙으로 분기하고,
그 위에 LLM이 추출한 스타일 속성을 얹는 혼합 방식을 씁니다.

### 3.5 해결 방향

1. LLM으로 구인글에서 **구조화된 속성 7개**(장소·목적·주제·시술 + 경력·지속성·긴급도)를 추출합니다.
2. Phase 3 LLM 속성 추출 후 **P3 결정적 보정**을 적용하고, **가중 Gower 거리 + average linkage 계층 군집**으로 확정 세그먼트를 도출합니다.
3. 확정 세그먼트 기반으로 **RF 분류기**를 학습합니다.
4. 신규 구인글에 대해 **자동 배정 서비스**(CLI batch/single + REST API)를 제공합니다.

세그먼트 체계는 한 번 확정되면 **버전으로 고정**합니다.
새 글이 올라올 때마다 전체를 재클러스터링하면 체계가 매번 흔들리고
이전 분류와의 비교가 불가능해지기 때문입니다.
실제 적용 시에는 신규 글 배정만 수행하고, 체계 노후화가 확인될 때만 새 버전을 만듭니다.

## 4. 버전 이력

|  | v1.0 | v2.0 → **v2.1-tune (현재)** |
|---|---|---|
| 분리 기준 | job_type (model / beauty / photo) | **job_type × payment_group** (9셀) |
| 속성 추출 | 키워드 + TF-IDF | LLM 구조화 속성 7개 |
| 클러스터링 | K-means (텍스트 임베딩) | P3 보정 → C-1 → 가중 Gower(6축) → average linkage(raw K) → dominant 병합 + micro 흡수 |
| 분류기 | TF-IDF / RoBERTa | **RandomForest** (속성 원-핫) |
| 세그먼트 수 | 타입별 7~10개 | **48개 운영 세그먼트** + 정보부족형 115건 |
| 서비스 형태 | — | CLI(batch/single/stats) + **FastAPI REST API** |

v1 코드(`block3/`, `block4/`, `m1_~m3_*.py`, `b3_~b4_*.py`, `schema_attrs.py`)는
의사결정 이력 비교용으로 유지하고 있으며, **현재 구현 코드는 `v2_*.py`** 입니다.

## 5. 파이프라인 (Task 0~7)

```
Task 0  DB 스키마 변경 + config.py (segment_id 컬럼 추가)
Task 1  카테고리 + Phase1 키워드 통합 데이터셋 생성
Task 2  LLM 키워드 → 차원 발견 + 유사어 정규화 → 스키마 확정
Task 3  Phase 3 — LLM 속성 추출 (100건 샘플 → 9,941건 배치)
Task 4  클러스터링 (v2.0 baseline → v2.1-tune 확정, 48세그)
Task 5  RF 분류기 학습 (8셀, v2.1-tune 라벨 기준 hold-out 재현 99~100% 확인)
Task 6  배정 서비스 — CLI(batch/single/stats) + FastAPI REST API
Task 7  [대기] 정기 배치 배포 — `v2_service --mode batch` 단발 실행으로 시연 가능
```

**처리 흐름 (Task 4 — v2.1-tune 클러스터링)**

```
Phase3 속성 (9,941건)
→ P3-헤어 / P3-스냅 / P3-목적 / P3-장소 (결정적 보정)
→ C-1: 4축 정보충실도 score=0 → 정보부족 (segment_id = -1)
→ 가중 Gower(6축) → average linkage(raw K, 셀별 스케일)
→ 1차 dominant 병합 + micro 흡수 (min 25건, photo는 시술 축 제외)
→ cluster_assignments_v21_tune.csv 저장
```

**처리 흐름 (Task 6 — 신규 글 배정 런타임)**

```
미배정 구인글 → job_type + payment → 셀 결정
→ DB 카테고리 조회
→ Phase 3 LLM 속성 추출 (Ollama)
→ C-1: 4축 전부 불명확(score=0) → segment_id = -1 (정보부족)
→ RF 예측 (rf_{pg}_{rt}.pkl)
→ DB UPDATE (segment_id, segment_version, segment_confidence, segment_assigned_at)
```

| Task | 모듈 | 역할 |
|---|---|---|
| 0 | `config.py` | DB 스키마 + payment 그룹 매핑 |
| 1 | `scripts/v2_prepare_dataset.py` | 카테고리 + Phase1 키워드 통합 |
| 2 | `scripts/v2_phase2_discovery.py` | LLM 차원 발견 + 유사어 정규화 |
| 3 | `scripts/v2_phase3_{core,sample,batch}.py` | LLM 속성 추출 (Ollama / Qwen2.5:14b) |
| 4 | `scripts/v2_clustering_v21.py` | v2.1-tune 확정 (P3 → C-1 → Gower → average linkage → 1차 병합) |
| 5 | `scripts/v2_rf_classifier.py` | 셀별 RandomForest 학습 |
| 6 | `scripts/v2_service.py` | CLI 배정 서비스 (batch / single / stats) |
| 6 | `scripts/v2_api.py` | FastAPI REST API (단건·배치·predict-only) |

## 6. 핵심 설계 결정

| 결정 | 이유 |
|---|---|
| `job_type × payment_group` 9셀 독립 클러스터링 | 모델/뷰티/포토 도메인, 페이 유형에 따라 구인 성격이 전혀 다름 |
| P3 결정적 보정 (클러스터링 직전) | LLM 환각·누락을 규칙으로 보정해 학습·추론 일관성 확보 |
| 가중 Gower 거리 (null-aware) | 범주형 + 불명확(결측) 처리 + 축별 변별력 차이를 가중치로 반영 |
| 정보충실도 선분리 (C-1) | 4축(장소·목적·주제·시술) 전부 불명확(score=0)이면 클러스터링·배정 모두 정보부족 처리 |
| 셀별 raw K + average linkage | Gower 거리 행렬 위 계층적 군집 — 셀마다 기본 K × 튜닝 스케일 적용 |
| 1차 dominant 병합 | raw 군집을 dominant merge_key 기준으로 통합, 25건 미만 소규모 흡수 |
| RF (텍스트 X, 속성 O) | 7개 범주형 속성 → 원-핫 → RF. 텍스트 분류보다 설명 가능성·재학습 비용이 우수 |
| Phase 3 재사용 | 학습과 추론에서 동일한 `extract_attributes()` 를 사용해 일관성 보장 |
| FastAPI REST API 분리 | 배치 CLI와 실시간 API 관심사 분리, DB 비의존 추론 가능 |

## 7. 핵심 구현 디테일

### 7.1 가중 Gower 거리 (null-aware)

v2.0 K-means에서는 `"불명확"`이 원-핫 벡터의 실제 값처럼 들어가면서,
글의 의미가 아니라 **어떤 정보가 비어 있는지**가 군집을 만드는 문제가 있었습니다.
v2.1에서는 Gower 거리 계산에서 결측을 별도 처리하고 축별 가중치를 적용합니다.

| 비교 케이스 | 처리 |
|---|---|
| 둘 다 명시·일치 | 거리 0 |
| 둘 다 명시·불일치 | 거리 1 |
| 한쪽만 명시 | 약한 거리 0.35, 가중치 0.5 |
| 둘 다 불명확 | 해당 차원은 거리 계산에서 제외 |

**축별 가중치** (`GOWER_DIM_WEIGHTS`, `CLUSTERING_6DIM` 순서):

| 차원 | 가중치 | 비고 |
|---|---|---|
| 촬영 장소 | 1.0 | |
| 촬영 목적 | 1.0 | |
| 촬영 주제 | 3.0 | 변별력 높음 |
| 시술 종류 | 3.0 | 변별력 높음 |
| 경력 조건 | 0.0 | merge_key로만 활용 |
| 작업 지속성 | 0.0 | merge_key로만 활용 |

핵심 구현은 `scripts/v2_gower.py`에 있고, 클러스터링 파이프라인에서는
`scripts/v2_clustering_v21.py`가 이 거리를 사용합니다.

### 7.2 정보충실도 선분리 (C-1)

클러스터링(Task 4)과 배정(Task 6) 모두 동일 기준을 씁니다.
촬영 장소·목적·주제·시술 **4축** 중 명시된 값이 하나도 없으면(score=0) `segment_id = -1`(정보부족)로 처리하고,
세그먼트 기반 추천 후보에서는 제외한 채 시간순 노출 경로로 유지합니다.

| status | DB 저장 | 의미 |
|---|---|---|
| `ok` | 저장 | RF 예측 성공 |
| `info_poor` | `segment_id = -1` 저장 | 4축 정보충실도 score=0 |
| `no_rf_model` | `segment_id = -1` 저장 | `photo×n3`처럼 RF 미학습 셀 |
| `phase3_fail` | NULL 유지 | LLM 추출 실패, 다음 배치에서 재시도 |
| `invalid_recruit_type` | NULL 유지 | 지원하지 않는 recruit type |

저신뢰도(`confidence < 0.7`)는 배정은 수행하되 `LOW_CONF`로 별도 로그합니다.

### 7.3 v2.1-tune 클러스터링 파이프라인

`scripts/v2_clustering_v21.py`의 `TUNE_CLUSTER_CONFIG`가 확정 파이프라인입니다.
기본 실행(`python scripts/v2_clustering_v21.py`) 시 아래 단계만 적용됩니다.

| 단계 | 내용 | 구현 |
|---|---|---|
| 1. P3 보정 | 헤어·스냅·목적·장소 결정적 보정 | `v2_phase3_core.apply_p3_corrections_phase3_frame` |
| 2. C-1 | 4축 score=0 → 정보부족(-1), 나머지만 클러스터링 | `info_density_row` + `INFO_POOR_THRESHOLD=1` |
| 3. Gower | 6축 null-aware 가중 거리 (GW1) | `v2_gower.gower_distance_matrix` |
| 4. average linkage | 셀별 raw K × 튜닝 스케일 → 계층적 군집 | `AgglomerativeClustering(linkage="average")` + `CELL_RAW_K` |
| 5. 1차 병합 | dominant merge_key 기준 통합, 25건 미만 흡수 | `v2_segment_ops.merge_clusters_by_key` |

**셀별 raw K (기본값 × `TUNE_CLUSTER_CONFIG` 스케일)**

| 셀 | 기본 K | 스케일 | 적용 K |
|---|---|---|---|
| model×n2 | 16 | 1.5 | 24 |
| model×n3 | 5 | 2.0 | 10 |
| model×pay | 7 | 1.5 | 10 |
| beauty×n2 | 5 | 2.5 | 12 |
| beauty×n3 | 3 | 2.0 | 6 |
| photo×n2 | 5 | 2.0 | 10 |
| photo×pay | 2 | 1.0 | 2 |
| photo×n3 | — | — | 단일 세그먼트 (HDBSCAN/v2.0과 동일) |

### 7.4 1차 dominant 병합

average linkage raw 군집을 `ClusterInfo.merge_key` 기준으로 통합합니다.

- **병합 키**: 목적·장소·주제 (+ model/beauty는 시술, + 경력·지속성은 명시값만)
- **dominant 규칙**: 축별 1등 값이 70% 이상이면 해당 값, 미달이면 `혼재`, 1등이 불명확이면 `불명확`
- **표시명**: merge_key에서 `불명확`·`혼재` 축은 segment_key에서 제외
- **소규모 흡수**: `min_segment_size=25` 미만 군집은 dominant 유사 키로 흡수

### 7.5 Phase 3 결정적 보정 (P3)

LLM 추출 결과를 클러스터링 직전에 규칙 기반으로 보정하는 단계입니다.
`schema_v2.py`에 정의된 보정 종류:

| 보정 | 역할 |
|---|---|
| P3-헤어 | 헤어 카테고리 + 시술 미기재 → `헤어 시술 미언급` (LLM 추론 차단) |
| P3-스냅 | 카테고리·본문 신호 → 촬영 주제 `스냅` (결정적 적용) |
| P3-목적 | 촬영 목적 결정적 보정 |
| P3-장소 | 촬영 장소 결정적 보정 |

P3-스냅은 Tier2(기본): `인물스냅` 카테고리 + 본문 스냅 키워드가 있을 때만 주제=스냅 적용.

### 7.6 실험용 후처리 (기본 OFF)

`ClusterRunConfig` 플래그로 켜는 experiments 단계입니다. **v2.1-tune 확정 결과에는 포함되지 않습니다.**

| 플래그 | 내용 | 비고 |
|---|---|---|
| `skip_coarse_merge=False` | 주제 혼재/불명확 축 2차 coarse 병합 | `--coarse` CLI |
| `apply_c2_absorb=True` | 목적·장소 dom 모두 혼재/불명확 세그 → -1 흡수 | C-2 / F-2A |
| `apply_f1_split=True` | 1,000건+ & 장소 불명확 25%+ 대형 세그 장소 split | F-1 |
| `apply_f1p_purpose_split=True` | 대형 세그 포트폴리오/비포트폴리오 split | F-1-P |
| `apply_f2b_rename=True` | dominant 기준 segment_key 재생성 | F-2B |

비교 스크립트: `v2_f1_postsplit_compare.py`, `v2_clustering_experiment.py`

### 7.7 DB 저장 컬럼

공개용 명칭 기준으로 신규 글 배정 결과는 다음 컬럼에 저장합니다.

```sql
ALTER TABLE job_postings
  ADD COLUMN segment_id          INT          NULL,
  ADD COLUMN segment_version     VARCHAR(10)  NULL,
  ADD COLUMN segment_confidence  FLOAT        NULL,
  ADD COLUMN segment_assigned_at DATETIME     NULL,
  ADD INDEX idx_segment_assignment (segment_id, is_deleted, is_expired);
```

MySQL은 partial index를 지원하지 않으므로 `segment_id IS NULL` 큐는
복합 인덱스의 leftmost column으로 처리합니다.

### 7.8 REST API (v2_api.py)

FastAPI 기반 실시간 배정 API입니다. DB 조회 없이 제목·본문·카테고리를 직접 받아 처리할 수 있습니다.

| Method | Endpoint | 역할 |
|---|---|---|
| POST | `/segment` | 단건 배정 (Phase3 LLM + RF) |
| POST | `/segment/batch` | 다건 배정 (최대 50건) |
| POST | `/segment/predict-only` | Phase3 속성이 이미 있을 때 RF만 호출 |
| GET | `/segments` | 전체 세그먼트 카탈로그 |
| GET | `/segments/{cell_key}` | 특정 셀의 세그먼트 목록 |
| GET | `/stats` | DB 기준 배정 통계 |
| GET | `/health` | 서비스 상태 확인 |

## 8. 추출 속성 스키마

운영 스키마는 `data/v2/schema_definition_v2.json`에 정의되어 있으며, 7개 속성으로 구성됩니다.

**Dimensions (클러스터링 6축 + RF 입력)**

| 속성 | 값 | Gower 가중치 |
|---|---|---|
| 촬영 장소 | 스튜디오 / 야외 / 홈스냅 / 불명확 | 1.0 |
| 촬영 목적 | 포트폴리오 / 뷰티·메이크업 / 프로필·증명사진 / 패션·룩북·화보 / 불명확 | 1.0 |
| 촬영 주제 | 웨딩 / 한복 / 커플 / 스냅 / 불명확 | 3.0 |
| 시술 종류 | 컷 / 펌 / 컬러 / 속눈썹 / 눈썹 타투 / 입술 타투 / 두피케어 / 헤어 시술 미언급 / 불명확 | 3.0 |
| 경력 조건 | 경력 무관 / 경력자 우대 / 불명확 | 0.0 (merge_key) |
| 작업 지속성 | 1회성 / 지속·정기 / 불명확 | 0.0 (merge_key) |

**Filter Tag (RF 입력, 클러스터링 미사용 — 1개)**

| 속성 | 값 |
|---|---|
| 긴급도 | 긴급 / 일반 / 불명확 |

## 9. 확정 클러스터링 결과 (v2.1-tune)

- **세그먼트**: 48개 운영 세그먼트
- **정보부족형**: 115건 (1.2%)
- **최대 세그먼트**: model×n2 최대 세그먼트 = 1,321건 (27.2%)
- **확정 파이프라인**: P3(헤어·스냅·목적·장소) → C-1 → 가중 Gower(6축, GW1) → average linkage(raw K) → 1차 dominant 병합 + micro 흡수(min 25)
- **RF 학습**: 8셀 학습, 1셀(photo×n3) SKIP. v2.1-tune 라벨 기준 hold-out 재현 정확도 99~100% 확인 (추천 품질 지표 아님)

RF 평가는 확정된 세그먼트 라벨을 신규 글 배정기가 얼마나 안정적으로 재현하는지 보는 지표입니다. 학습 입력은 Phase 3 속성 7개를 원-핫 인코딩한 값이고, 라벨은 v2.1-tune 클러스터링 결과의 `segment_id`입니다. 이 수치는 고객 반응률 개선을 의미하지 않으며, 비즈니스 유효성은 클릭률·지원률 조인 실험으로 별도 검증해야 합니다.

## 10. 디렉터리

```
job-segmenter/
├── README.md
├── config.py                   # 공통 설정 (DB, payment 매핑, 버전)
├── schema_v2.py                # v2 스키마 해석 공통 모듈 (6dim, 가중치, P3 보정)
├── schema_attrs.py             # v1 속성 정의 (참고용)
├── requirements.txt
├── scripts/
│   ├── v2_prepare_dataset.py   # Task 1
│   ├── v2_phase2_discovery.py  # Task 2 (차원 발견)
│   ├── v2_phase3_core.py       # Task 3 — extract_attributes (학습·추론 공용)
│   ├── v2_phase3_sample.py     # Task 3-A — 100건 샘플 검증
│   ├── v2_phase3_batch.py      # Task 3-B — 전체 배치 추출
│   ├── v2_clustering.py        # Task 4 — v2.0 K-means baseline
│   ├── v2_clustering_v21.py    # Task 4 — v2.1-tune 확정 (P3 → C-1 → Gower → average linkage → 1차 병합)
│   ├── v2_clustering_hdbscan.py  # HDBSCAN 실험 (미채택)
│   ├── v2_merge_hdbscan.py       # HDBSCAN 병합 실험 (미채택)
│   ├── v2_clustering_experiment.py  # 클러스터링 실험 (탐색)
│   ├── v2_gower.py             # null-aware 가중 Gower 거리
│   ├── v2_segment_ops.py       # dominant 계산·세그먼트 이름·병합
│   ├── v2_k_analysis.py        # K 결정 (Δsil elbow)
│   ├── v2_k_selection.py       # K 결정 (CH index)
│   ├── v2_tune_metrics.py      # 튜닝 지표 비교
│   ├── v2_rf_classifier.py     # Task 5
│   ├── v2_inference.py         # Task 6 — RF 추론 + C-1 판정
│   ├── v2_service.py           # Task 6 — CLI 배정 서비스 (batch/single/stats)
│   ├── v2_api.py               # Task 6 — FastAPI REST API
│   ├── verify_v2_api.py        # API 검증 스크립트
│   ├── v2_marketer_report.py   # 마케터 리포트
│   ├── v2_cluster_segment_examples.py  # 세그먼트 예시 문서 생성
│   ├── export_v21_segment_view.py      # 세그먼트 뷰 내보내기
│   ├── export_segment_catalog_json.py  # 세그먼트 카탈로그 JSON 내보내기
│   ├── v2_operational_viability.py     # 운영 가능성 분석
│   ├── v2_catchall_cell_compare.py     # catchall 셀 비교
│   ├── v2_f1_postsplit_compare.py      # post-split 전후 비교
│   ├── v2_generic_hair_simulation.py   # 헤어 시술 미언급 시뮬레이션
│   ├── v2_model_n2_seg1_compare.py     # model×n2 세그먼트 비교
│   ├── v2_p3_snap_tier_ab.py           # P3-스냅 Tier A/B 비교
│   │
│   └── (v1 참고용)
│       ├── m1_*.py / m2_*.py / m3_*.py    # v1 Phase 1~3
│       ├── b3_*.py / b4_*.py              # v1 Block3·4 (인코딩·RoBERTa·API)
│       └── cluster_examples_*.py          # v1 산출물 도큐먼트 생성
│
└── (v1 참고용)
    ├── block3/   # 인코딩·상수
    └── block4/   # RoBERTa 분류·API
```

원본 저장소의 `docs/`와 `data/v2/`는
민감한 데이터와 내부 운영 기록을 포함하고 있어 이 저장소에서는 제외했습니다.

## 11. 주요 의사결정 흐름 (코드로 추적)

설계·구현 기록 문서(`docs/`)는 비공개이므로,
주요 의사결정은 코드의 docstring과 주석에서 다음 순서로 확인할 수 있습니다.

1. **전체 파이프라인** → 본 문서 §5 + 각 `scripts/v2_*.py` 상단 docstring
2. **속성 스키마 의도** → `schema_v2.py` (6차원·필터태그 분리 근거, Gower 가중치, P3 보정 정의)
3. **LLM 추출 규칙** → `scripts/v2_phase3_core.py`의 `BASE_PROMPT`
   (시술/장소/목적/주제 추출 규칙 — v3~v7까지 반복 개선한 결과)
4. **클러스터링 v2.1-tune** → `scripts/v2_clustering_v21.py` (`TUNE_CLUSTER_CONFIG`, `ClusterRunConfig` 플래그)
5. **RF + 배정 서비스** → `scripts/v2_rf_classifier.py`, `v2_service.py`, `v2_inference.py`
6. **REST API** → `scripts/v2_api.py` (FastAPI 엔드포인트·에러 처리·lifespan)

## 12. 실행 (참고)

> 운영 데이터·모델·DB 접속정보가 포함되어 있지 않아 그대로 실행할 수는 없습니다.
> 코드 구조 확인을 위한 명령 예시입니다.

```bash
# 1) 의존성
pip install -r requirements.txt

# 2) Ollama 설치 + 모델 pull
ollama pull qwen2.5:14b

# 3) DB 설정 (택1)
#    a. local_config.py 생성:
#         DB_CONFIG = {"host": ..., "user": ..., "password": ..., "db": ...}
#    b. 또는 환경변수:
#         APP_MYSQL_HOST / APP_MYSQL_USER / APP_MYSQL_PASSWORD / APP_MYSQL_DB

# 4) 클러스터링 (v2.1-tune)
python scripts/v2_clustering_v21.py
# experiments — coarse 2차 병합 ON:
# python scripts/v2_clustering_v21.py --coarse

# 5) RF 재학습
python scripts/v2_rf_classifier.py

# 6) CLI 배정 서비스
python scripts/v2_service.py --mode stats
python scripts/v2_service.py --mode single --id <jobPostId>
python scripts/v2_service.py --mode batch --limit 500

# 7) REST API 서버
python scripts/v2_api.py --host 0.0.0.0 --port 8766
# → POST http://localhost:8766/segment  (단건)
# → POST http://localhost:8766/segment/batch  (다건, 최대 50건)
# → GET  http://localhost:8766/health
```

## 13. 기술 스택

| 영역 | 기술 |
|---|---|
| 언어 | Python 3.13 |
| LLM | Ollama (Qwen2.5:14b, self-hosted) |
| 클러스터링 | scikit-learn (AgglomerativeClustering, average linkage), 커스텀 가중 Gower |
| 분류기 | scikit-learn RandomForestClassifier |
| DB | MySQL (PyMySQL) |
| REST API | FastAPI + Uvicorn |
| v1 분류(참고) | TF-IDF, KLUE-RoBERTa (HuggingFace Transformers) |
