# 딜레이 완화 아키텍처 비교 — 실험 설계서

**작성**: 2026-06-15 · **상태**: 설계 확정 · 사인오프 완료 · 구현 착수

---

## 1. 연구 질문

액션 딜레이가 있는 환경에서, **에이전트에 어떤 obs/action 히스토리 구조를 주면 딜레이로 잃은 성능을 회복하는가?** 그리고 그 회복은 **딜레이 크기에 따라 어떻게 달라지는가?**

부가 질문: 같은 아키텍처의 **delay=0 모델에서 warm-start하면** 지연 task 학습이 빨라지는가(scratch 대비)?

핵심 구조: **obs 히스토리 × action 히스토리의 2D ablation.**

---

## 2. 두 개의 축

### 축 1 — 환경 (액션 딜레이), 6수준
`delay k ∈ {0, 3, 6, 10, 15, 20}`  (naive baseline 실험과 동일 축)

### 축 2 — 모델 아키텍처, 5종

| # | 모델 | obs 입력 | action 입력 | 알고리즘 / 정책 |
|---|------|---------|-------------|-----------------|
| 1 | **naive** | o_t (1장) | — | PPO `CnnPolicy` (feedforward) — 대조군, **855 재사용** |
| 2 | **GRU** | o_t (1장) | — | `RecurrentPPO` `CnnLstmPolicy` (순환 메모리) |
| 3 | **ActStack** | o_t (1장) | a_{t-20..t-1} (20개) | PPO `MultiInputPolicy` |
| 4 | **ObsStack** | o_{t-19..t} (20장) | — | PPO `CnnPolicy` + `VecFrameStack(20)` |
| 5 | **Both** | o_{t-19..t} (20장) | a_{t-20..t-1} (20개) | PPO `MultiInputPolicy` |

**2D ablation 관점:**

| | act 없음 | act 20개 |
|---|:---:|:---:|
| **obs 1장** | naive | ActStack |
| **obs 20장** | ObsStack | Both |

별도 probe: **GRU** = o_t + LSTM 순환 (naive에 순환 메모리만 추가 — 명시적 스택 없이 회복되는지).

규약·결정사항:
- **action 인덱스**: 정책이 a_t를 결정하는 시점이라 입력엔 a_t(현재)를 못 넣는다. "최근 N개 action" = 가장 최근 a_{t-1}부터 뒤로 N개(a_{t-N..t-1}).
- **스택 크기 고정 20**: obs·action 스택은 delay와 무관하게 항상 20 (20 ≥ 최대 delay → 미적용 action 버퍼 전체 포함). 따라서 아키텍처는 delay에 불변, 환경 delay만 바뀐다.
- **GRU만 순환(LSTM)**: o_t만 입력받고 LSTM 은닉상태로 히스토리를 암묵적으로 압축(**action 입력 없음**, stock `CnnLstmPolicy`). 나머지(ActStack/ObsStack/Both)는 feedforward로 raw 스택을 직접 입력. naive는 둘 다 없음.
- **GRU→LSTM**: sb3-contrib에 GRU 없음 → 동일 계열 게이트 RNN인 LSTM. 보고 시 "GRU(LSTM)" 표기.

---

## 3. 실험 그리드 (6 × 5 = 30셀)

| 모델 ＼ delay | 0 | 3 | 6 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|
| **naive** | 855 ✓ | 784 ✓ | 533 ✓ | 362 ✓ | 208 ✓ | −54 ✓ |
| **GRU** | T₀ | T | T | **T**\* | T | T |
| **ActStack** | T₀ | T | T | **T**\* | T | T |
| **ObsStack** | T₀ | T | T | **T**\* | T | T |
| **Both** | T₀ | T | T | **T**\* | T | T |

`T₀` = delay=0 학습(scratch; 상위 delay의 pretrain 소스) · `T` = 신규 학습(scratch + pretrain) · `*` = Phase 1 파일럿(delay=10)

### 재사용으로 절약되는 셀
- **naive 행** = 딜레이 없이 학습된 기존 855를 delay축으로 평가 (이미 완료, `delay_converged.json`)
- **naive의 delay=0** = 855 그대로

> 비-naive 아키텍처는 delay=0에서도 naive와 **구조가 달라**(MultiInput/Recurrent) 855를 재사용할 수 없다. 각 아키텍처를 delay=0에서 별도 학습(T₀)하며, 이는 상위 delay의 pretrain 소스로도 쓰인다.

### 실제 신규 학습 수
```
delay=0 소스:  4 arch × 1            =  4
scratch:       4 arch × 5 delay       = 20
pretrain:      4 arch × 5 delay       = 20   (각 delay를 해당 arch의 delay=0에서 warm-start)
합계 = 44 학습
```

---

## 4. 초기화 2종 (scratch / pretrain)

| init | 정의 |
|------|------|
| **scratch** | 무작위 초기화부터 delay=k 학습 |
| **pretrain** | 같은 아키텍처의 **delay=0 모델**(T₀)에서 warm-start 후 delay=k 학습 |

**pretrain** — 각 비-naive 아키텍처를 먼저 delay=0에서 scratch 학습(T₀)하고, 그 가중치를 상위 delay 학습의 초기값으로 쓴다. 스택 크기가 delay와 무관하게 20으로 고정이라 **아키텍처(관측·정책 구조)가 delay에 불변** → 완전 가중치 로드 가능, shape 불일치 없음.

> 이전의 "855 warm-start"는 폐기. naive(`CnnPolicy`)와 다른 아키텍처(MultiInput/Recurrent)는 구조가 달라 가중치 호환이 안 된다. 같은 아키텍처의 delay=0 pretrain이 올바른 방식.

목적: scratch vs pretrain **수렴 속도(샘플 효율)** 비교 — delay=0 사전학습이 지연 task 학습을 가속하는가.

---

## 5. 학습 수렴 정의 (stop rule)

- 50k 스텝마다 eval(20 에피소드, deterministic)
- `StopTrainingOnNoModelImprovement(max_no_improvement_evals=8, min_evals=10)` — best eval 8회 연속(400k步) 미갱신 시 정지
- 안전 상한 **5M 스텝**, 8 병렬(SubprocVecEnv), TITAN RTX
- 셀별 best_model 저장: `models/variants/<arch>/<init>/d<delay>/`

---

## 6. 평가 프로토콜

- 각 셀 best_model을 **그 셀의 delay k에서** SEM 기반 수렴 평가(SEM<5.0, 8병렬) → 평균 ± 95% CI + 실패율(보상<0)
- 스택 크기 고정 20이라 아키텍처는 delay 불변이지만, 학습·평가는 셀의 delay에서 수행

---

## 7. 측정 지표

1. **성능**: 수렴 보상 평균 ± 95% CI
2. **완화 효과**: 같은 delay의 naive 대비 회복량 (예: delay=10 naive 362 대비 +Δ)
3. **실패율**: 보상<0 비율
4. **샘플 효율**: 수렴까지 스텝 수 (scratch vs pretrain)
5. **2D 효과 분해**: obs 스택 효과(naive↔ObsStack) vs action 스택 효과(naive↔ActStack) vs 결합(Both) vs 순환(naive↔GRU)

---

## 8. 컴퓨트 & 실행 순서

**총 44 학습 × 수렴(~3–4h) ≈ 5–6일** 연속 GPU (순차, TITAN RTX, 8 병렬).

- **Phase 0**: delay=0 소스 · scratch · {GRU, ActStack, ObsStack, Both} = 4 학습
  → delay=0 열 결과 + pretrain 소스(T₀) 확보
- **Phase 1 (파일럿)**: delay=10 · scratch · 4 arch = 4 학습
  → 모델축 전체를 한 delay에서 비교 + naive(362) 대비 효과 1차 확인
- **Phase 2 (파일럿)**: delay=10 · pretrain · 4 arch = 4 학습
  → scratch vs pretrain 샘플 효율 비교
- **Phase 3**: 나머지 scratch — delays{3,6,15,20} × 4 arch = 16 학습
- **Phase 4**: 나머지 pretrain — delays{3,6,15,20} × 4 arch = 16 학습

각 Phase 종료마다 중간 비교 보고. 순서·범위는 신호 보고 조정.

---

## 9. 산출물 · 경로 규칙

```
models/variants/<arch>/<init>/d<delay>/best_model.zip     # 셀별 best
runs/variants/<arch>/<init>/d<delay>/                      # monitor/eval 로그
runs/tb_logs/<arch>_d<delay>_<init>/                       # TensorBoard
results/variant_compare.png · variant_results.json         # 최종 비교
```
셀 식별자: `<arch>_d<delay>_<init>` (예: `gru_d10_scratch`, `both_d20_pretrain`)

---

## 10. 구현 메모

- **AugmentDelayWrapper(env, delay, n_obs, n_act)**: ObsActDelayWrapper로 딜레이 적용 + 최근 n_obs 프레임 채널 concat + 최근 n_act emit action flatten → `Dict({"image", "act_hist"})` 반환. **ActStack=(n_obs=1, n_act=20)·Both=(n_obs=20, n_act=20)** 두 경우만 사용.
- **GRU**: stock `RecurrentPPO`(`CnnLstmPolicy`) — o_t만 입력, LSTM 은닉상태가 히스토리를 담당(action 입력·커스텀 파라미터 없음).
- **ObsStack**: `CnnPolicy` + `VecFrameStack(20)` (60채널) — action 없음, Dict 불필요.
- **ActStack/Both**: `MultiInputPolicy`(CombinedExtractor: image→NatureCNN, act_hist→MLP).
- **naive**: `CnnPolicy`, 단일 프레임 — 재사용(855), 재학습 안 함.

---

## 11. 한계 · 주의

- **GRU→LSTM 대체** (sb3-contrib 한정).
- 모든 스택 모델은 **delay별 별도 학습** (환경 동역학이 delay에 종속). 아키텍처 자체는 delay 불변.
- delay=20은 naive 85% 실패 — 완화 모델도 학습 난이도 높음(미수렴 가능, 상한 5M 종료).
- PPO 진동 특성상 best_model(plateau 직전 최고점) 기준 비교.

---

*코드*: `src/train_variant.py` · `src/wrappers.py`(AugmentDelay) · `src/eval_variant.py` · `src/common.py`
