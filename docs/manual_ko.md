# SLM 프로젝트 학습 매뉴얼 (입문자용)

이 문서는 **AI/딥러닝을 처음 공부하는 사람**이 이 저장소를 통해
`Transformer 언어모델을 바닥부터 구현`하는 과정을 이해하도록 돕기 위한 안내서입니다.

---

## 1. 이 프로젝트는 무엇인가?

이 프로젝트의 목표는 다음입니다.

- PyTorch/TensorFlow/JAX 같은 대형 프레임워크 없이
- NumPy만 사용해서
- 언어모델의 핵심 구성요소를 직접 구현하기

즉, "라이브러리를 쓰는 법"이 아니라
**딥러닝 엔진이 어떻게 동작하는지**를 직접 만들어 보며 이해하는 프로젝트입니다.

---

## 2. 왜 이런 프로젝트가 필요한가?

일반적으로 딥러닝 프레임워크를 쓰면 빠르게 모델을 만들 수 있지만,
다음 질문에 답하기는 어렵습니다.

- 역전파(backpropagation)는 실제로 어떻게 계산되는가?
- 왜 gradient가 누적되어야 하는가?
- matmul/add/relu 같은 기본 연산만으로 네트워크가 어떻게 학습되는가?

이 프로젝트는 그 질문을 해결합니다.

- `Tensor + autograd`를 직접 구현해 역전파 원리를 체득
- gradcheck(유한차분)로 수학적으로 구현이 맞는지 검증
- 이후 Layer/Attention/Transformer로 확장할 기반 확보

---

## 3. 현재까지 구현된 범위 (Milestone 1 완료)

Milestone 1 요구사항:

- `src/tensor/tensor.py`
  - `Tensor(data, requires_grad)`
  - 연산: `add`, `mul`, `matmul`, `sum`, `mean`, `relu`
  - `backward()` (reverse topological order)
- `tests/test_gradcheck.py`
  - 각 연산의 gradient를 유한차분으로 검증
- `demos/demo_mlp.py` (선택)
  - 작은 MLP를 학습시켜 loss 감소 확인

추가로 학습/운영 편의를 위해:

- `tests/test_tensor_autograd.py` (gradient 누적 검증)
- `pyproject.toml` (의존성/pytest 설정)
- `.github/workflows/ci.yml` (push/PR 시 자동 테스트)

---

## 4. 핵심 개념 정리 (처음 배우는 사람용)

### 4.1 Tensor

Tensor는 숫자를 담는 상자입니다.

- 스칼라: `3.14`
- 벡터: `[1, 2, 3]`
- 행렬: `[[1, 2], [3, 4]]`

딥러닝에서 모든 데이터와 가중치는 Tensor로 표현됩니다.

### 4.2 Gradient(기울기)

Gradient는 "파라미터를 어느 방향으로 얼마나 바꾸면 loss가 줄어드는지" 알려주는 값입니다.

- 값이 크면 크게 수정
- 값이 작으면 조금 수정
- 부호(+/-)에 따라 증가/감소 방향 결정

### 4.3 Autograd(자동미분)

Autograd는 연산 과정을 기록해 두었다가,
나중에 체인룰(chain rule)로 gradient를 자동 계산하는 시스템입니다.

### 4.4 Backpropagation

순전파(forward): 입력 -> 출력 -> loss 계산  
역전파(backward): loss에서 거꾸로 내려가며 gradient 계산

### 4.5 Reverse Topological Order가 왜 필요한가?

그래프에서 "부모 연산보다 자식 연산의 gradient가 먼저 계산"되어야
올바른 chain rule이 적용됩니다.

그래서:

1. 그래프를 DFS로 순회해서 위상 순서를 만들고
2. 그 순서를 뒤집어 역전파

를 수행합니다.

---

## 5. 코드로 보는 구현 설명

아래 설명은 `src/tensor/tensor.py` 기준입니다.

### 5.1 Tensor가 저장하는 정보

- `data`: 실제 숫자 값 (`np.ndarray`)
- `requires_grad`: gradient 계산 대상인지 여부
- `grad`: 누적 gradient 저장소
- `_prev`: 이 Tensor를 만들 때 사용한 입력 Tensor들
- `_backward`: 현재 연산의 로컬 미분 규칙 함수
- `_op`: 디버깅용 연산 이름

핵심 아이디어:

- forward 시에는 값을 계산하고
- backward 시에 쓸 "미분 함수(_backward)"를 함께 저장합니다.

### 5.2 gradient 누적(`+=`)이 중요한 이유

하나의 Tensor가 여러 경로에서 재사용되면,
각 경로에서 오는 gradient를 합쳐야 합니다.

예: `y = x*x + x`  
`dy/dx = 2x + 1` 이므로 두 경로의 기여가 더해져야 합니다.

코드에서는 `_accumulate_grad()`가 이를 처리합니다.

### 5.3 각 연산의 미분 규칙

- `add`: `d(a+b)/da = 1`, `d(a+b)/db = 1`
- `mul`: `d(a*b)/da = b`, `d(a*b)/db = a`
- `matmul`: 행렬미분 규칙 사용
  - `dL/dA = dL/dOut @ B^T`
  - `dL/dB = A^T @ dL/dOut`
- `sum`: 출력 gradient를 입력 shape로 broadcast
- `mean`: `sum / N` 이므로 `sum` gradient에 `1/N` 곱함
- `relu`: 입력이 0보다 큰 위치만 gradient 통과

### 5.4 broadcasting gradient 처리

forward에서 broadcasting이 일어나면,
backward에서 gradient shape를 원래 입력 shape로 되돌려야 합니다.

`_unbroadcast()`가 다음을 수행합니다.

- 차원 수가 늘어난 축을 sum
- 원래 크기가 1이었던 축을 keepdims sum

---

## 6. 테스트가 무엇을 보장하는가?

### 6.1 gradcheck (`tests/test_gradcheck.py`)

아이디어:

- 해석적 미분(autograd 결과)
- 수치 미분(유한차분)

두 값이 거의 같아야 구현이 맞습니다.

유한차분 공식:

`f'(x) ≈ (f(x+eps) - f(x-eps)) / (2*eps)`

검증 대상:

- add, mul, matmul, sum, mean, relu

### 6.2 누적 검증 (`tests/test_tensor_autograd.py`)

`x`가 여러 경로에서 재사용될 때 gradient가 합산되는지 확인합니다.

---

## 7. 사용 방법 (처음부터 따라하기)

### 7.1 환경 준비

- Python 3.11+

### 7.2 의존성 설치

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 7.3 테스트 실행

```bash
python -m pytest -q
```

정상이라면 모든 테스트가 통과합니다.

### 7.4 데모 실행

```bash
python demos/demo_mlp.py
```

출력 예시(학습 진행):

- `step=000 loss=...`
- `step=100 loss=...`
- ...

loss가 점점 감소하면 autograd + 파라미터 업데이트가 정상 동작하는 것입니다.

---

## 8. 파일 구조 설명

- `src/tensor/`
  - autograd 핵심 엔진
- `tests/`
  - 수치검증/동작검증
- `demos/`
  - 학습 예시 실행 스크립트
- `src/nn`, `src/attention`, `src/transformer`, `src/tokenizer`, `src/training`, `src/sampling`
  - 이후 마일스톤에서 채워질 영역

---

## 9. 다음 단계에서 배우게 될 것

Milestone 2부터는 보통 아래 순서가 자연스럽습니다.

1. `nn` 기본층 구현
   - Linear, LayerNorm, Embedding, Dropout
2. attention 구현
   - scaled dot-product, causal mask, multi-head
3. transformer block 구현
4. tokenizer + training loop + sampling

이 순서는 "작은 부품 -> 조립 -> 학습 -> 생성" 흐름이라 학습 효율이 높습니다.

### 현재 진행 상황 업데이트

`nn`의 첫 구성요소로 `Linear` 레이어가 추가되었습니다.

- 파일: `src/nn/linear.py`
- 수식: `y = xW + b`
- 테스트: `tests/test_nn_linear.py`
  - 출력 shape 검증
  - weight/bias gradcheck
  - 학습 시 loss 감소 검증
- 데모: `demos/demo_linear_regression.py`
  - 선형 회귀 문제를 직접 학습
  - step별 loss 감소 출력

---

## 10. 자주 묻는 질문 (FAQ)

### Q1. 이미 PyTorch가 있는데 왜 굳이 직접 구현하나요?

직접 구현하면:

- 역전파 원리를 정확히 이해
- 디버깅 능력 향상
- 논문 수식을 코드로 바꾸는 감각 강화

결과적으로 PyTorch를 더 잘 쓰게 됩니다.

### Q2. 수치미분은 느린데 왜 쓰나요?

느리지만 "정답 검증기"로 매우 유용합니다.
초기 엔진 구현 단계에서는 속도보다 정확성이 중요합니다.

### Q3. 지금 상태가 제품 배포 가능한가요?

아직은 학습/연구용 기반 단계입니다.
하지만 CI/패키징 기초가 준비되어 있어 확장하기 좋은 상태입니다.

### Q4. Linear 레이어는 왜 먼저 구현하나요?

Linear는 거의 모든 딥러닝 모델의 기본 블록입니다.
Attention, MLP, Transformer 모두 내부적으로 Linear를 사용하므로
이 레이어를 먼저 안정적으로 만들면 이후 구현 속도가 빨라집니다.

---

## 11. 학습 루틴 추천

아래 루틴을 반복하면 이해가 빠릅니다.

1. 연산 하나(add/mul 등) 수식으로 미분 직접 해보기
2. `tensor.py`에서 해당 `_backward` 확인
3. `test_gradcheck.py` 실행해 수치검증 확인
4. `demo_mlp.py`로 학습 loss 감소 관찰
5. 작은 코드 변경 후 테스트가 깨지는지 확인

핵심은 "**수식 -> 코드 -> 테스트 -> 학습 결과**"를 한 사이클로 묶는 것입니다.
