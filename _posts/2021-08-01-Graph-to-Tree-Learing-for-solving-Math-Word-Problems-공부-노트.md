---
layout: post
title: Graph-to-Tree Learing for solving Math Word Problems 공부 노트
published: false
---
# Abstract  
최근의 트리 기반 신경망 모델들은 MWP의 풀이표현 생성에 대해 유망하지만, 이들 대부분 모델은 관계성과 수량 정보간의 순서를 잘 포착하지 못함.     
이는 잘못된 수량표현과 부정확한 풀이 표현을 초래함.     
본 논문에서는 Grpah2Tree 라는 그래프 기반 인코더와 트리 기반 디코더의 장점을 결합해 더 나은 정답표현을 생성하는 새로운 딥러닝 아키텍처를 제시함. Graph2Tree 프레임워크는 MWP에서의 관계성과 순서 정보의 효율적인 표현을 통해 기존 방법의 한계를 해결하도록 설계된 Quantity Cell Graph 와 Quantity Comparison Graph 두 가지 그래프를 포함.     
두가지 데이터 셋에 대해 광범위한 실험을 함. 실험결과는 Grpah2Tree가 두 데이터 셋에서 첨단의 baseline을 능가하는 것을 명확히 보여줌    

# 1. Introduction
MWP는 일반적으로 문제에 대해 설명하고 미지수에 대해 질문하는 짧은 서술로 구성. 초기 연구는 통계적 머신러닝이나 시멘틱 파싱으로 접근. 하지만 이런 방법들은 특성과 표현 형식을 설정하는데 엄청난 노력이 필요해서 non scalable.    
근 몇년, 딥러닝 기반 모델들이 MWP해결을 위해 개발되는 중. Wang et al.(2017) 에서 큰 규모의 MWP 데이터 셋을 제시하고 생짜 seq2seq를 적용. 어떤 연구진은 내재된 또는 외재된 트리를 통해 seq2seq의 표현 생성을 개선하는 것을 제안. 수량의 표현을 개선하는 것은 더 나은 풀이 표현을 위한 잠재적 접근. 기존의 모델은 문제 내의 관계성과 수량 표현을 잘 포착하지 못하므로 최종결과가 부정확해진다.    
수량 표현의 향상을 위해 수량과 관계있는 설명문간의 관계가 모델화 될 필요 있으나 이런 관계성은 흔히 사용되는 방식인 재귀적 모델에선 효과적으로 모델링안됨. Quantity schema와 Qset에서 영감을 받아, 수량과 쓸모있는 설명문을 연결시키는 Quantity Cell Graph를 설계. 먼저 관련된 명사 동사 형용사 단위 비율을 설명문으로부터 추출. 그 다음 추출된 단어가 수량에 직접 연결된 이웃 노드로 표현한 그래프를 만듬. 마지막으로 신경망 모델이 만들어진 Quantity Cell Graph에 기반한 숨은 수량 표현의 학습에 사용됨    
MWP의 기존 방법에서 수량의 수치적 품질 손실도 구린결과 만들 수 있음. 이러한 제한을 해결하기 위해 수치적 기계 독해력 모델에서 영감을 받은 Quantity Comparison Graph 제시. Quantity Comparison Graph.의 직관은 수량의 수치적 품질을 유지하고, 풀이 표현이 실제 산술 순서를 더 반영하도록 하는 관계를 표현하는 휴리스틱을 활용한다.    
정답 표현 생성과정도 개선하려고 함.

# 2. 문제 수식화
p 수학 문장 문제 텍스트 -> 문자 토큰과 수치값으로 이루어진 시퀀스    
Vp = {v1, · · · , vm} 단어 토큰    
nP = {n1, · · · , nl} 숫자 값의 집합     
MWP를 푸는 것: 문제의 값들을 이해하고 그 값들 간의 복잡한 수학적 관계를 이해하는 것    
P는 의존성 파싱이나 품사 태깅같은 구조적 정보로 문자열을 보충한 그래프 G로 변환된다    

트리 T 구성: 상수, 연산자, nP의 값    
트리 T 출력으로 나오는 수학 표현 Ep    
Vcon: 특수한 상수 eg. 1, 파이    
Vop: +, - , *, /    
Vdec: Vcon+Vop+nP    
모델의 목표: P(Ep|P) = P(T|G, Vdec)    

# 3. Methodology
크게 봤을 때
BiLSTM으로 텍스트 인풋 인코딩 BiLSTM 의 단어 수준 출력을 노드 표현으로 사용
텍스트에서 수치를 추출해서 Quantity cells 생성하고 이로 부터 Quantity cell graph와 Quantity comparison graph 생성
그래프 트랜스포머에서 mutliGCN부분에 앞에서 만든 Quantity Cell Graph 와 Quantity Comparison Graph과 노드표현을 입력으로 전달
트리 베이스 디코더에서 목표 수식 Eq 생성
## 3.1. Graph-Based Encoder
기존의 graph transformer model의 구조에서 따옴.
### 3.1.1 Node Representation Initialization
노드 표현 초기화를 위해서 MWP를 인풋으로 하는 BiLSTM의 단어 수준 은닉층 값을 사용
### 3.1.2 Quantity Cell
np Vp를 그래프의 노드로 간주    
Quantity Cell := 값에 관련된 노드들의 부분 집합 QC = {Q1, Q2, · · · , Qm}    
Qi := {ni, v1i, .. , vqi}    
Qi를 구성하는 것들: 값, 관련 명사, 관련 형용사, 관련 동사, 단위와 비율    
### 3.1.3 Quantity Graph Construction
Quantity Cell Graph := Qi = {ni} ∪ {v1i , · · · , vni} ni와 각 vji 사이에 무향 에지    
Quantity Comparison Graph := ni , nj 큰 수에서 작은 수로 유향 에지    
두 그래프는 연결이 있으면 1 없으면 0인 인접행렬로 표현    
### 3.1.4 Graph Transformer
입력으로 두 그래프의 인접행렬과 초기 노드 표현    
GCN 연산 GCN(Ak, X) = GConv2(Ak, GConv1(Ak, X))    
GConv(Ak, X) = relu(AkXTWgk)    
Z = || GCN(Ak, H)	||:=concat.    
residual connection, layernorm     
Zˆ = Z + LayerNorm(Z)     
Z⁻ = Zˆ + LayerNorm(F F N(Zˆ))    
F F N(x) = max(0, xWf1 +bf1)Wf2 +bf2    
zg = F C(M inP ool(Z))    
## 3.2 Tree-Based Decoder
Goal-driven tree 구조(GTS)에서 따옴    
수량 노드를 잎으로 연산 노드는 무조건 두개의 자식노드를 갖도록 하는 트리    
제일 중앙의 연산자를 만들고, 왼쪽부터. 잎 노드가 생성되는 동안 반복    
### 3.2.1 Tree Initialize
context vector zg에 따라 루트 노드 qroot 생성.    
노드 종류 세가지: 연산자, 상수, P에 포함되는 숫자. 상수와 np의 숫자는 리프 노드에 있어야만 함    
### 3.2.2 Pre-Order Tree Generation
* 스텝1: qroot 만 있는 트리로 시작. GTS Attention 모듈로 노드 임베딩 Z⎺을 전역 그래프 벡터 Gc로 인코드    
Gc = GTS-Attn(qroot, Z⎺)    
* 스텝2: 왼쪽 자식 노드 생성    
ql = GTS-Left(qp, Gc)    
y^ = GTS-Predict(ql, Gc)    
y^이 연산자라면 빈 자식노드 두개를 생성하여 스텝2를 반복함.
상수 또는 np의 숫자라면 스텝3
* 스텝3: 오른쪽 자식노드 생성
qr = GTS-Right(qt, Gc, tl)
y^r = GTS-Predict(qr, Gc)
tl = GTS-Subtree(y^l, ql)
y^r이 연산자라면 스텝2로, 숫자라면 스텝4로
* 스텝4: 백트래킹해서 비어있는 오른쪽 노드를 찾음. 만약 빈 노드가 없다면 생성 종료. 빈 노드를 찾으면 스텝2로
### 3.3 Model Learing
L(T, P) = −logprob(yt |qt , Gc, P)
