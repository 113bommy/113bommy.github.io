---
layout: post
title:  "Comparative Analysis of Machine Learning Models for Molecular Toxicity Prediction"
date:   2024-02-28 00:10:00 +0900
categories: [Project]
tags: [Project, Toxicity, Molecular]
---
# Tox21 독성 예측 프로젝트 — 타깃별 이진 분류 실험 기록

## I. 프로젝트 개요
이번 겨울 방학 동안 진행한 화학물질 독성 예측 프로젝트입니다.  
데이터는 Tox21 공개 데이터셋을 사용했고, **multi-label 구조**이지만 실제 학습은 **각 타깃별 독립적인 이진 분류**로 진행했습니다. 즉, 하나의 화합물에 대해 12개 타깃 각각의 독성 여부를 **0(비독성) / 1(독성)**로 개별 예측하는 구조입니다.

## II. 데이터 소개
- **원본 데이터**: `0.Data/tox21.xlsm` (시트명: "Tox21")
- **입력**: SMILES (분자 구조를 문자열로 표현한 형식)
- **타깃(총 12개)**:
  - **NR 계열** (핵 수용체 반응 관련):  
    NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma
  - **SR 계열** (스트레스 반응 경로 관련):  
    SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
- **라벨 값**:
  - `0` = 비독성
  - `1` = 독성
  - `NaN` = 실험 데이터 없음(결측)

## III. 타깃 의미
- **NR 계열**: 특정 호르몬/대사 경로에 관여하는 핵 수용체의 활성 여부를 측정합니다.  
  예) 안드로겐 수용체(AR), 에스트로겐 수용체(ER), 지방대사 관련 PPAR-γ 등
- **SR 계열**: 세포가 받는 스트레스 반응 경로의 활성 여부를 나타냅니다.  
  예) 산화 스트레스(ARE), DNA 손상 반응(p53), 열충격(HSE) 등

## IV. 전처리 과정
1. **EDA** (`1.DataDescription` 폴더)  
   - 레이블 불균형 확인  
   - 결측치 비율 분석  
   - SMILES 길이 및 분포 탐색
2. **전처리** (`2.DataPreprocessing/2-03.tox21_Preprocessing.ipynb`)  
   - 결측값 제거/대체  
   - 피처 스케일링  
   - RDKit 기반 Morgan Fingerprint 생성 → 화합물 특성 벡터화

## V. 모델 피팅
`3.ModelFitting` 폴더의 각 노트북에서 **타깃별로 독립적인 모델 학습**을 진행했습니다.  
이번 실험에서는 **`NR-AR`** 타깃을 중심으로 학습 및 비교를 수행했습니다.  
- **NR-AR**: 미토콘드리아 막전위(Mitochondrial Membrane Potential) 관련 독성 여부  
  - 세포 에너지 대사와 직결되는 중요한 독성 지표  
  - 불균형 데이터 특성이 있어 모델 성능 비교 시 주의가 필요

**실험한 모델 목록**:
- 로지스틱 회귀 (`3-01.Logistic_Regression.ipynb`)
- 결정 트리 (`3-02.Decision_Tree_Classifier.ipynb`)
- 랜덤 포레스트 (`3-03.Random_Forest_Classifier.ipynb`)
- 그래디언트 부스팅 (`3-04.Gradient_Boosting_Classifier.ipynb`)
- XGBoost (`3-05.XGB_Classifier.ipynb`)
- LightGBM (`3-06.LGBM_Classifier.ipynb`)
- LDA/QDA (`3-07`, `3-08`)
- PLS 회귀 기반 분류 (`3-09`)
- MLP (`3-10.MLP_Classifier.ipynb`)

모델 입력은 전처리된 분자 피처를 사용했고, 모델 출력은 **NR-AR 라벨만 추출**하여 사용했습니다.

## VI. 평가
- **평가 노트북**: `4.ModelEvaluation/4-1.Evaluation.ipynb`
- **평가 지표**: ROC-AUC, F1, Precision, Recall, PR-AUC
- **결과 요약**:
  - 트리 기반 부스팅 모델(XGB, LGBM)이 안정적인 성능을 보임
  - MLP는 하이퍼파라미터 조정 시 경쟁력 확보 가능
  - 데이터 불균형 영향으로 F1보다는 ROC-AUC 차이를 주요 지표로 활용

## VII. 향후 발전 가능성
NR-AR 타깃 하나만 집중해서 진행해도 모델 간 성능 차이가 꽤 뚜렷했습니다.  
다음 단계에서는:
- **멀티태스크 모델**로 타깃 간 상관성을 활용
- **GNN(Graph Neural Network)**으로 분자 그래프 직접 학습
등이 가능해보입니다.
---

## 실행 가이드
1. `0.Data/tox21.xlsm` 준비
2. `1.DataDescription` → `2.DataPreprocessing` 순서로 노트북 실행
3. `3.ModelFitting`에서 **NR-AR 라벨** 선택 후 학습
4. `4.ModelEvaluation`으로 성능 비교


## 해당 프로젝트 깃허브 링크입니다.

[![Repo Card](https://github-readme-stats.vercel.app/api/pin/?username=113bommy&repo=Chemical-Safety-Management-Internship-2023-Winter-UOS&theme=default)](https://github.com/113bommy/Chemical-Safety-Management-Internship-2023-Winter-UOS)