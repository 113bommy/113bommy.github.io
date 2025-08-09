---
layout: post
title:  "Satellite Image Multi‑label Classification"
date:   2024-01-30 00:10:00 +0900
categories: [Project]
tags: [Project, Multi-Label, CV]
---

## 1) 프로젝트 개요
대회 베이스라인을 넘기 위해 **전처리별 성능 차이**에 집중한 멀티라벨 분류 프로젝트입니다.  
한 장의 위성 이미지에서 **최대 60개 라벨을 동시에 예측**하며, 제출 포맷은 각 라벨의 **확률값**입니다.

- 데이터 크기: train 65,496 / test 43,665
- 라벨 수: 60 (예: trees, pavement, buildings, water, road, airplane, airport …)
- 불균형: 예를 들어 `trees`는 약 65% 양성, `football field` 등은 ~1% 수준으로 매우 희소

## 2) 베이스라인 파이프라인
- **백본**: ResNet50
- **헤드/출력**: `Linear(1000 → 60)` → `Sigmoid`
- **손실**: `BCELoss` (라벨별 독립 이진 분류)
- **입력 크기**: 224×224
- **옵티마이저**: Adam(lr=3e-4), batch 32, 20 epochs
- **스플릿**: 8:2 학습/검증(hold-out)
- **변환**: ToTensor → Resize(224) → Normalize(Imagenet mean/std)
- **제출**: 라벨별 확률을 CSV로 저장(60개 컬럼)

## 3) 전처리 실험
### A. Raw (baseline_raw.ipynb)
- RGB 원본 + 기본 정규화
- ResNet50 + Sigmoid + BCELoss

### B. Grayscale (baseline_grayscale.ipynb)
- 이미지를 1채널로 변환 → ResNet 첫 Conv를 `in_channels=1`으로 교체
- 사전학습 이점이 사라져 **성능 하락** 경향 (from scratch로 재학습)

### C. Sharpening (baseline_sharpening.ipynb)
- `cv2.filter2D`로 **샤프닝 커널** 적용 후 학습 → **검증 손실 및 점수 개선**
```python
kernel = np.array([[ 0, -1,  0],
                   [-1,  5, -1],
                   [ 0, -1,  0]], dtype=np.float32)
sharpened = cv2.filter2D(img, ddepth=-1, kernel=kernel)
```

## V. 결과 요약
- 전처리 성능 순위: **Sharpening > Raw > Grayscale**
- Sharpening 적용 시 mAP/PR-AUC 전반적인 향상
- Raw 대비 Sharpening은 명암 대비를 강화해 미세한 객체 경계 인식이 개선됨
- Grayscale은 색상 정보 손실과 pretrained weight 효과 감소로 성능 하락
- 상위 성능 모델들의 예측 확률을 **앙상블(평균)** 하여 최고 점수 달성

## VI. 인사이트
- 멀티라벨 분류는 **Softmax** 대신 **Sigmoid + BCELoss** 조합이 필수
- 클래스 불균형이 심한 경우, 정확도(Accuracy)보다 **PR-AUC / mAP** 모니터링이 더 중요
- 간단한 전처리(Sharpening)만으로도 모델의 특징 추출 능력 향상 가능
- Grayscale 변환은 색상 의존도가 낮은 문제에서만 유리할 수 있음

## VII. 다음 단계
- **백본 확장**: ConvNeXt, EfficientNetV2, CoAtNet 등 최신 아키텍처 적용
- **손실 함수 변경**: Focal Loss, Asymmetric Loss로 클래스 불균형 완화
- **데이터 증강**: RandAugment, CutMix/Mixup, TTA 적용으로 일반화 성능 향상
- **클래스별 threshold 최적화**로 라벨별 예측 민감도 조절
- **다양한 조합 앙상블**로 안정적인 성능 확보

## VIII. 프로젝트를 통해 배운 점
- 대회 상위권 참가자들의 결과를 분석해본 결과, **여러 모델의 예측을 앙상블** 하여 성능을 극대화한 경우가 많았음.
- 그러나 본 프로젝트에서는 **연산 자원의 한계**로 인해 대형 모델 사용이 어려웠고, 앙상블 기법 자체를 시도하지 못함.
- 이번 경험을 통해 **앙상블의 효과와 다양한 조합 전략**에 대해 깊이 이해하게 되었으며, 추후 프로젝트에서는 이러한 방법론을 적극적으로 활용할 계획

## Implementation Code
프로젝트에 사용된 코드는 아래 깃허브 링크를 통해 확인할 수 있습니다.

[![Repo Card](https://github-readme-stats.vercel.app/api/pin/?username=113bommy&repo=dacon_multi_label_classificaiton&theme=default)](https://github.com/113bommy/dacon_multi_label_classificaiton.git)