---
layout: post
title:  "Comparative Analysis of Time‑Series Imputation Methods"
date:   2024-06-12 00:10:00 +0900
categories: [Project]
tags: [Project, Time Series, Imputation]
---
## I. 프로젝트 개요
이 프로젝트는 단변량(Univariate)과 다변량(Multivariate) 시계열 데이터의 결측치를 보완하는 다양한 보간(Imputation) 기법을 비교·분석하는 것을 목표로 했습니다. 통계 기법부터 머신러닝, 딥러닝 기반 모델까지 폭넓게 실험하여 각 방법의 성능과 특징을 평가했습니다.

## II. 분석 대상 및 방법론
### 1. 단변량 시계열
- **통계 기법:** ARIMA, Holt-Winters, 단순 평균 대체 등
- **머신러닝 기법:** Support Vector Regression(SVR)
- 다양한 결측률과 결측 패턴을 가정하여 성능 비교

### 2. 다변량 시계열
- **통계·머신러닝 기법**과 함께 **SAITS(Self-Attention-based Imputation for Time Series)** 딥러닝 모델 적용
- 건물 전력 소비 데이터를 대상으로 다변량 결측 복원 성능 평가

## III. 주요 성과
- 각 방법론의 장단점 및 결측 패턴별 성능 변화를 비교 분석
- SAITS 모델이 다변량 환경에서 높은 정확도를 보이며 복잡한 상관관계를 잘 반영함을 확인
- 전통적 통계 기법은 단순한 경우 안정적이고 효율적인 성능 제공

## IV. 결론 및 시사점
- 데이터 특성과 결측 패턴에 따라 최적의 보간 기법이 다름을 확인
- 향후 연구에서는 대규모 산업 데이터나 실시간 스트리밍 데이터 환경에서 활용 가능한 결측 대체 방법으로 확장 가능

{% include pdf_viewer.html src="/assets/pdf/Comparative_Analysis_of_Time_series_Imputation_Methods.pdf" %}

[Comparative Analysis of Time-series Imputation Methods](/assets/pdf/Comparative_Analysis_of_Time_series_Imputation_Methods.pdf)

