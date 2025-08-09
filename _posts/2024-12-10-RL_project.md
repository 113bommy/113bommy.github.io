---
layout: post
title:  "Reinforcement Learning Project"
date:   2024-12-10 00:10:00 +0900
categories: [Project]
tags: [Project, RL]
---
# Plasticity·Constrained RL based Racing Agent
##### Reinforcement Learning Project
![RL_gif](/assets/img/Project/RL/CarRacing_eval_Constrained.gif)

이 프로젝트는 **안전하며 다양한 상황에 대응 가능한 레이싱카**를 구현하는 것을 목표로 했습니다.

## I. Plasticity 조절
- **목적:** 학습 중 환경 적응 능력(가소성) 유지
- **방법:**
  - 병렬 환경에서 다양한 데이터 수집 (Replay Buffer)
  - **Replay Ratio**, **Shrink & Perturb 주기** 조절
  - 특정 환경에 과적응하는 것을 방지

## II. Constrained Reinforcement Learning
- **목적:** 규칙을 지키는 안전한 주행 학습
- **방법:**
  - 최고 속도 제한, 도로 이탈 방지 등 제약 조건 부여
  - **라그랑지안 최적화**로 정책과 제약 조건을 동시에 만족

## III. Transformer 기반 시각 확장
- **문제:** CNN은 공간 정보만 처리, 시간 정보 반영 어려움
- **해결:** CNN + Transformer 결합
  - CNN → 공간 정보 추출
  - Transformer → 시간적 흐름 인코딩
- **결과:** CNN 대비 평균 점수 상승

## IV. 주요 성능
- **CNN Only 평균 점수:** 약 631.6
- **CNN + Transformer 평균 점수:** 약 798.6
- **Plasticity + Constrained RL 모델:** 안정적 주행, 최대 900점 기록

---

**결론:**  
Plasticity 조절, Constrained RL, 그리고 Transformer 기반 시각 확장은 레이싱 AI의 안전성과 적응력을 크게 향상시켰으며, 기존 CNN 모델 대비 성능 향상과 규칙 준수 주행을 동시에 달성했습니다.

{% include pdf_viewer.html src="/assets/pdf/RL_project.pdf" %}

## Implementation Code
Car Racing Project 구현 코드는 아래 깃허브 링크를 통해 확인할 수 있습니다.

[![Repo Card](https://github-readme-stats.vercel.app/api/pin/?username=113bommy&repo=RL_project_car_racing&theme=default)](https://github.com/113bommy/RL_project_car_racing.git)