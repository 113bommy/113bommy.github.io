---
layout: post
title:  "Expanded Spectrum‑Based Fault Localization for Automated Program Repair"
date:   2025-06-30 00:10:00 +0900
categories: [Project]
tags: [Project, APR, Automated Program Repair, KCC]
---

# Expanded SBFL 기반 LLM 자동 프로그램 수정
#### KCC 2025 Project

이번 프로젝트 **"Expanded SBFL 기반 LLM 자동 프로그램 수정"** 은 Python 논리 오류를 더 정밀하게 찾아내어 Large Language Model(LLM)의 자동 프로그램 수정(Auto Program Repair, APR) 성능을 높이는 것을 목표로 했습니다.  

일반적인 Spectrum-Based Fault Localization(SBFL)은 빠르고 언어에 구애받지 않지만, 실제 오류 위치를 정확히 집어내기에는 한계가 있습니다. 이를 보완하기 위해 **조건문 중심의 신뢰 가능한 예측만 선별**하고, 해당 영역의 변수와 제어 흐름을 확장하는 **3단계 컨텍스트 확장 기법(Expanded SBFL)** 을 적용했습니다.  

![Figure](/assets/img/Project/KCC_2025/figure.png)

## I. Expanded SBFL 확장 방식
- **Depth 1:** 의심 라인이 포함된 분기 전체 포함  
- **Depth 2:** 의심 라인에서 사용된 변수 관련 라인 추가  
- **Depth 3:** Depth 1 범위 + 변수 확장 결합  

## II. 실험 결과
- 데이터셋: Google DeepMind CodeContest (Python)  
- 모델: DeepSeek-Coder-V2-Lite-Instruct  
- **Pass@1 성능**: 최대 37.89% → GT Fault Localization(38.67%)에 근접  
- 반복문 오류·코드 누락과 같이 SBFL 신호가 약한 경우에는 개선 효과 제한  

## III. 결론
Expanded SBFL은 신뢰도 높은 오류 예측과 풍부한 코드 컨텍스트를 제공하여 LLM 기반 자동 프로그램 수정의 성공률을 높였습니다. 앞으로는 SBFL이 취약한 오류 유형까지 보완할 수 있는 로컬라이제이션 기법이 필요합니다.


KCC 2025 정보과학회 학부생 인공지능 응용 부문으로 제출했습니다.

{% include pdf_viewer.html src="/assets/pdf/KCC_2025_TraceFix.pdf" %}

아래는 해당 논문 발표를 위한 Poster입니다.

{% include pdf_viewer.html src="/assets/pdf/KCC_2025_Poster.pdf" %}
