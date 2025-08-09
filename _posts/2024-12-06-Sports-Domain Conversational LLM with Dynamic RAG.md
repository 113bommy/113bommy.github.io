---
layout: post
title:  "Sports‑Domain Conversational LLM with Dynamic RAG"
date:   2024-12-06 00:10:00 +0900
categories: [Project]
tags: [Project, LLM, RAG]
---


# 스포츠 도메인 특화 LLM 개발

## I. 프로젝트 개요
본 프로젝트는 **축구 중심 스포츠 도메인 특화 LLM 서비스**를 개발하는 것을 목표로 했습니다.  
기존 범용 LLM의 한계(전문 지식 부족, 높은 Hallucination 발생률)와 기존 스포츠 특화 LLM의 제한적인 데이터 분석 능력을 개선하기 위해 **RAG(Retrieval-Augmented Generation)** + **Sports-Centric LLM 학습**을 결합한 시스템을 구축했습니다.  

---

## II. 사용 기술
- **데이터 전처리:**  
  - Q&A 템플릿 적용, 질문·답변 파싱, 오류 데이터 플래그 관리  
  - 학습(train)·검증(dev)·테스트(test) 데이터셋 분할  
- **웹 UI 개발:**  
  - `Streamlit` 기반 대화형 Chat UI 제작  
  - 실시간 질의응답, 데이터 시각화, 분석 기능 제공  
- **RAG 구현:**  
  - 나무위키 + 축구 전문 데이터 크롤링(총 668문서)  
  - 불필요 내용 제거 및 질문 유형별 인덱싱  
  - Gemini tool-use 기반 **Dynamic RAG** 적용  
- **LLM 학습:**  
  - 축구 전술, 선수 기록, 규칙 등 **Sports-Centric Q&A 데이터셋** 학습  
  - RAG 결합으로 학습 데이터 + 실시간 검색 데이터 동시 활용  
- **데이터 시각화:**  
  - `Matplotlib`, `Plotly` 등으로 선수 능력치 플롯 및 포지션별 지표 시각화  
- **크롤링 및 데이터 처리:**  
  - `BeautifulSoup`, `Requests` 활용  
  - 텍스트 전처리 및 인덱싱 최적화  

---

## III. 주요 기능
1. **축구 용어·개념 검색**  
   - 전술, 기술, 규칙 등 전문 개념을 알기 쉽게 설명  
2. **선수 전적 검색**  
   - K리그 선수의 경기 기록과 활약상 제공  
3. **선수 능력치 시각화**  
   - 포지션별 주요 지표(득점, 패스 성공률, 드리블 성공률 등)를 그래프로 표시  
4. **RAG 기반 검색**  
   - 학습 데이터 + 외부 데이터베이스 결합으로 신뢰성 있는 최신 정보 제공  
5. **포지션별 세부 분석**  
   - 골키퍼, 수비수, 미드필더, 공격수별 맞춤 지표 분석 및 비교  

---

## IV. 한계점 & 개선 방향
- **한계점**  
  - SFT + RAG 결합 학습 미진행 → 성능 제한  
  - Gemini 임베딩의 바이트 제한 → 문서 파편화로 검색 품질 저하 가능성  
- **향후 계획**  
  1. SFT + RAG 결합 학습으로 성능 고도화  
  2. Llama 등 오픈소스 LLM 적용으로 상용 모델 의존도 축소  
  3. 기능 확장  
     - **전략 분석가용:** 포메이션 분석, 유사 선수 추천, 상대 전략 대비  
     - **팬용:** 경기 요약, 베스트 선수 추천, 토너먼트 시뮬레이션  
     - **스카우팅용:** 선수 지표 비교, 성장 가능성 예측, 재계약 여부 판단  

---

## V. 프로젝트 의의
이번 프로젝트는 **스포츠 데이터 활용의 실질적 가능성**을 보여주었으며,  
데이터 기반 의사결정과 사용자 경험을 동시에 향상시킬 수 있는 LLM 서비스 구조를 제시했습니다.

{% include pdf_viewer.html src="/assets/pdf/capstone_design_final_report.pdf" %}

[Sports‑Domain Conversational LLM with Dynamic RAG Final Report](/assets/pdf/capstone_design_final_report.pdf)

{% include pdf_viewer.html src="/assets/pdf/capstone_design_final_presentation.pdf" %}

[Sports‑Domain Conversational LLM with Dynamic RAG Final Presentation](/assets/pdf/capstone_design_final_presentation.pdf)
