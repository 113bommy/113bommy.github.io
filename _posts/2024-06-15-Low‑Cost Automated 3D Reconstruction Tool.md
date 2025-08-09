---
layout: post
title:  "Low‑Cost Automated 3D Reconstruction Tool"
date:   2024-06-15 00:10:00 +0900
categories: [Project]
tags: [Project, CV, 3D Reconstruction]
---

# Low-Cost 3D Reconstruction Tool

## I. 프로젝트 개요
본 프로젝트는 여러 대의 카메라를 사용해야 하는 기존 3D Reconstruction의 비용 문제를 해결하기 위해,  
물체를 회전시키며 단일 저해상도 카메라로 촬영하는 방식으로 품질 저하를 최소화하면서 비용 절감을 목표로 했습니다.
이를 통해 제품 디자인, 온라인 쇼핑몰 3D 모델 생성 등에서 활용 가능한 저비용 솔루션을 제공합니다.

## II. 시스템 설계
- 회전판: 아두이노 보드 + 스텝 모터 + 모터 드라이버
- 카메라: 원래 카메라 모듈 + SD 카드 사용
- 촬영 주기: SD 카드 저장 소요 시간(약 0.5초) 고려, 1초로 설정
- 이미지 전처리: 카메라 왜곡 보정, 색상 대비 조정
- 3D 재구성 파이프라인:
  - 특징점 추출: SIFT
  - 특징점 매칭: SIFT 연속 처리 후 RANSAC으로 잘못된 매칭 제거
  - 재구성: SfM(Structure from Motion) 기반 Sparse/Dense Reconstruction
- 결과 확인: MeshLab으로 재구성된 모델을 시각적으로 확인

## III. 문제 해결 과정
- 배경 노이즈 감소: A4 용지 또는 컬러 보드 사용
- 어두운 환경 보정: 조명 추가
- 카메라-물체 거리 최적화: 15~25cm 범위 실험 후 20cm 결정
- 저해상도 이미지 문제: 전처리 및 크로마키 배경 제거로 해결
- Grid Search로 촬영 환경에 맞는 최적 파라미터 탐색

## IV. 실험 결과
- 단일 저해상도 카메라 + 회전 테이블로 물체의 구조와 색상을 일정 수준 복원
- 완벽하지는 않지만 후처리를 위한 기본 뼈대 모델로 충분히 활용 가능

## V. 결론 및 개선 방향
- 비용 절감 + 품질 유지 달성
- 향후 개선 방안:
  - 고해상도 카메라 사용 (단일 카메라로 비용 절감 가능)
  - 다각도 촬영
  - 저해상도 이미지에 특화된 딥러닝 모델 Fine-Tuning 적용

{% include pdf_viewer.html src="/assets/pdf/3D_ReconstructionTool.pdf" %}

[3D Reconstruction Tool Final Report](/assets/pdf/3D_ReconstructionTool.pdf)

