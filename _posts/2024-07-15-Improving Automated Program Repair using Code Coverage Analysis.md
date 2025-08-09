---
layout: post
title:  "Improving Automated Program Repair using Code Coverage Analysis"
date:   2024-07-15 00:10:00 +0900
categories: [Project]
tags: [Project, APR, Automated Program Repair, KCC]
---

## I. 연구 개요
이 프로젝트는 **CodeT5** 모델과 **코드 커버리지(Code Coverage) 분석**을 결합하여 C++ 프로그램의 **논리적 오류**를 효과적으로 수정하는 방법을 제안합니다. 기존 APR 연구가 문법 오류에 집중한 반면, 본 연구는 **테스트 케이스 실행 결과와 코드 실행 흐름 분석**을 통해 **논리 오류**를 찾아내고 수정하는 데 초점을 맞추었습니다.

## II. 연구 배경
- 논리 오류는 컴파일러가 탐지하지 못해 테스트 케이스 실행 결과를 통해서만 발견 가능.
- **스펙트럼 기반 오류 위치 추정(SBFL)**은 성공/실패 테스트 케이스의 실행 라인 데이터를 비교해 오류 가능성이 높은 라인을 찾는 전통적 기법.
- 본 연구는 SBFL을 **딥러닝 기반 APR**과 결합하여 모델의 오류 위치 인식 능력을 강화.

## III. 데이터셋 구축 과정
1. **데이터 출처**: Google DeepMind의 **CodeContests** 데이터셋 (정답 코드, 오답 코드, 테스트 케이스 포함)
2. **데이터 페어링**: Edit Distance ≤ 9 조건으로 논리 오류 중심 데이터 생성
3. **코드 포맷팅**: Clang-format으로 데이터 페어링의 노이지를 최소화
4. **커버리지 수집**:  
   - `gcov`로 성공/실패 테스트 케이스별 실행 라인 수집  
   - 실행 라인과 인접 라인에 가중치를 부여하여 **의심 라인** 추출
5. **학습 입력 구성**: 추출한 의심 라인 번호를 코드 첫 줄에 주석 형태로 삽입

## IV. 모델 구조 및 실험
- **Baseline**: 잘못된 코드 입력 → 수정 코드 생성 (CodeT5)
- **w/ GT Line**: 실제 오류 라인 번호 제공
- **w/ Predicted Line**: 커버리지 분석 기반 의심 라인 번호 제공
- **평가 지표**:
  - **Perfect Repair**: 모든 테스트 케이스 통과
  - **Partial Repair**: 일부 테스트 케이스 성능 향상
  - **Perfect Localization**: 예측 라인과 실제 오류 라인이 완벽 일치
  - **Partial Localization**: 예측 라인 범위에 실제 오류 포함

## V. 주요 실험 결과
- Clang-format 적용만으로도 오류 위치 추정 및 수정 성능 향상
- Predicted Line 제공 시:
  - Clang-format 적용 → 성능 크게 향상
  - 미적용 → 향상 폭 제한적
- GT Line 제공 시:
  - 오류 수정 및 위치 추정 정확도가 압도적으로 향상
  - 정확한 오류 위치 정보의 중요성 입증
  
![Repair rate](/assets/img/Project/KCC_2024/Repair_Rate.png)
![Repair rate](/assets/img/Project/KCC_2024/Localization_Rate.png)

## VI. 결론
- **코드 커버리지 기반 오류 위치 추정**은 APR의 성능을 높이는 핵심 요소
- **코드 포맷팅 + 커버리지 분석** 조합이 모델의 오류 수정 능력을 크게 개선
- 향후 정교한 커버리지 기반 위치 추정 알고리즘을 통해 APR 성능을 더욱 강화 가능

{% include pdf_viewer.html src="/assets/pdf/KCC_2024_Logical_Error_Fix_using_Code_Coverage.pdf" %}

[3D Reconstruction Tool Final Report](/assets/pdf/KCC_2024_Logical_Error_Fix_using_Code_Coverage.pdf)

{% include pdf_viewer.html src="/assets/pdf/KCC_2024_Logical_Error_Fix_using_Code_Coverage_poster.pdf" %}

[3D Reconstruction Tool Final Report](/assets/pdf/KCC_2024_Logical_Error_Fix_using_Code_Coverage_poster.pdf)

