---
layout: post
title:  "250211 Research Note"
date:   2024-02-07 00:10:00 +0900
categories: [LLM, Enviroment]   
---
## 2025.02.11 Research Note

To Do
1. SBFL Method Soundness 검증 완료
2. SBFL Method를 사용해서 Sound하게 검증할 수 있는 데이터로 Fault localization Result 데이터 생성
3. *Deepseek coder LLM 로컬 환경 설정*
4. Prompt Engineering 정리
5. Deepseek coder Zero / One / Few Shot으로 Fault Localization 진행 후, 2번과 성능 비교

Additional
1. Teps 준비

## 1. SBFL Method Soundness 검증

SBFL 방법론을 사용해서 sound하게 검증할 수 있는 logical error type를 찾을 수 있다면, 굳이 LLM을 사용하지 않아도 저렴한 Computational Cost만으로 Fault Localize가 가능하기 때문에 **Sound한 Defect type**을 검증하는 과정이 필요하다.

SBFL 방법론 자체가 여러 Test Case에 대한 Coverage 정보를 바탕으로 오류의 위치를 추정하는 것이기 때문에 이를 제대로 활용하기 위해서는 통과하는 Test Case와 실패하는 Test Case의 **Coverage 데이터에 차이**가 존재하는 데이터만을 검증할 수 있다는 가설을 세웠다.

더 자세하게, 여기에서 사용하는 Coverage 데이터는 각 라인이 실행되었는지, 혹은 실행되지 않았는지에 대한 정보만을 담고 있다.

만약 반복문이 존재한다면, 여러번 실행 과정에서 coverage 결과가 존재할텐데, 해당 데이터들이 섞이는 현상이 발생해 제대로 예측이 불가능할 것이다. 따라서 나는 sound한 defect case를 찾기 위해서 prediction 결과가 **반복문 밖에 존재하는 분기문인 경우을 가정**하고 진행했다.

검증을 간단히 하기 위해서 Predicted line이 single line branch인 경우에만 검증 대상으로 잡았다.

