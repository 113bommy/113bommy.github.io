---
layout: post
title:  "Direct Preference Optimization"
date:   2025-02-18 00:10:00 +0900
categories: [Transformer]   
---

# **Direct Preference Optimization** - Your Language Model is Secretly a Reward Model

## I. Abstract

대규모 Corpus를 통해서 학습한 Language Model(LM)은, 정보를 학습하거나 특정 task에 대한 추론 성능은 뛰어날 지라도, **Unsupervised된 학습**의 특성상 LM의 동작을 완전히 규제하는 것은 어렵다.

따라서 Fine-Tuning을 진행하는데, 대표적인 방법은 2가지이다.

#### 1. SFT(Supervised Fine Tuning)
    이는 사람이 제공한 데이터를 바탕으로 정답을 학습시키는 지도학습에 해당.
#### 2. Reinforcement Learning
    RLHF 방식으로 Human Feedback 정보를 강화학습을 통해 Preferece를 반영하는 model을 생성하고, 해당 model의 estimated reward를 사용하여 이를 최대화하는 방향으로의 학습을 진행함.

RLHF 방식은 따라서 **Unstable Procedure**에 해당함.
이 논문에서는, 기존 방법론을 개선하여 RLHF에 대한 새로운 reward model parameterization 방식을 도입하여 성능을 높이고자 했다.

## II - Introduction

Abstract에서 언급한 것과 같이, 매우 다양한 지식과 능력들 중에서 모델의 원하는 답변이나 행동을 선택하는 것은 매우 어렵다. 

기존의 방법론들은 간단하게, human preference data를 LM에게 instill(서서히 주입)하는 방식으로 진행되었다.
이는 또다시 `SFT`와 `RHLF`방법으로 분류 가능하다.

`RLHF`방법은 Human preference를 반영한 `reward model`을 학습한 이후, 기존의 language model을 크게 변형시키지 않는 선에서 `policy`를 최적화하는 방식을 의미한다.

다만, 위 과정은 `multiple LM` 훈련을 동반하므로, training 과정에서 매우 많은 연산을 필요로 한다.

이 논문에서는 `DPO` 방식을 새롭게 적용하여 직접 model의 policy를 학습하고자 한다. 이때 DPO는 기존의 model들과 마찬가지로 theoretical preference model에 의존하긴 하지만, `preference loss`를 구현하는 과정에서 차이가 존재한다.

## III - Preliminaries

