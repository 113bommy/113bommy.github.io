---
layout: post
title:  "Matrix Calculus - Backpropagation"
date:   2023-09-12 23:10:00 +0900
categories: [Computer Vision, CS231n]
---
#### Matrix Calculus

****

* CS231n Lecture4에서 Backpropagation을 위해 Matrix와 vector들에 대한 편미분값을 구하는 내용이 존재한다. 이 부분이 잘 이해가 되지 않아 추가적인 공부를 진행했다.
* 행렬과 행렬 간의 미분을 다루는 내용은 "Matrix Calculus"라고 부른다.

아래의 내용은 CS231n github에 게시된 **Vector, Matrix, Tensor Derivatives** (https://cs231n.github.io/optimization-2/) 파일의 내용과 Matrix Cookbook을 참고하여 공부한 후 정리한 내용이다. 

****

Matrix Calculus에서 주로 다루는 대상은 Scalar, Vector, Matrix 세가지로 구분할 수 있다. 이번 노트에서는 스칼라를 x, y같은 일반적인 변수로, 벡터를 **x, y**와 같은 볼드체로, 행렬을 X, Y와 같은 대문자로 표현할 것이다.

이번 노트에서는 Backpropagation에 관련한 내용을 위주로 다루고 있기 때문에, 행렬의 곱으로 표현된 행렬에 대한 편미분만을 다루어 정리했다.

Backpropagation을 위한 Gradient 계산을 목적으로 행렬의 편미분을 다루는 것이므로, 명확한 공식이 존재하는 편미분들에 대해서 중점적으로 다루고, <span style='background-color: #fff5b1'>나머지 Problem-dependent한 미분들은 추가적으로 공부할 예정이다.</span>

|            | Scalar    | Vector            | Matrix            |
| ---------- | --------- | ----------------- | ----------------- |
| **Scalar** | ∂y/∂x     | ∂**y**/∂x         | ∂Y/∂x             |
| **Vector** | ∂y/∂**x** | ∂**y**/∂**x**     | Problem-dependent |
| **Matrix** | ∂y/∂X     | Problem-dependent | Problem-dependent |

****

1. **벡터와 벡터 사이의 편미분** (Partial derivative of Vector over Vector)

* (**y** = W**x**)의 **∂y/∂x** 벡터 y에 대한 벡터 x의 편미분

y = Wx에서 y를 nx1, W를 nxm, x를 mx1의 벡터와 행렬로 가정하자.

y의 3번째 원소는 W 행렬의 3번째 행과 x 벡터의 내적 값으로 표현된다. 이를 수식으로 표현하면, 아래와 같다.

![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?y_3%20%3D%20%5Cbegin%7Bbmatrix%7D%20%26%20w_%7B3%2C1%7D%2Cw_%7B3%2C2%7D%2Cw_%7B3%2C3%7D%20...%20w_%7B3%2Cm%7D%20%26%20%5Cend%7Bbmatrix%7D%20*%20%5Cbegin%7Bbmatrix%7D%20x_1%5C%5C%20x_2%5C%5C%20x_3%5C%5C%20...%5C%5C%20x_4%20%5Cend%7Bbmatrix%7D)

![This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.](https://latex.codecogs.com/gif.latex?y_3%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7DW_%7B3%2Ci%7D*x_i)

![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cdelta%20y_3%7D%7B%5Cdelta%20x_k%7D%20%3D%20W_%7B3%2Ck%7D)

각 원소 **y_i**에 대한 **x_j**의 편미분값은 W_(i, j)와 같으므로 이 행렬에 대한 Jacobian Matrix는 아래와 같다.

![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdelta%20y_1/%5Cdelta%20x_1%20%26%20%5Cdelta%20y_1/%5Cdelta%20x_2%20%26...%20%26%20%5Cdelta%20y_1/%5Cdelta%20x_m%5C%5C%20%5Cdelta%20y_2/%5Cdelta%20x_1%20%26%20%5Cdelta%20y_2/%5Cdelta%20x_2%20%26%20...%20%26%20%5Cdelta%20y_2/%5Cdelta%20x_m%5C%5C%20...%26%20...%20%26%20...%20%26%20...%5C%5C%20%5Cdelta%20y_n/%5Cdelta%20x_1%26%20...%20%26%20...%20%26%20%5Cdelta%20y_n/%5Cdelta%20x_m%20%5Cend%7Bbmatrix%7D)

![img](https://latex.codecogs.com/gif.latex?%3D%20%5Cbegin%7Bbmatrix%7D%20W_%7B1%2C1%7D%20%26%20W_%7B1%2C2%7D%20%26%20...%20%26%20W_%7B1%2Cm%7D%5C%5C%20W_%7B2%2C1%7D%20%26%20W_%7B2%2C2%7D%20%26%20...%26%20...%5C%5C%20...%20%26%20...%20%26%20...%20%26%20...%5C%5C%20W_%7Bn%2C1%7D%20%26%20...%20%26%20...%20%26W_%7Bn%2Cm%7D%20%5Cend%7Bbmatrix%7D%20%3DW)

따라서 **y** (**y** = W**x**)에 대한 **x**의 편미분은 앞에 곱해진 W matrix와 동일하다.

* (**y** = **x**W)의 **∂y/∂x** 벡터 y에 대한 벡터 x의 편미분

이 벡터는 y가 1xn인 row vector, x가 1xm인 row vector, W가 nxm인 matrix에 해당한다. 따라서 y를 일반화하면 다음과 같다.

![img](https://latex.codecogs.com/gif.latex?y_i%20%3D%20%5Csum_%7Bk%20%3D%201%7D%5E%7Bm%7Dx_k*W_%7Bk%2Ci%7D)

![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cdelta%20y_i%7D%7B%5Cdelta%20x_j%7D%20%3D%20W_%7Bj%2Ci%7D)

![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cdelta%20y_1/%5Cdelta%20x_1%20%26%20%5Cdelta%20y_1/%5Cdelta%20x_2%20%26...%20%26%20%5Cdelta%20y_1/%5Cdelta%20x_m%5C%5C%20%5Cdelta%20y_2/%5Cdelta%20x_1%20%26%20%5Cdelta%20y_2/%5Cdelta%20x_2%20%26%20...%20%26%20%5Cdelta%20y_2/%5Cdelta%20x_m%5C%5C%20...%26%20...%20%26%20...%20%26%20...%5C%5C%20%5Cdelta%20y_n/%5Cdelta%20x_1%26%20...%20%26%20...%20%26%20%5Cdelta%20y_n/%5Cdelta%20x_m%20%5Cend%7Bbmatrix%7D)

![img](https://latex.codecogs.com/gif.latex?%3D%5Cbegin%7Bbmatrix%7D%20W_%7B1%2C1%7D%20%26%20W_%7B2%2C1%7D%20%26%20...%20%26%20W_%7Bn%2C1%7D%5C%5C%20W_%7B1%2C2%7D%20%26...%20%26%20...%20%26%20...%5C%5C%20...%26%20...%20%26...%20%26...%20%5C%5C%20W_%7B1%2Cm%7D%26...%20%26%20...%20%26%20W_%7Bn%2Cm%7D%20%5Cend%7Bbmatrix%7D%20%3D%20W%5ET)

따라서 **y**(**y** = **x**W)에 대한 **x**의 편미분은 뒤에 곱해진 벡터W의 Transpose matrix와 동일하다.

****

2. 벡터를 스칼라로 편미분(∂**y**/∂x)

벡터 **y**를 n개의 원소를 가진 벡터로, x를 스칼라로 가정하면, y에 대한 x로의 편미분은 아래와 같이 유도할 수 있다.

![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cdelta%20y_1%7D%7B%5Cdelta%20x%7D%20%5C%5C%20%5Cfrac%7B%5Cdelta%20y_2%7D%7B%5Cdelta%20x%7D%20%5C%5C%20...%5C%5C%20%5Cfrac%7B%5Cdelta%20y_n%7D%7B%5Cdelta%20x%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20p_1%5C%5C%20p_2%5C%5C%20...%5C%5C%20p_n%20%5Cend%7Bbmatrix%7D)

x가 스칼라이므로 1x1 벡터로 보자면, **x앞에 nx1 행렬이나 x 뒤에 1xn 행렬**이 곱해진 것으로 생각할 수 있다.

위의 벡터와 벡터 사이의 편미분을 생각해본다면, x 앞에 nx1 행렬이 곱해진 경우, x로의 편미분은 **nx1의 행렬**이 나올 것이고, x 뒤에 1xn이 곱해진 경우 x로의 편미분은 **1xn의 transpose 형태인 nx1 형태의 행렬**이 나올 것이다. 

위의 두 결과를 비교하면, 동일한 결론이 나온다는 것을 확인할 수 있다.

****

3. 행렬을 스칼라로 편미분 (∂Y/∂x)

행렬 Y를 nxm 행렬로, x를 스칼라로 가정하면, 행렬 Y에 대한 스칼라 x의 편미분은 아래와 같다.

![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cdelta%20Y%7D%7B%5Cdelta%20x%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cdelta%20Y_%7B1%2C1%7D%7D%7B%5Cdelta%20x%7D%26%5Cfrac%7B%5Cdelta%20Y_%7B1%2C2%7D%7D%7B%5Cdelta%20x%7D%20%26%20...%20%26%5Cfrac%7B%5Cdelta%20Y_%7B1%2Cm%7D%7D%7B%5Cdelta%20x%7D%5C%5C%20%5Cfrac%7B%5Cdelta%20Y_%7B2%2C1%7D%7D%7B%5Cdelta%20x%7D%26%5Cfrac%7B%5Cdelta%20Y_%7B2%2C2%7D%7D%7B%5Cdelta%20x%7D%26%20...%20%26%20...%5C%5C%20...%26...%26...%26...%5C%5C%20%5Cfrac%7B%5Cdelta%20Y_%7Bn%2C1%7D%7D%7B%5Cdelta%20x%7D%26...%26...%26%5Cfrac%7B%5Cdelta%20Y_%7Bn%2Cm%7D%7D%7B%5Cdelta%20x%7D%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20p_%7B1%2C1%7D%26p_%7B1%2C2%7D%26...%26p_%7B1%2Cm%7D%5C%5C%20p_%7B2%2C1%7D%26p_%7B2%2C2%7D%26...%26...%5C%5C%20...%26...%26...%26...%5C%5C%20p_%7Bn%2C1%7D%26...%26...%26p_%7Bn%2Cm%7D%20%5Cend%7Bbmatrix%7D)

Y를 nxm 행렬로 생각한다면, x는 1x1행렬에 불과하기 때문에, 행렬 Y를 행렬과 x의 곱 형태로는 표현할 수 없다. 

****

4. 스칼라를 벡터로 미분(∂y/∂**x**)

y를 스칼라로, x를 n개의 원소로 가지는 벡터로 가정하면, 스칼라 y에 대한 벡터x로의 미분은 아래와 같다.

![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x_1%7D%26%20%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x_2%7D%26...%26%20%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x_n%7D%20%5Cend%7Bbmatrix%7D)

**y**는 1x1 벡터로, **x**는 nx1또는 1xn 벡터로 볼 수 있다. **x**가 nx1 벡터인 경우, **y** = **px** (p는 1xn) 벡터로 표현할 수 있고, 이 경우, **y**에 대한 **x**의 편미분은 **p**벡터이므로 n개 원소를 가진 row vector가 된다. 반대로 **x**가 1xn 벡터인 경우, **y = xp **(p는 nx1 벡터)로 표현이 가능하다. 이 경우, **y**에 대한 **x**의 편미분은 **p**벡터의 Transpose이므로 n개 원소인 row vector가 된다.

위의 두 결과를 비교하면, 동일한 결론이 나온다.

****

5. 스칼라를 행렬로 미분(∂y/∂X)

![img](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x_%7B1%2C1%7D%7D%26%20%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x_%7B2%2C1%7D%7D%26...%26%20%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x_%7Bn%2C1%7D%7D%5C%5C%20%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x_%7B1%2C2%7D%7D%26%20%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x_%7B2%2C2%7D%7D%26...%26%20...%5C%5C%20...%26%20...%26...%26%20...%5C%5C%20%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x_%7B1%2Cm%7D%7D%26%20...%20%26...%26%20%5Cfrac%7B%5Cdelta%20y%7D%7B%5Cdelta%20x_%7Bn%2Cm%7D%7D%5C%5C%20%5Cend%7Bbmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20p_%7B1%2C1%7D%26p_%7B2%2C1%7D%26...%26p_%7Bn%2C1%7D%5C%5C%20p_%7B1%2C2%7D%26p_%7B2%2C2%7D%26...%26...%5C%5C%20...%26...%26...%26...%5C%5C%20p_%7B1%2Cm%7D%26...%26...%26p_%7Bn%2Cm%7D%20%5Cend%7Bbmatrix%7D)

스칼라 y를 행렬에 대해 미분한다면, 편미분 행렬의 각 원소는 y에 대해서 transpose된 자리의 x값으로 편미분해주는 것과 동일하다.

****

* Vector/Matrix, Matrix/Vector, Matrix/Matrix끼리의 편미분

위에서 스칼라, 벡터, 행렬에 대한 편미분을 정리하여 6가지 경우의 수를 다루었다.

이론적으로 벡터와 행렬간의 편미분, 행렬과 행렬간의 편미분을 다룰 수 있다. 그 내용은 위 링크의 pdf 파일에 정리되어 있다.

다만, 이번 Note에서는 Backpropagation을 위한 행렬의 미분을 다루는 것이기에, 이에 맞추어 정리했다.

CS231n-Lecture4의 아래 예시를 살펴보자.

![IMG_0334](/assets/img/CS231n/Matrix_Calculus/IMG_0334.png)

위의 f(x, W)의 예시에서는 Backpropagtion을 위해 Wx의 행렬을 q로 두고, L2 operation과 행렬의 곱셈 연산으로 Computational graph를 생성했다. 

이후, L2 operation의 Local gradient 2q를 구한 뒤, 행렬 곱의 Local gradient를 구하는 과정을 진행해주었다.

∂**q**/∂**x** 연산은 앞선 벡터와 벡터 간의 편미분에서 다루었던 것처럼, 벡터 **x**앞에 곱해진 W 행렬이 그대로 연산 결과로 도출되게 된다. 반면, ∂**q**/∂W의 경우에는 matrix의 크기가 정확하게 정해지지 않는 문제점이 있다.

하지만, 가장 중요한 부분인, matrix 편미분에서 나누는 matrix의 크기와 미분값이 일치해야 한다는 규칙에 따라 ∂**q**/∂W는 x(1x2)로 결정이 되고, 2x2 행렬의 크기를 맞추기 위해 2q와 transpose한 x과의 곱을 미분값으로 사용했다. 또한, ∂**q**/∂**x**는 W값이 사용되므로 2*W*q를 사용하여 x 행렬의 크기를 맞췄다.

이렇듯, 행렬과 벡터 사이의 미분을 사용해야 한다면, problem-dependent하므로 **계산 결과를 결정하기 복잡해진다는 문제점**이 있다. 언제나 최적의 계산 방법은 아니겠지만, 스칼라를 결과로 도출하는 함수 전체 f(x,W)를 이용하여 matrix를 생성하는 편이 좋다고 생각한다. 

아래 식은 스칼라 f(x, W)에 대한 벡터 x의 편미분과 행렬 W의 편미분을 나타낸 것이다.

![ㄹ](/assets/img/CS231n/Matrix_Calculus/x_derivative.png)

![W_derivative](/assets/img/CS231n/Matrix_Calculus/W_derivative.png)

행렬의 편미분 연산에 스칼라가 포함되어 있다면, 편미분 연산이 명확해지기 때문에 이와 같은 방법을 사용하는 편이 더 좋을 것 같다. 하지만, 이 경우 Computational Graph보다 연산 과정이 훨씬 복잡해진다. 이를 하나의 게이트로 이용할 지는 상황에 따라 달라질 것이다.

추후에 더 좋은 방법이 있다면, 또다른 Note에 정리할 것이다.
