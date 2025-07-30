---
layout: post
title:  "Seq2Seq Code"
date:   2024-02-05 00:10:00 +0900
categories: [RNN]   
---
## **Seq2Seq 코드**

```python
import os
import shutil
import zipfile

import pandas as pd
import tensorflow as tf
import urllib3
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
```


```python
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def download_zip(url, output_path):
    response = requests.get(url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"ZIP file downloaded to {output_path}")
    else:
        print(f"Failed to download. HTTP Response Code: {response.status_code}")

url = "http://www.manythings.org/anki/fra-eng.zip"
output_path = "fra-eng.zip"
download_zip(url, output_path)

path = os.getcwd()
zipfilename = os.path.join(path, output_path)

with zipfile.ZipFile(zipfilename, 'r') as zip_ref:
    zip_ref.extractall(path)

# 영어와 프랑스어의 병렬 데이터를 받아옴.
```

    ZIP file downloaded to fra-eng.zip
    


```python
# fra.txt 파일을 탭을 구분자로 하여 읽어들이고, 구분자로 구분되어 생성된 각 행의 열을 'src', 'tar', 'lic' 이름의 행에 저장함.
lines = pd.read_csv('fra.txt', names=['src', 'tar', 'lic'], sep='\t')
del lines['lic']
print('전체 샘플의 개수 :',len(lines))
# 필요한 정보인 source와 target data만 남기고 lic 열은 삭제해줌.
```

    전체 샘플의 개수 : 229803
    

읽어들인 데이터들 중에서 'src와 'tar'열의 데이터만을 다시 저장해주고,
전체 대이터 23만개 중에서 6만개의 데이터만을 사용해 학습을 진행한다.


```python
lines = lines.loc[:, 'src':'tar']
lines = lines[:100000]
lines.sample(10)
lines.head(10)
```





  <div id="df-9aac7923-3073-4c34-9256-d50fd146abad" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>src</th>
      <th>tar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Go.</td>
      <td>Va !</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Go.</td>
      <td>Marche.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Go.</td>
      <td>En route !</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Go.</td>
      <td>Bouge !</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Hi.</td>
      <td>Salut !</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Hi.</td>
      <td>Salut.</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Run!</td>
      <td>Cours !</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Run!</td>
      <td>Courez !</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Run!</td>
      <td>Prenez vos jambes à vos cous !</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Run!</td>
      <td>File !</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-9aac7923-3073-4c34-9256-d50fd146abad')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-9aac7923-3073-4c34-9256-d50fd146abad button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-9aac7923-3073-4c34-9256-d50fd146abad');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e5a7aecd-2b7c-4f97-b2f1-567cdcb59ede">
  <button class="colab-df-quickchart" onclick="quickchart('df-e5a7aecd-2b7c-4f97-b2f1-567cdcb59ede')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e5a7aecd-2b7c-4f97-b2f1-567cdcb59ede button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




아래 코드는 pandas의 열 데이터에 대한 변환을 수행하는 코드이다. lines.tar를 사용하여 tar 열 데이터를 참조할 수 있도록 했고, lines.tar.apply(function)을 사용하여 각 데이터를 변환하여 적용해주었다.

lambda x는 재참조를 하지 않을 함수를 간단하게 생성하는 함수이다.

```python
lambda x : '\t '+ x + ' \n'
```
따라서 위의 코드는 데이터의 가장 앞과 가장 뒤에 데이터를 추가하는 함수를 구현한 것이다.


```python
lines.tar = lines.tar.apply(lambda x : '\t '+ x + ' \n')
lines.sample(10)
```





  <div id="df-ffc688a2-4b85-4ecf-8d47-8f465dacfb9f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>src</th>
      <th>tar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>45862</th>
      <td>You don't want that.</td>
      <td>\t Tu ne veux pas cela. \n</td>
    </tr>
    <tr>
      <th>5255</th>
      <td>I hate flies.</td>
      <td>\t Je déteste les mouches. \n</td>
    </tr>
    <tr>
      <th>33256</th>
      <td>I can't wait to go.</td>
      <td>\t J'ai hâte d'y aller. \n</td>
    </tr>
    <tr>
      <th>52508</th>
      <td>Tom is always crying.</td>
      <td>\t Tom pleure tout le temps. \n</td>
    </tr>
    <tr>
      <th>11676</th>
      <td>I want to live.</td>
      <td>\t Je veux vivre. \n</td>
    </tr>
    <tr>
      <th>35456</th>
      <td>My family is small.</td>
      <td>\t Ma famille est petite. \n</td>
    </tr>
    <tr>
      <th>50725</th>
      <td>Look at this picture.</td>
      <td>\t Regardez cette photo. \n</td>
    </tr>
    <tr>
      <th>44034</th>
      <td>This guy is a loser.</td>
      <td>\t Ce mec est un nullard. \n</td>
    </tr>
    <tr>
      <th>75380</th>
      <td>I've told you the truth.</td>
      <td>\t Je vous ai dit la vérité. \n</td>
    </tr>
    <tr>
      <th>72934</th>
      <td>How long will this last?</td>
      <td>\t Combien de temps ceci durera-t-il ? \n</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-ffc688a2-4b85-4ecf-8d47-8f465dacfb9f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-ffc688a2-4b85-4ecf-8d47-8f465dacfb9f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-ffc688a2-4b85-4ecf-8d47-8f465dacfb9f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-af933c31-8927-458a-8524-cdc633f57524">
  <button class="colab-df-quickchart" onclick="quickchart('df-af933c31-8927-458a-8524-cdc633f57524')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-af933c31-8927-458a-8524-cdc633f57524 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>




  문자 집합을 구축하기 위해서 set 자료형을 사용하는 것을 확인할 수 있다. Set 자료형의 경우 중복되는 값은 저장할 수 없기에 char data set을 생성할 때 활용할 수 있다.

  lines.src와 lines.tar 코드를 활용하여 각 열에 대한 정보를 행 단위로 받는 것이고,
  for char in line 코드를 문자열 line에 사용하여 각 문자로 자동적인 parsing이 가능하도록 하였다.


```python
# 문자 집합 구축
src_vocab = set()
for line in lines.src: # 1줄씩 읽음
    for char in line: # 1개의 문자씩 읽음
        src_vocab.add(char)

tar_vocab = set()
for line in lines.tar:
    for char in line:
        tar_vocab.add(char)

```


```python
src_vocab_size = len(src_vocab) + 1
tar_vocab_size = len(tar_vocab) + 1
print('source 문장의 char 집합 :',src_vocab_size)
print('target 문장의 char 집합 :',tar_vocab_size)
```

    source 문장의 char 집합 : 82
    target 문장의 char 집합 : 106
    

src_vocab와 tar_vocab 집합을 임의로 정렬하고, 일부를 출력해준다.

set 자료형은 정렬 sorted(set()) 형태로 사용하지 않는다면 인덱스 슬라이싱을 사용할 수 없기에 sorted 명령어를 사용하여 정렬한 후에, 해당 값을 출력해주어야 한다.


```python
src_vocab = sorted(list(src_vocab))
tar_vocab = sorted(list(tar_vocab))
print(src_vocab[45:75])
print(tar_vocab[45:75])
```

    ['W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    ['T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w']
    

enumerate() 함수의 경우, 데이터의 인덱스와 값을 반환해주는 역할을 한다.
따라서 사전의 형태로 각 word에 대응되는 인덱스를 저장해주는 코드이다.

해당 dictionary를 활용하여 문자열을 정수 데이터로 전환하는 과정에 사용할 수 있을 것이다.


```python
src_to_index = dict([(word, i+1) for i, word in enumerate(src_vocab)])
tar_to_index = dict([(word, i+1) for i, word in enumerate(tar_vocab)])
print(src_to_index)
print(tar_to_index)
```

    {' ': 1, '!': 2, '"': 3, '$': 4, '%': 5, '&': 6, "'": 7, ',': 8, '-': 9, '.': 10, '/': 11, '0': 12, '1': 13, '2': 14, '3': 15, '4': 16, '5': 17, '6': 18, '7': 19, '8': 20, '9': 21, ':': 22, '?': 23, 'A': 24, 'B': 25, 'C': 26, 'D': 27, 'E': 28, 'F': 29, 'G': 30, 'H': 31, 'I': 32, 'J': 33, 'K': 34, 'L': 35, 'M': 36, 'N': 37, 'O': 38, 'P': 39, 'Q': 40, 'R': 41, 'S': 42, 'T': 43, 'U': 44, 'V': 45, 'W': 46, 'X': 47, 'Y': 48, 'Z': 49, 'a': 50, 'b': 51, 'c': 52, 'd': 53, 'e': 54, 'f': 55, 'g': 56, 'h': 57, 'i': 58, 'j': 59, 'k': 60, 'l': 61, 'm': 62, 'n': 63, 'o': 64, 'p': 65, 'q': 66, 'r': 67, 's': 68, 't': 69, 'u': 70, 'v': 71, 'w': 72, 'x': 73, 'y': 74, 'z': 75, '\xa0': 76, '°': 77, 'é': 78, 'ï': 79, '’': 80, '€': 81}
    {'\t': 1, '\n': 2, ' ': 3, '!': 4, '"': 5, '$': 6, '%': 7, '&': 8, "'": 9, '(': 10, ')': 11, ',': 12, '-': 13, '.': 14, '0': 15, '1': 16, '2': 17, '3': 18, '4': 19, '5': 20, '6': 21, '7': 22, '8': 23, '9': 24, ':': 25, '?': 26, 'A': 27, 'B': 28, 'C': 29, 'D': 30, 'E': 31, 'F': 32, 'G': 33, 'H': 34, 'I': 35, 'J': 36, 'K': 37, 'L': 38, 'M': 39, 'N': 40, 'O': 41, 'P': 42, 'Q': 43, 'R': 44, 'S': 45, 'T': 46, 'U': 47, 'V': 48, 'W': 49, 'X': 50, 'Y': 51, 'Z': 52, 'a': 53, 'b': 54, 'c': 55, 'd': 56, 'e': 57, 'f': 58, 'g': 59, 'h': 60, 'i': 61, 'j': 62, 'k': 63, 'l': 64, 'm': 65, 'n': 66, 'o': 67, 'p': 68, 'q': 69, 'r': 70, 's': 71, 't': 72, 'u': 73, 'v': 74, 'w': 75, 'x': 76, 'y': 77, 'z': 78, '\xa0': 79, '«': 80, '»': 81, 'À': 82, 'Ç': 83, 'É': 84, 'Ê': 85, 'Ô': 86, 'à': 87, 'â': 88, 'ç': 89, 'è': 90, 'é': 91, 'ê': 92, 'ë': 93, 'î': 94, 'ï': 95, 'ô': 96, 'ù': 97, 'û': 98, 'œ': 99, '\u2009': 100, '\u200b': 101, '‘': 102, '’': 103, '\u202f': 104, '‽': 105}
    

lines.src에서 line에 해당하는 source값을 하나씩 받아온다.

```python
for char in line
```
위 구문을 통해서 문자열 parsing을 사용하여 character를 하나씩 받아오고
```python
encoded_line.append(src_to_index[char])
```
위 코드를 사용하여 미리 선언한 list에 각 문자열에 대응되는 숫자를 append 시켜준다.
append의 경우, list의 차원을 증가시키지 않으므로, 1차원 리스트 안에서 여러개의 정수가 존재하는  형태가 될 것이다.

```python
encoder_input.append(encoded_line)
```
위 코드를 사용하여 encoder_input이라는 저장 리스트 안에 리스트를 append 시켜준다.

for문을 사용하여 계속해서 1차원 리스트를 리스트 안에 append시켜주므로, 결국 2차원 리스트 형태로 저장이 될 것이다.

## 저장 형태

저장 형태를 잘 살펴보면 2차원 리스트라는 것은 쉽게 파악이 가능하다.

세부적으로 살펴보자면, 2차원 리스트의 하위 리스트는 한 line을 통째로 바꾸는 모습을 확인할 수 있다.


```python
encoder_input = []

# 1개의 문장
for line in lines.src:
  encoded_line = []
  # 각 줄에서 1개의 char
  for char in line:
    # 각 char을 정수로 변환
    encoded_line.append(src_to_index[char])
  encoder_input.append(encoded_line)
print('source 문장의 정수 인코딩 :',encoder_input[:5])
```

    source 문장의 정수 인코딩 : [[30, 64, 10], [30, 64, 10], [30, 64, 10], [30, 64, 10], [31, 58, 10]]
    


```python
decoder_input = []
for line in lines.tar:
  encoded_line = []
  for char in line:
    encoded_line.append(tar_to_index[char])
  decoder_input.append(encoded_line)
print('target 문장의 정수 인코딩 :',decoder_input[:5])

```

    target 문장의 정수 인코딩 : [[1, 3, 48, 53, 3, 4, 3, 2], [1, 3, 39, 53, 70, 55, 60, 57, 14, 3, 2], [1, 3, 31, 66, 3, 70, 67, 73, 72, 57, 3, 4, 3, 2], [1, 3, 28, 67, 73, 59, 57, 3, 4, 3, 2], [1, 3, 45, 53, 64, 73, 72, 3, 4, 3, 2]]
    


```python
decoder_target = []
for line in lines.tar:
  timestep = 0
  encoded_line = []
  for char in line:
    if timestep > 0:
      encoded_line.append(tar_to_index[char])
    timestep = timestep + 1
  decoder_target.append(encoded_line)
print('target 문장 레이블의 정수 인코딩 :',decoder_target[:5])
```

    target 문장 레이블의 정수 인코딩 : [[3, 48, 53, 3, 4, 3, 2], [3, 39, 53, 70, 55, 60, 57, 14, 3, 2], [3, 31, 66, 3, 70, 67, 73, 72, 57, 3, 4, 3, 2], [3, 28, 67, 73, 59, 57, 3, 4, 3, 2], [3, 45, 53, 64, 73, 72, 3, 4, 3, 2]]
    


```python
max_src_len = max([len(line) for line in lines.src])
max_tar_len = max([len(line) for line in lines.tar])
print('source 문장의 최대 길이 :',max_src_len)
print('target 문장의 최대 길이 :',max_tar_len)
# 문장의 길이를 전부 동일하게 맞추기 위해서 max 함수를 사용함.
```

    source 문장의 최대 길이 : 27
    target 문장의 최대 길이 : 76
    


```python
encoder_input = pad_sequences(encoder_input, maxlen=max_src_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_tar_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_tar_len, padding='post')
print('source 문장의 정수 인코딩 :',encoder_input[:10])
print('target 문장의 정수 인코딩 :',decoder_input[:10])
print('target 문장 레이블의 정수 인코딩 :',decoder_target[:10])
# 문장 길이를 동일하게 만들어주기 위해 padding을 사용함.
# 최대 입력 소스와 최대 길이, padding 옵션에는 post를 사용했으므로 원래 데이터 이후의 값이라면 0이 들어간다.
```

    source 문장의 정수 인코딩 : [[30 64 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0]
     [30 64 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0]
     [30 64 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0]
     [30 64 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0]
     [31 58 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0]
     [31 58 10  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0]
     [41 70 63  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0]
     [41 70 63  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0]
     [41 70 63  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0]
     [41 70 63  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
       0  0  0]]
    target 문장의 정수 인코딩 : [[  1   3  48  53   3   4   3   2   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  1   3  39  53  70  55  60  57  14   3   2   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  1   3  31  66   3  70  67  73  72  57   3   4   3   2   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  1   3  28  67  73  59  57   3   4   3   2   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  1   3  45  53  64  73  72   3   4   3   2   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  1   3  45  53  64  73  72  14   3   2   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  1   3  29  67  73  70  71 104   4   3   2   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  1   3  29  67  73  70  57  78 104   4   3   2   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  1   3  42  70  57  66  57  78   3  74  67  71   3  62  53  65  54  57
       71   3  87   3  74  67  71   3  55  67  73  71   3   4   3   2   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  1   3  32  61  64  57   3   4   3   2   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]]
    target 문장 레이블의 정수 인코딩 : [[  3  48  53   3   4   3   2   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  3  39  53  70  55  60  57  14   3   2   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  3  31  66   3  70  67  73  72  57   3   4   3   2   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  3  28  67  73  59  57   3   4   3   2   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  3  45  53  64  73  72   3   4   3   2   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  3  45  53  64  73  72  14   3   2   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  3  29  67  73  70  71 104   4   3   2   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  3  29  67  73  70  57  78 104   4   3   2   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  3  42  70  57  66  57  78   3  74  67  71   3  62  53  65  54  57  71
        3  87   3  74  67  71   3  55  67  73  71   3   4   3   2   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]
     [  3  32  61  64  57   3   4   3   2   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
        0   0   0   0]]
    

단어 단위 번역기로 사용하는 경우, Embedding Layer를 사용하여 워드 임베딩을 사용해야 하지만. 이 경우 문자 단위 번역기에 해당하므로 Embedding Layer를 사용하지 않아도 된다.

띠라서 one hot vector를 사용한다.


```python
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)
print('source 문장의 one hot 인코딩 :',encoder_input[:10])
print('target 문장의 one hot 인코딩 :',decoder_input[:10])
print('target 문장 레이블의 one hot 인코딩 :',decoder_target[:10])
print(encoder_input.shape)
print(decoder_input.shape)
print(decoder_target.shape)
```

    source 문장의 one hot 인코딩 : [[[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     ...
    
     [[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]]
    target 문장의 one hot 인코딩 : [[[0. 1. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 1. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 1. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     ...
    
     [[0. 1. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 1. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 1. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]]
    target 문장 레이블의 one hot 인코딩 : [[[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     ...
    
     [[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]
    
     [[0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      [0. 0. 0. ... 0. 0. 0.]
      ...
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]
      [1. 0. 0. ... 0. 0. 0.]]]
    (100000, 27, 82)
    (100000, 76, 106)
    (100000, 76, 106)
    


```python
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
import numpy as np
if tf.config.list_physical_devices('GPU'):
  tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
```

## **Encoder Input**

Tensorflow Input을 사용하여 입력 시퀀스의 형태를 정의한다.
shape를 (None, src_vocab_size)를 사용하여 각 시퀀스의 크기를 지정해준다.

Input layer에 0ne-hot encoded input이 들어간다. 또한, encoder에 들어가는 source의 크기는 각 단어별로 src_vocab_size 길이의 리스트 형태를 가져야 하므로 해당 shape를 사용하는 것으로 생각할 수 있다.

## **Encoder LSTM**

우리가 Seq2Seq에 사용할 RNN Unit을 지정하는 것이라고 생각할 수 있다. 이번 예제에서는 LSTM 유닛을 사용하여 RNN을 구현할 것이기에, 이를 사용하는 코드를 작성하는 것이고, return_state = True의 코드를 사용하여 최종 state를 반환하도록 했다. 이 최종 state는 Decoder에서 사용된다.

<span style="background-color: #FFFF00"> LSTM의 구조에 관해서 더 공부해보아야 한다. 왜 256unit이라는 것을 사용하는지 다시 공부해야겠다. </span>

LSTM의 구조에 관해서 더 자세히 공부해볼 예정이지만, unit의 크기라는 것은 쉽게 말해서 hidden state의 크기를 결정하는 인자라고 볼 수 있다. 256unit이라는 것은 hidden state가 256개의 항의 개수, 256차원을 가진다라고 생각하면 된다.

```python
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
```
위 코드를 사용하여 one hot encoded data가 LSTM 레이어를 통과한 이후의 결과값을 저장할 수 있다. LSTM 레이어를 통과한 이후의 결과값은 3개가 존재하는데, 먼저 output, hidden state, cell state이다.

다른 부분들은 쉽게 이해되지만, **cell state에 대한 부분이 제대로 이해되지 않는다.** 여기에서는 간단하게 LSTM의 장기적인 의존성을 관리하는 데에 사용한다라고 이해할 수 있다.

**결과적으로 hidden state와 cell state를 사용하여 encoder의 최종 state를 생성하고(list 형태로) 이를 활용하여 Decoder의 Input값을 생성할 수 있다.**


```python
encoder_inputs = Input(shape=(None, src_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True)

# encoder_outputs은 여기서는 불필요
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# LSTM은 바닐라 RNN과는 달리 상태가 두 개. 은닉 상태와 셀 상태.
encoder_states = [state_h, state_c]
```

## **Decoder Input**

decoder input은 tensorflow의 Input을 사용하여 encoder의 input과 동일하게, one hot encoded data의 각 문자의 크기(변주범위)를 제공하는 것으로 정의된다.

## **Decoder LSTM**

encoder에서 RNN 기반의 LSTM cell이 사용되었으므로, decoder에서도 Lstm이 사용되어야 한다. 따라서 tensorflow의 LSTM() 함수를 사용하여 LSTM 레이어를 작성해주었다.

return_sequences 옵션을 사용하여 모든 레이어에 대한 출력을 시퀀스로 반환해준다.

* return_sequences = True

  return sequence 옵션이 True로 설정된다면, 매 시점의 출력을 저장하여 (batch_size, sequence_length, units)의 형태로 출력값을 지니게 된다. Decoder에서 이 옵션을 true로 설정한다면 매 시점의 정보 형태를 유지한다는 의미이므로 Seq2Seq 모델이나 시계열 데이터 예측 등에 사용된다.

* return_sequences = False

  return sequence 옵션이 False로 설정된다면, 매 시점의 정보를 저장하는 것이 아니라 마지막 하나의 출력만을 저장하는 RNN 유닛이 생성된다. 이 경우 전체 데이터를 활용하여 하나의 정보로 압축하는 형태로 사용하는 것이기에, 긍정/부정이나 카테고리 분류 등등에 사용할 수 있다. 이 옵션은 출력의 형태가 (batch_size, units)의 형태이다.

우리는 Seq2Seq 모델에 대해서 공부하고 있으므로, 출력의 형태를 유지해야 하며 이 옵션은 True로 설정해두어야 한다.

****
추가적으로, Lstm의 initial state에 해당하는 값으로 encoder_states를 사용하는 모습을 확인할 수 있는데, 이는 Encoder의 최종 출력층에 해당하는 Context Vector를 initial state로 활용하는 모습이다.

## Decoder Output

Decoder의 동작을 생각해보자. 인코더에서 받은 Context Vector를 바탕으로 출력값을 생성해내는 역할을 하는데, 이 특성을 이해한다면, LSTM의 출력값에 해당하는 output, hidden state, cell state중에서 hidden, cell state는 필요하지 않다는 것을 생각할 수 있다.

해당 처리를 거쳐서 나온 Output에 대해서 생각해보자.decoder_outputs의 데이터를 Dense layer를 만들어 해당 레이어를 통과시키는 모습을 볼 수 있다. Dense Layer는 pytorch의 nn.Linear와 같은 FC Layer에 해당한다고 생각하면 된다. 선형 변환과 같은 Full connected layer를 정의한 뒤, 해당 결과값을 바탕으로 softmax를 사용하여 각 단어별 확률을 계산하는 방식을 사용한다.

<span style="color:blue">**여기서 왜 activation function 이전에 Fully connected layer를 붙여주어야 하는지에 대한 의문점이 생겼다.**</span>

이는 다른 post에서 정리하겠다.

## **Model 훈련**

Encoder의 hidden state, cell state, Decoder의 output을 받는 Model을 구성한다.

Pytorch 에서는 optimizer로 구현된 것과 critieon으로 표현하는 loss값을 comile 함수를 통해서 지정해준다. 이후 모델의 validation 분할 비율과 배치 사이즈 등등을 결정해 모델 피팅을 진행한다.


```python
decoder_inputs = Input(shape=(None, tar_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)

# 디코더에게 인코더의 은닉 상태, 셀 상태를 전달.
decoder_outputs, _, _= decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_softmax_layer = Dense(tar_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
```


```python
model.fit(x=[encoder_input, decoder_input], y=decoder_target, batch_size=2048, epochs=400, validation_split=0.2)
```

    Epoch 1/400
    40/40 [==============================] - 16s 303ms/step - loss: 1.8627 - val_loss: 1.8056
    Epoch 2/400
    40/40 [==============================] - 8s 202ms/step - loss: 1.2851 - val_loss: 1.4231
    Epoch 3/400
    40/40 [==============================] - 8s 199ms/step - loss: 1.1586 - val_loss: 1.3593
    Epoch 4/400
    40/40 [==============================] - 8s 202ms/step - loss: 1.0881 - val_loss: 1.3415
    Epoch 5/400
    40/40 [==============================] - 8s 203ms/step - loss: 1.0328 - val_loss: 1.2631
    Epoch 6/400
    40/40 [==============================] - 8s 204ms/step - loss: 0.9685 - val_loss: 1.2279
    Epoch 7/400
    40/40 [==============================] - 8s 208ms/step - loss: 0.9196 - val_loss: 1.1424
    Epoch 8/400
    40/40 [==============================] - 8s 208ms/step - loss: 0.8703 - val_loss: 1.0723
    Epoch 9/400
    40/40 [==============================] - 8s 210ms/step - loss: 0.8290 - val_loss: 1.0398
    Epoch 10/400
    40/40 [==============================] - 8s 211ms/step - loss: 0.7976 - val_loss: 1.0031
    Epoch 11/400
    40/40 [==============================] - 9s 213ms/step - loss: 0.7740 - val_loss: 0.9795
    Epoch 12/400
    40/40 [==============================] - 9s 214ms/step - loss: 0.7555 - val_loss: 0.9554
    Epoch 13/400
    40/40 [==============================] - 9s 214ms/step - loss: 0.7397 - val_loss: 0.9376
    Epoch 14/400
    40/40 [==============================] - 9s 218ms/step - loss: 0.7256 - val_loss: 0.9271
    Epoch 15/400
    40/40 [==============================] - 9s 218ms/step - loss: 0.7129 - val_loss: 0.9104
    Epoch 16/400
    40/40 [==============================] - 9s 220ms/step - loss: 0.7023 - val_loss: 0.8979
    Epoch 17/400
    40/40 [==============================] - 9s 221ms/step - loss: 0.6909 - val_loss: 0.8834
    Epoch 18/400
    40/40 [==============================] - 9s 223ms/step - loss: 0.6820 - val_loss: 0.9196
    Epoch 19/400
    40/40 [==============================] - 9s 225ms/step - loss: 0.6736 - val_loss: 0.8752
    Epoch 20/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.6634 - val_loss: 0.8582
    Epoch 21/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.6552 - val_loss: 0.8546
    Epoch 22/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.6475 - val_loss: 0.8441
    Epoch 23/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.6397 - val_loss: 0.8327
    Epoch 24/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.6322 - val_loss: 0.8164
    Epoch 25/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.6270 - val_loss: 0.8074
    Epoch 26/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.6194 - val_loss: 0.8001
    Epoch 27/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.6135 - val_loss: 0.7950
    Epoch 28/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.6077 - val_loss: 0.7944
    Epoch 29/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.6016 - val_loss: 0.7912
    Epoch 30/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.5967 - val_loss: 0.7713
    Epoch 31/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.5908 - val_loss: 0.7710
    Epoch 32/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.5862 - val_loss: 0.7590
    Epoch 33/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.5811 - val_loss: 0.7490
    Epoch 34/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.5765 - val_loss: 0.7401
    Epoch 35/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.5719 - val_loss: 0.7496
    Epoch 36/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.5673 - val_loss: 0.7353
    Epoch 37/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.5628 - val_loss: 0.7353
    Epoch 38/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.5585 - val_loss: 0.7277
    Epoch 39/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.5539 - val_loss: 0.7241
    Epoch 40/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.5505 - val_loss: 0.7252
    Epoch 41/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.5458 - val_loss: 0.7233
    Epoch 42/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.5426 - val_loss: 0.7138
    Epoch 43/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.5382 - val_loss: 0.7032
    Epoch 44/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.5346 - val_loss: 0.7102
    Epoch 45/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.5305 - val_loss: 0.7002
    Epoch 46/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.5265 - val_loss: 0.6916
    Epoch 47/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.5231 - val_loss: 0.6896
    Epoch 48/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.5198 - val_loss: 0.6773
    Epoch 49/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.5155 - val_loss: 0.6877
    Epoch 50/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.5127 - val_loss: 0.6739
    Epoch 51/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.5092 - val_loss: 0.6704
    Epoch 52/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.5063 - val_loss: 0.6644
    Epoch 53/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.5029 - val_loss: 0.6690
    Epoch 54/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.4997 - val_loss: 0.6598
    Epoch 55/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.4970 - val_loss: 0.6559
    Epoch 56/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4940 - val_loss: 0.6548
    Epoch 57/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4910 - val_loss: 0.6535
    Epoch 58/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.4884 - val_loss: 0.6467
    Epoch 59/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.4857 - val_loss: 0.6472
    Epoch 60/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.4830 - val_loss: 0.6415
    Epoch 61/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.4802 - val_loss: 0.6272
    Epoch 62/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4773 - val_loss: 0.6369
    Epoch 63/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4755 - val_loss: 0.6252
    Epoch 64/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4733 - val_loss: 0.6256
    Epoch 65/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4701 - val_loss: 0.6284
    Epoch 66/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4674 - val_loss: 0.6173
    Epoch 67/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4652 - val_loss: 0.6222
    Epoch 68/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4630 - val_loss: 0.6145
    Epoch 69/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4608 - val_loss: 0.6064
    Epoch 70/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.4583 - val_loss: 0.6104
    Epoch 71/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4563 - val_loss: 0.6060
    Epoch 72/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.4536 - val_loss: 0.6058
    Epoch 73/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4519 - val_loss: 0.6044
    Epoch 74/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4493 - val_loss: 0.6001
    Epoch 75/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4471 - val_loss: 0.5982
    Epoch 76/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4450 - val_loss: 0.5908
    Epoch 77/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4433 - val_loss: 0.5904
    Epoch 78/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4412 - val_loss: 0.5927
    Epoch 79/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4394 - val_loss: 0.5879
    Epoch 80/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4373 - val_loss: 0.5803
    Epoch 81/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4355 - val_loss: 0.5856
    Epoch 82/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4340 - val_loss: 0.5857
    Epoch 83/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4313 - val_loss: 0.5833
    Epoch 84/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4303 - val_loss: 0.5738
    Epoch 85/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4283 - val_loss: 0.5733
    Epoch 86/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4261 - val_loss: 0.5747
    Epoch 87/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4253 - val_loss: 0.5687
    Epoch 88/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.4228 - val_loss: 0.5755
    Epoch 89/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4213 - val_loss: 0.5662
    Epoch 90/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4199 - val_loss: 0.5610
    Epoch 91/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.4182 - val_loss: 0.5645
    Epoch 92/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4166 - val_loss: 0.5678
    Epoch 93/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4151 - val_loss: 0.5583
    Epoch 94/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4142 - val_loss: 0.5546
    Epoch 95/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4120 - val_loss: 0.5596
    Epoch 96/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4112 - val_loss: 0.5500
    Epoch 97/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.4088 - val_loss: 0.5510
    Epoch 98/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4080 - val_loss: 0.5489
    Epoch 99/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4064 - val_loss: 0.5489
    Epoch 100/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.4048 - val_loss: 0.5454
    Epoch 101/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.4038 - val_loss: 0.5488
    Epoch 102/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.4025 - val_loss: 0.5490
    Epoch 103/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.4009 - val_loss: 0.5410
    Epoch 104/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3999 - val_loss: 0.5400
    Epoch 105/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3986 - val_loss: 0.5470
    Epoch 106/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3971 - val_loss: 0.5389
    Epoch 107/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3962 - val_loss: 0.5369
    Epoch 108/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3948 - val_loss: 0.5400
    Epoch 109/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3936 - val_loss: 0.5363
    Epoch 110/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3923 - val_loss: 0.5303
    Epoch 111/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3917 - val_loss: 0.5391
    Epoch 112/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3897 - val_loss: 0.5353
    Epoch 113/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3891 - val_loss: 0.5314
    Epoch 114/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3875 - val_loss: 0.5354
    Epoch 115/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3868 - val_loss: 0.5400
    Epoch 116/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3860 - val_loss: 0.5234
    Epoch 117/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3842 - val_loss: 0.5284
    Epoch 118/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3833 - val_loss: 0.5237
    Epoch 119/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3821 - val_loss: 0.5280
    Epoch 120/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.3808 - val_loss: 0.5288
    Epoch 121/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3799 - val_loss: 0.5174
    Epoch 122/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3791 - val_loss: 0.5177
    Epoch 123/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3778 - val_loss: 0.5183
    Epoch 124/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3769 - val_loss: 0.5193
    Epoch 125/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3761 - val_loss: 0.5183
    Epoch 126/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3749 - val_loss: 0.5152
    Epoch 127/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.3737 - val_loss: 0.5177
    Epoch 128/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3731 - val_loss: 0.5083
    Epoch 129/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3720 - val_loss: 0.5090
    Epoch 130/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.3709 - val_loss: 0.5090
    Epoch 131/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3701 - val_loss: 0.5127
    Epoch 132/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3690 - val_loss: 0.5123
    Epoch 133/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3684 - val_loss: 0.5055
    Epoch 134/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3672 - val_loss: 0.5039
    Epoch 135/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3664 - val_loss: 0.5053
    Epoch 136/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3656 - val_loss: 0.5143
    Epoch 137/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3646 - val_loss: 0.5029
    Epoch 138/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3636 - val_loss: 0.5059
    Epoch 139/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3632 - val_loss: 0.5072
    Epoch 140/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3622 - val_loss: 0.5019
    Epoch 141/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.3612 - val_loss: 0.5001
    Epoch 142/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3604 - val_loss: 0.5016
    Epoch 143/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3600 - val_loss: 0.4969
    Epoch 144/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3587 - val_loss: 0.4953
    Epoch 145/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.3582 - val_loss: 0.4951
    Epoch 146/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3574 - val_loss: 0.5000
    Epoch 147/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3566 - val_loss: 0.4941
    Epoch 148/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3557 - val_loss: 0.4961
    Epoch 149/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.3554 - val_loss: 0.4932
    Epoch 150/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3544 - val_loss: 0.4956
    Epoch 151/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3534 - val_loss: 0.4896
    Epoch 152/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.3529 - val_loss: 0.4960
    Epoch 153/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3523 - val_loss: 0.4882
    Epoch 154/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3513 - val_loss: 0.4917
    Epoch 155/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3506 - val_loss: 0.4898
    Epoch 156/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3505 - val_loss: 0.4862
    Epoch 157/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3492 - val_loss: 0.4912
    Epoch 158/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3486 - val_loss: 0.4879
    Epoch 159/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3480 - val_loss: 0.4907
    Epoch 160/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3472 - val_loss: 0.4846
    Epoch 161/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3465 - val_loss: 0.4834
    Epoch 162/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.3458 - val_loss: 0.4872
    Epoch 163/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3451 - val_loss: 0.4833
    Epoch 164/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.3448 - val_loss: 0.4795
    Epoch 165/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3438 - val_loss: 0.4875
    Epoch 166/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3430 - val_loss: 0.4789
    Epoch 167/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3429 - val_loss: 0.4779
    Epoch 168/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3420 - val_loss: 0.4789
    Epoch 169/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.3413 - val_loss: 0.4778
    Epoch 170/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.3409 - val_loss: 0.4827
    Epoch 171/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3402 - val_loss: 0.4814
    Epoch 172/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3395 - val_loss: 0.4737
    Epoch 173/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3391 - val_loss: 0.4741
    Epoch 174/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3380 - val_loss: 0.4757
    Epoch 175/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3378 - val_loss: 0.4776
    Epoch 176/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3371 - val_loss: 0.4772
    Epoch 177/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3366 - val_loss: 0.4739
    Epoch 178/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3358 - val_loss: 0.4724
    Epoch 179/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.3355 - val_loss: 0.4756
    Epoch 180/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3344 - val_loss: 0.4733
    Epoch 181/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3341 - val_loss: 0.4746
    Epoch 182/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3337 - val_loss: 0.4742
    Epoch 183/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3331 - val_loss: 0.4769
    Epoch 184/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3325 - val_loss: 0.4724
    Epoch 185/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3320 - val_loss: 0.4692
    Epoch 186/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.3314 - val_loss: 0.4684
    Epoch 187/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3309 - val_loss: 0.4692
    Epoch 188/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3302 - val_loss: 0.4681
    Epoch 189/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3302 - val_loss: 0.4723
    Epoch 190/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3291 - val_loss: 0.4661
    Epoch 191/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3287 - val_loss: 0.4647
    Epoch 192/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3283 - val_loss: 0.4622
    Epoch 193/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3277 - val_loss: 0.4683
    Epoch 194/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3271 - val_loss: 0.4623
    Epoch 195/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.3266 - val_loss: 0.4662
    Epoch 196/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3262 - val_loss: 0.4650
    Epoch 197/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.3255 - val_loss: 0.4609
    Epoch 198/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.3253 - val_loss: 0.4626
    Epoch 199/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3247 - val_loss: 0.4644
    Epoch 200/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3241 - val_loss: 0.4655
    Epoch 201/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3238 - val_loss: 0.4700
    Epoch 202/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3232 - val_loss: 0.4577
    Epoch 203/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3225 - val_loss: 0.4605
    Epoch 204/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3222 - val_loss: 0.4572
    Epoch 205/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3216 - val_loss: 0.4594
    Epoch 206/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3211 - val_loss: 0.4618
    Epoch 207/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.3206 - val_loss: 0.4600
    Epoch 208/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3203 - val_loss: 0.4661
    Epoch 209/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3198 - val_loss: 0.4560
    Epoch 210/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3193 - val_loss: 0.4612
    Epoch 211/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3189 - val_loss: 0.4545
    Epoch 212/400
    40/40 [==============================] - 9s 231ms/step - loss: 0.3184 - val_loss: 0.4581
    Epoch 213/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3182 - val_loss: 0.4585
    Epoch 214/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3175 - val_loss: 0.4543
    Epoch 215/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3176 - val_loss: 0.4530
    Epoch 216/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3163 - val_loss: 0.4586
    Epoch 217/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3164 - val_loss: 0.4542
    Epoch 218/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3156 - val_loss: 0.4507
    Epoch 219/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3151 - val_loss: 0.4497
    Epoch 220/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3149 - val_loss: 0.4491
    Epoch 221/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3144 - val_loss: 0.4497
    Epoch 222/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3145 - val_loss: 0.4495
    Epoch 223/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3135 - val_loss: 0.4573
    Epoch 224/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3142 - val_loss: 0.4531
    Epoch 225/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3124 - val_loss: 0.4620
    Epoch 226/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3128 - val_loss: 0.4508
    Epoch 227/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3118 - val_loss: 0.4508
    Epoch 228/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3113 - val_loss: 0.4488
    Epoch 229/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3113 - val_loss: 0.4460
    Epoch 230/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3107 - val_loss: 0.4515
    Epoch 231/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3102 - val_loss: 0.4561
    Epoch 232/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3098 - val_loss: 0.4482
    Epoch 233/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3097 - val_loss: 0.4467
    Epoch 234/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3087 - val_loss: 0.4511
    Epoch 235/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3085 - val_loss: 0.4499
    Epoch 236/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3082 - val_loss: 0.4530
    Epoch 237/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3081 - val_loss: 0.4515
    Epoch 238/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.3075 - val_loss: 0.4479
    Epoch 239/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3071 - val_loss: 0.4437
    Epoch 240/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3066 - val_loss: 0.4431
    Epoch 241/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3063 - val_loss: 0.4467
    Epoch 242/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3059 - val_loss: 0.4524
    Epoch 243/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3059 - val_loss: 0.4442
    Epoch 244/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3045 - val_loss: 0.4450
    Epoch 245/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3047 - val_loss: 0.4432
    Epoch 246/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3044 - val_loss: 0.4442
    Epoch 247/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.3041 - val_loss: 0.4426
    Epoch 248/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3033 - val_loss: 0.4436
    Epoch 249/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3033 - val_loss: 0.4435
    Epoch 250/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3030 - val_loss: 0.4363
    Epoch 251/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3023 - val_loss: 0.4414
    Epoch 252/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3022 - val_loss: 0.4413
    Epoch 253/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3016 - val_loss: 0.4459
    Epoch 254/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3015 - val_loss: 0.4388
    Epoch 255/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.3008 - val_loss: 0.4420
    Epoch 256/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.3007 - val_loss: 0.4391
    Epoch 257/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.3004 - val_loss: 0.4429
    Epoch 258/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2999 - val_loss: 0.4356
    Epoch 259/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2997 - val_loss: 0.4409
    Epoch 260/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2995 - val_loss: 0.4358
    Epoch 261/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2992 - val_loss: 0.4382
    Epoch 262/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2984 - val_loss: 0.4436
    Epoch 263/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2981 - val_loss: 0.4379
    Epoch 264/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2979 - val_loss: 0.4387
    Epoch 265/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2975 - val_loss: 0.4429
    Epoch 266/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2975 - val_loss: 0.4385
    Epoch 267/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2968 - val_loss: 0.4432
    Epoch 268/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.2967 - val_loss: 0.4411
    Epoch 269/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2962 - val_loss: 0.4353
    Epoch 270/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2960 - val_loss: 0.4371
    Epoch 271/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2955 - val_loss: 0.4331
    Epoch 272/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2950 - val_loss: 0.4352
    Epoch 273/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2947 - val_loss: 0.4350
    Epoch 274/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2946 - val_loss: 0.4345
    Epoch 275/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2941 - val_loss: 0.4372
    Epoch 276/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2938 - val_loss: 0.4339
    Epoch 277/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2935 - val_loss: 0.4363
    Epoch 278/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2932 - val_loss: 0.4338
    Epoch 279/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2928 - val_loss: 0.4316
    Epoch 280/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2927 - val_loss: 0.4350
    Epoch 281/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2922 - val_loss: 0.4297
    Epoch 282/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2920 - val_loss: 0.4361
    Epoch 283/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2916 - val_loss: 0.4312
    Epoch 284/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2912 - val_loss: 0.4344
    Epoch 285/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2909 - val_loss: 0.4318
    Epoch 286/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2905 - val_loss: 0.4305
    Epoch 287/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2902 - val_loss: 0.4363
    Epoch 288/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2901 - val_loss: 0.4341
    Epoch 289/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2898 - val_loss: 0.4333
    Epoch 290/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2894 - val_loss: 0.4304
    Epoch 291/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2892 - val_loss: 0.4349
    Epoch 292/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2888 - val_loss: 0.4304
    Epoch 293/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2882 - val_loss: 0.4304
    Epoch 294/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2883 - val_loss: 0.4267
    Epoch 295/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2879 - val_loss: 0.4384
    Epoch 296/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2874 - val_loss: 0.4332
    Epoch 297/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.2874 - val_loss: 0.4293
    Epoch 298/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2870 - val_loss: 0.4312
    Epoch 299/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2865 - val_loss: 0.4263
    Epoch 300/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2863 - val_loss: 0.4289
    Epoch 301/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2862 - val_loss: 0.4273
    Epoch 302/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2856 - val_loss: 0.4314
    Epoch 303/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2856 - val_loss: 0.4255
    Epoch 304/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2852 - val_loss: 0.4246
    Epoch 305/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2848 - val_loss: 0.4255
    Epoch 306/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2846 - val_loss: 0.4288
    Epoch 307/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2843 - val_loss: 0.4289
    Epoch 308/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2842 - val_loss: 0.4250
    Epoch 309/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2835 - val_loss: 0.4276
    Epoch 310/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2834 - val_loss: 0.4277
    Epoch 311/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2832 - val_loss: 0.4220
    Epoch 312/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2827 - val_loss: 0.4256
    Epoch 313/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2825 - val_loss: 0.4380
    Epoch 314/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2824 - val_loss: 0.4238
    Epoch 315/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2820 - val_loss: 0.4233
    Epoch 316/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2817 - val_loss: 0.4224
    Epoch 317/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2814 - val_loss: 0.4232
    Epoch 318/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2812 - val_loss: 0.4257
    Epoch 319/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2809 - val_loss: 0.4236
    Epoch 320/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2804 - val_loss: 0.4217
    Epoch 321/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2807 - val_loss: 0.4225
    Epoch 322/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2799 - val_loss: 0.4233
    Epoch 323/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2800 - val_loss: 0.4243
    Epoch 324/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2796 - val_loss: 0.4250
    Epoch 325/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2789 - val_loss: 0.4277
    Epoch 326/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2792 - val_loss: 0.4258
    Epoch 327/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2786 - val_loss: 0.4257
    Epoch 328/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2784 - val_loss: 0.4244
    Epoch 329/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2780 - val_loss: 0.4235
    Epoch 330/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2779 - val_loss: 0.4267
    Epoch 331/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2778 - val_loss: 0.4219
    Epoch 332/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2774 - val_loss: 0.4248
    Epoch 333/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2769 - val_loss: 0.4193
    Epoch 334/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2770 - val_loss: 0.4223
    Epoch 335/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2766 - val_loss: 0.4210
    Epoch 336/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2763 - val_loss: 0.4207
    Epoch 337/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2759 - val_loss: 0.4232
    Epoch 338/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2759 - val_loss: 0.4181
    Epoch 339/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2754 - val_loss: 0.4180
    Epoch 340/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2751 - val_loss: 0.4189
    Epoch 341/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2752 - val_loss: 0.4251
    Epoch 342/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2749 - val_loss: 0.4207
    Epoch 343/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2747 - val_loss: 0.4211
    Epoch 344/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2742 - val_loss: 0.4199
    Epoch 345/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2741 - val_loss: 0.4196
    Epoch 346/400
    40/40 [==============================] - 9s 231ms/step - loss: 0.2737 - val_loss: 0.4184
    Epoch 347/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2733 - val_loss: 0.4192
    Epoch 348/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2731 - val_loss: 0.4215
    Epoch 349/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2730 - val_loss: 0.4244
    Epoch 350/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2728 - val_loss: 0.4212
    Epoch 351/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2725 - val_loss: 0.4192
    Epoch 352/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2720 - val_loss: 0.4200
    Epoch 353/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2720 - val_loss: 0.4215
    Epoch 354/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2719 - val_loss: 0.4233
    Epoch 355/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2716 - val_loss: 0.4176
    Epoch 356/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2714 - val_loss: 0.4193
    Epoch 357/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2710 - val_loss: 0.4178
    Epoch 358/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2707 - val_loss: 0.4166
    Epoch 359/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2705 - val_loss: 0.4215
    Epoch 360/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2704 - val_loss: 0.4136
    Epoch 361/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2698 - val_loss: 0.4216
    Epoch 362/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2701 - val_loss: 0.4124
    Epoch 363/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2695 - val_loss: 0.4172
    Epoch 364/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2694 - val_loss: 0.4178
    Epoch 365/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2689 - val_loss: 0.4143
    Epoch 366/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2687 - val_loss: 0.4160
    Epoch 367/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2688 - val_loss: 0.4129
    Epoch 368/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2683 - val_loss: 0.4196
    Epoch 369/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2682 - val_loss: 0.4167
    Epoch 370/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2678 - val_loss: 0.4168
    Epoch 371/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2678 - val_loss: 0.4161
    Epoch 372/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2674 - val_loss: 0.4188
    Epoch 373/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2672 - val_loss: 0.4148
    Epoch 374/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2669 - val_loss: 0.4164
    Epoch 375/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2668 - val_loss: 0.4158
    Epoch 376/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2667 - val_loss: 0.4155
    Epoch 377/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2662 - val_loss: 0.4436
    Epoch 378/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2670 - val_loss: 0.4146
    Epoch 379/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2659 - val_loss: 0.4108
    Epoch 380/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2657 - val_loss: 0.4139
    Epoch 381/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2652 - val_loss: 0.4121
    Epoch 382/400
    40/40 [==============================] - 9s 230ms/step - loss: 0.2654 - val_loss: 0.4102
    Epoch 383/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2650 - val_loss: 0.4116
    Epoch 384/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2646 - val_loss: 0.4136
    Epoch 385/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2646 - val_loss: 0.4164
    Epoch 386/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2644 - val_loss: 0.4147
    Epoch 387/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2642 - val_loss: 0.4165
    Epoch 388/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2637 - val_loss: 0.4155
    Epoch 389/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2639 - val_loss: 0.4105
    Epoch 390/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2633 - val_loss: 0.4140
    Epoch 391/400
    40/40 [==============================] - 9s 226ms/step - loss: 0.2632 - val_loss: 0.4111
    Epoch 392/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2627 - val_loss: 0.4126
    Epoch 393/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2627 - val_loss: 0.4145
    Epoch 394/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2627 - val_loss: 0.4123
    Epoch 395/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2625 - val_loss: 0.4116
    Epoch 396/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2621 - val_loss: 0.4114
    Epoch 397/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2622 - val_loss: 0.4138
    Epoch 398/400
    40/40 [==============================] - 9s 229ms/step - loss: 0.2615 - val_loss: 0.4151
    Epoch 399/400
    40/40 [==============================] - 9s 227ms/step - loss: 0.2615 - val_loss: 0.4139
    Epoch 400/400
    40/40 [==============================] - 9s 228ms/step - loss: 0.2614 - val_loss: 0.4113
    




    <keras.src.callbacks.History at 0x798c7e983790>






```python
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
```

## Test 과정에서 필요한 디코더 설계

Test 과정에서 사용하는 디코더의 경우, First Cell에는 Context Vector와 SOS 신호가 들어가며, 두번째 Cell부터는 이전 cell의 hidden state와 cell state, 그리고 output이 현재 cell의 입력값으로 활용된다.

따라서 이를 위해서 필요한 것들은 아래와 같다.

* Context Vector - Encoder_states
* decoder hidden state - state_h
* decoder cell state - state_c

Input layer가 256이라는 것은, LSTM이 256 크기의 hidden state와 cell state를 사용하기 에 설정되는 값이다. 결국, Layer를 설계하는 과정이라고 보면 된다. (실제 값이 아닌 정의)

따라서, decoder에 입력되는 state들을 사전에 정의해 놓은 것이 decoder_states_inputs이다.

입력 Layer를 설정했다면, 이제 LSTM Layer를 정의해야 한다. LSTM Layer의 정의를 위해선 여러가지 옵션이 존재하는데, training 단계에서 정의한 것과 동일한 옵션을 사용해야 한다.

```python
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)
```

따라서 decoder_lstm의 정의 과정이 생략되었지만, LSTM 설정이 완료되었으므로, LSTM의 input과 Input Layer의 Output을 연결시켜주어야 한다. LSTM의 구조에서 입력값으로 Hidden state와 input값이 사용되는 것은 알고 있다. 이를 위해서, 입력값을 사전에 정의해둔 decoder_inputs를 사용하여 형태를 잡는 것이고, hidden state는 최초의 hidden state만을 decoder_state_inputs로 사용하여 형태를 잡은 것이다.

그렇다면, 최초의 cell을 제외한 나머지 형태에 대해서 이어주는 코드를 사용해야 한다. 이를 위해서
```python
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
```
위의 코드를 사용하게 되는데,
lstm Layer의 결과값에 해당하는 output, hidden state, cell state를 전부 받아서 이를 다음 cell에 연결해주어야 한다.

```python
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)
```
해당 구현은 Model 안에 input과 output을 정의하는 것으로 사용할 수 있다.

Model을 정의하는 부분을 잘 살펴보아야 한다.

Model의 인자에 넘겨주는 Input과 Output은 각각 모델이 처리할 **초기 데이터와 최종 데이터** 형태를 나타내는 것에 불과하고, 실제 세부적인 구현은 Input()이나 Dense() 등의 명령어를 통해서 구현하는 것이다.

Test 단계에서는 Model을 통해 Decoder를 구현해야 한다. 따라서 Context Vector와 최초의 input(SOS)를 받고, 최종 출력층에서는 hidden state와 Output에 해당하는 값을 받는 model을 정의한다.

**따라서 최종적인 구성은 Model에 input과 output값을 정의해주는 것으로 마무리되며, model.fit()이나 model.predict() 등등의 명령어에 의해서 input값으로 어떤 값을 사용할지 자동적으로 프레임워크수준에서 결정되게 된다.**



```python
# 이전 시점의 상태들을 저장하는 텐서
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 문장의 다음 단어를 예측하기 위해서 초기 상태(initial_state)를 이전 시점의 상태로 사용.
# 뒤의 함수 decode_sequence()에 동작을 구현 예정
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

# 훈련 과정에서와 달리 LSTM의 리턴하는 은닉 상태와 셀 상태를 버리지 않음.
decoder_states = [state_h, state_c]
decoder_outputs = decoder_softmax_layer(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)
```

## InttoText

앞선 과정에서 dictionary 자료형을 활용하여 문자를 key로, 정수를 value로 하는 번역기를 생성했는데, Test 과정에서 나온 output을 사용하여 다시 문자로 변환하기 위해서는 정수를 key로, 문자를 value로 하는 번역기(Dictionary)를 정의해야 한다.

해당 코드는 .items()를 통해 값을 전부 반환하는 코드를 통해서 받아 반대로 dictionary를 생성하는 코드를 작성할 수 있다.


```python
index_to_src = dict((i, char) for char, i in src_to_index.items())
index_to_tar = dict((i, char) for char, i in tar_to_index.items())
```

아래 decode_sequence 함수의 동작 과정은 다음과 같다.

1. encoder model에서 context vector를 추출한다.
2. decoder model의 입력 sequence에 해당하는 <sos>신호를 만들어낸다.이는 특정 자리에 1의 값을 대입하는 방식으로 이루어진다.
3. SOS 대입 이후 생성된 데이터를 이용하여 정답지 Index를 생성하고, 해당 인덱스를 바탕으로 다음 시점의 입력과 정답지를 생성한다.
4. 위 과정을 반복하여 정답지에 EOS 표지가 나오거나 최대 길이에 도달한다면 이를 종료하고 값을 반환한다.

model.predict()의 입력에 해당 정답지를 사용한다는 것을 통해서 LSTM 등의 구조에서 hidden state는 자동적으로 관리되고, input의 값에만 집중한다는 사실을 확인할 수 있었다.


```python
def decode_sequence(input_seq):
  # 입력으로부터 인코더의 상태를 얻음
  states_value = encoder_model.predict(input_seq)

  # <SOS>에 해당하는 원-핫 벡터 생성
  target_seq = np.zeros((1, 1, tar_vocab_size))
  target_seq[0, 0, tar_to_index['\t']] = 1.

  stop_condition = False
  decoded_sentence = ""

  # stop_condition이 True가 될 때까지 루프 반복
  while not stop_condition:
    # 이점 시점의 상태 states_value를 현 시점의 초기 상태로 사용
    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

    # 예측 결과를 문자로 변환
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_char = index_to_tar[sampled_token_index]

    # 현재 시점의 예측 문자를 예측 문장에 추가
    decoded_sentence += sampled_char

    # <eos>에 도달하거나 최대 길이를 넘으면 중단.
    if (sampled_char == '\n' or
        len(decoded_sentence) > max_tar_len):
        stop_condition = True

    # 현재 시점의 예측 결과를 다음 시점의 입력으로 사용하기 위해 저장
    target_seq = np.zeros((1, 1, tar_vocab_size))
    target_seq[0, 0, sampled_token_index] = 1.

    # 현재 시점의 상태를 다음 시점의 상태로 사용하기 위해 저장
    states_value = [h, c]

  return decoded_sentence

```


```python
for seq_index in [3,50,100,300,1001]: # 입력 문장의 인덱스
  input_seq = encoder_input[seq_index:seq_index+1]
  decoded_sentence = decode_sequence(input_seq)
  print(35 * "-")
  print('입력 문장:', lines.src[seq_index])
  print('정답 문장:', lines.tar[seq_index][2:len(lines.tar[seq_index])-1]) # '\t'와 '\n'을 빼고 출력
  print('번역 문장:', decoded_sentence[1:len(decoded_sentence)-1]) # '\n'을 빼고 출력

```

    1/1 [==============================] - 0s 386ms/step
    1/1 [==============================] - 0s 408ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    -----------------------------------
    입력 문장: Go.
    정답 문장: Bouge ! 
    번역 문장: Autrends ! 
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    -----------------------------------
    입력 문장: Hello!
    정답 문장: Bonjour ! 
    번역 문장: Ainez-nous ! 
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    -----------------------------------
    입력 문장: Got it!
    정답 문장: J'ai pigé ! 
    번역 문장: Aidez Tom. 
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 22ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    -----------------------------------
    입력 문장: Go home.
    정답 문장: Rentre à la maison. 
    번역 문장: Va tour le choix. 
    1/1 [==============================] - 0s 18ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 21ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 19ms/step
    1/1 [==============================] - 0s 20ms/step
    1/1 [==============================] - 0s 20ms/step
    -----------------------------------
    입력 문장: Get going.
    정답 문장: En avant. 
    번역 문장: Décure. 
    
