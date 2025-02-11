---
layout: post
title:  "Deepseek Coder Enviroment Setting"
date:   2024-02-07 00:10:00 +0900
categories: [NLP, Enviroment]   
---
## DeepSeek Coder Installation

Deepseek coder 모델 설치를 위해 **Ollama** 설치를 진행해야 한다.
**Ollama**는 Open Source Model을 Local 환경에서 구동할 수 있도록 해준다. 설치할 환경은 아래와 같다.

![Enviroment](/assets/img/Enviroment/DeepSeekCoder/image.png)

Ollama 공식 사이트에 나온 것처럼 간단하게 설치를 진행할 수 있다.
![Install](/assets/img/Enviroment/DeepSeekCoder/2.png)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
```bash
root@hyeongjun:/# curl -fsSL https://ollama.com/install.sh | sh
>>> Installing ollama to /usr/local
>>> Downloading Linux amd64 bundle
######################################################################## 100.0%
>>> Creating ollama user...
>>> Adding ollama user to render group...
>>> Adding ollama user to video group...
>>> Adding current user to ollama group...
>>> Creating ollama systemd service...
WARNING: systemd is not running
>>> NVIDIA GPU installed.
>>> The Ollama API is now available at 127.0.0.1:11434.
>>> Install complete. Run "ollama" from the command line
```
설치는 위와 같이 진행된다.

```bash
root@hyeongjun:/# ollama
Usage:
  ollama [flags]
  ollama [command]

Available Commands:
  serve       Start ollama
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  stop        Stop a running model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for ollama
  -v, --version   Show version information

Use "ollama [command] --help" for more information about a command.
```
위와 같은 화면이 출력되었으면, 정상적으로 설치된 것이다.

Ollama를 설치한 이후 원하는 모델을 다운로드 받아 사용할 수 있다. 나는 DeepSeek Coder V2와 V1을 사용할 것이기에 해당 모델을 다운해주었다.
더 많은 모델은 (https://ollama.com/)에 접속해 찾아볼 수 있다.

```bash
ollama run deepseek-coder-v2:16b
ollama run deepseek-coder-v2:236b

ollama pull deepseek-coder-v2:16b
ollama pull deepseek-coder-v2:236b
```

위 코드 실행 시에 **Error Could not connect to Ollama app.** 오류가 발생하면 새로운 터미널에 아래 코드를 실행 시키고, 다시 진행하면 된다.
```bash
ollama serve
```

```bash
root@hyeongjun:/data/fault_localization/python# ollama run deepseek-coder-v2:16b
pulling manifest 
pulling 5ff0abeeac1d... 100% ▕███████████████████████████████████████████████████████████ ▏ 8.9 GB/8.9 GB  114 MB/s      0s
pulling 22091531faf0... 100% ▕████████████████████████████████████████████████████████████▏  705 B                         
pulling 4bb71764481f... 100% ▕████████████████████████████████████████████████████████████▏  13 KB                         
pulling 1c8f573e830c... 100% ▕████████████████████████████████████████████████████████████▏ 1.1 KB                         
pulling 19f2fb9e8bc6... 100% ▕████████████████████████████████████████████████████████████▏   32 B                         
pulling 34488e453cfe... 100% ▕████████████████████████████████████████████████████████████▏  568 B                         
verifying sha256 digest 
writing manifest 
success 
```
설치는 위와 같이 진행되고, 예시 sample을 넣어보면 아래와 같이 제대로 작성해주는 것을 확인할 수 있다.

**Prompt:** Generate python code that save double dictionary data into jsonl file

 Sure! Below is an example of how you can save a nested (double) dictionary in Python to a JSONL (JSON Lines) file. A 
JSONL file contains one JSON object per line, which makes it useful for handling large datasets efficiently.

Here's the code:

```python
import json

# Example double dictionary data
data = {
    "user1": {
        "name": "Alice",
        "age": 30,
        "address": {
            "street": "123 Elm St",
            "city": "Wonderland"
        }
    },
    "user2": {
        "name": "Bob",
        "age": 25,
        "address": {
            "street": "456 Maple Ave",
            "city": "Metropolis"
        }
    }
}

# Function to save nested dictionary to a JSONL file
def save_nested_dict_to_jsonl(data, filename):
    with open(filename, 'w') as f:
        for key, value in data.items():
            json_line = json.dumps(value)
            f.write(json_line + '\n')

# Specify the output JSONL file name
output_file = 'nested_dict.jsonl'

# Save the nested dictionary to a JSONL file
save_nested_dict_to_jsonl(data, output_file)
```

### Explanation:
1. **Data Structure**: The `data` variable is a nested dictionary where each key corresponds to another dictionary 
representing a user and their details including an address.
2. **Function Definition**: The `save_nested_dict_to_jsonl` function takes the data (a nested dictionary) and a 
filename as arguments. It iterates over the keys and values of the outer dictionary, converting each inner dictionary 
to a JSON string using `json.dumps`. Each JSON string is written to the file on a new line.
3. **Output File**: The output file is specified by `output_file` which is set to 'nested_dict.jsonl'.
4. **Calling the Function**: The function is called with the data and the filename as arguments.

This code will create a JSONL file where each line represents a user's details in the nested dictionary format. You 
can then read these files back into Python using libraries like `json` or specialized tools for handling JSON Lines.

