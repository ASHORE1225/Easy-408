#  在人工智能飞速发展的当下，基于 LlamaIndex 与 InternLM RAG 技术打造的 408 问题回答系统，为计算机专业考研学生以及相关知识的学习者，开启了一场别开生面的学习之旅。此系统完美融合了 LlamaIndex 卓越的索引构建能力与高效的数据组织能力，搭配基于 InternLM 模型的检索增强生成（RAG）技术，堪称珠联璧合。LlamaIndex 能够有条不紊地将海量 408 考研核心知识，像数据结构里复杂的算法逻辑、计算机组成原理中精妙的硬件架构、操作系统的进程管理机制，以及计算机网络的协议体系等，精心构建成一套条理清晰、易于快速检索的索引结构。而 InternLM RAG 技术更是锦上添花，在生成回答内容时，能精准关联从索引库中提取的相关知识点，从而让输出的答案兼具准确性与专业性。当用户输入 408 相关问题后，系统会即刻启动。LlamaIndex 如同一位训练有素的搜索专家，迅速从庞大的知识储备中定位出最契合的内容片段，紧接着 InternLM RAG 模型以这些片段为基石，精心雕琢出精准、详尽且条理分明的回答。无论是对数据结构中复杂算法的深度剖析，还是对计算机组成原理里晦涩概念的通俗解读，该系统都能应对裕如。这一创新的 408 问题回答系统，不单单是一款便捷的答疑工具，更像是一位如影随形的学习导师，助力学习者透彻理解专业知识，大幅提高学习效率，无疑是计算机专业学习道路上不可或缺的得力伙伴。


## 1. 前置

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ffada5ae8c84403e963d83dd98559298.jpeg#pic_center)


LlamaIndex 是一个上下文增强的 LLM 框架，旨在通过将其与特定上下文数据集集成，增强大型语言模型（LLMs）的能力。它允许您构建应用程序，既利用 LLMs 的优势，又融入您的私有或领域特定信息。
### RAG 效果比对

如图所示， `浦语 API ` 训练数据库中并没有收录到408的相关信息。左图中问答均未给出准确的答案。右图未对 `浦语 API ` 进行任何增训的情况下，通过 RAG 技术实现的新增知识问答。
在这里插入图片描述

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3d4bdd5f141444ae9a8fed8b9d14d0dc.jpeg#pic_center)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d4da806e1ac34959add7a5865de3484e.jpeg#pic_center)

## 2. 环境、模型准备

### 2.1 配置基础环境
这里以在趋势云服务器上部署LlamaIndex为例。


首先。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8162dbbf405a49c49b2b28cae0fac138.jpeg#pic_center)


进入开发机后，创建新的conda环境，命名为 `llamaindex`，在命令行模式下运行：
```bash
conda create -n llamaindex python=3.10
```
复制完成后，在本地查看环境。
```bash
conda env list
```
结果如下所示。
```bash
# conda environments:
#
base                  *  /root/.conda
llamaindex               /root/.conda/envs/llamaindex
```

运行 `conda` 命令，激活 `llamaindex` 然后安装相关基础依赖
**python** 虚拟环境:
```bash
conda activate llamaindex
```
**安装python 依赖包**
```bash
pip install einops==0.7.0 protobuf==5.26.1
```

### 2.2 安装 Llamaindex
安装 Llamaindex和相关的包
```bash
conda activate llamaindex
pip install llama-index==0.11.20
pip install llama-index-llms-replicate==0.3.0
pip install llama-index-llms-openai-like==0.2.0
pip install llama-index-embeddings-huggingface==0.3.1
pip install llama-index-embeddings-instructor==0.2.1
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 下载 Sentence Transformer 模型

源词向量模型 [Sentence Transformer](https://huggingface.co/sentencetransformers/paraphrase-multilingual-MiniLM-L12-v2):（我们也可以选用别的开源词向量模型来进行 Embedding，目前选用这个模型是相对轻量、支持中文且效果较好的，同学们可以自由尝试别的开源词向量模型）

https://modelscope.cn/models/Ceceliachenen/paraphrase-multilingual-MiniLM-L12-v2/summary

```
git lfs install

cd /root/model/

git clone https://www.modelscope.cn/Ceceliachenen/paraphrase-multilingual-MiniLM-L12-v2.git

mv paraphrase-multilingual-MiniLM-L12-v2 sentence-transformer
```


### 2.4 下载 NLTK 相关资源
我们在使用开源词向量模型构建开源词向量的时候，需要用到第三方库 `nltk` 的一些资源。正常情况下，其会自动从互联网上下载，但可能由于网络原因会导致下载中断，此处我们可以从国内仓库镜像地址下载相关资源，保存到服务器上。
我们用以下命令下载 nltk 资源并解压到服务器上：
```bash
cd /root
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
cd nltk_data
mv packages/*  ./
cd tokenizers
unzip punkt.zip
cd ../taggers
unzip averaged_perceptron_tagger.zip
```
之后使用时服务器即会自动使用已有资源，无需再次下载

## 3. 是否使用 LlamaIndex 前后对比

### 3.1 不使用 LlamaIndex RAG（仅API）
```python
test_internlm.py
from openai import OpenAI

base_url = "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
api_key = "eyJ0XBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIwMDE4NjAiLCJyb2wiOiJST0xFX1JFR0lTVEVSIiwiaXNzIjoiT3BlblhMYWIiLCJpYXQiOjE3MzI5NDAzNTksImNsaWVudElkIjoiZWJtcnZvZDZ5bzBubHphZWsxeXiLCJwaGuZSI6IjE2NjkyODYwODQ1IiwidXVpZCI6IjI2YjE3Mzg4LWI1OTktNGQ3OS1iZmQyLWMyNDVmZTE4MjA0NiIsImVtYWlsIjoiemhhb3Fpc2hlbmcyMDIxQG91dGxvb2suY29tIiwiZXhwIjoxNzQ4NDkyMzU5fQ.SFn88Qa2OuVH24Frivyo_rRyFjVZx32yPWwr0VLiBx45iRK7VZdWiTQR4Rt4OSo2DGAURghUxEu7LnViv6TGA"
model="internlm2.5-latest"

# base_url = "https://api.siliconflow.cn/v1"
# api_key = "sk-请填写准确的 token！"
# model="internlm/internlm2_5-7b-chat"

client = OpenAI(
    api_key=api_key , 
    base_url=base_url,
)

chat_rsp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "408是什么？"}],
)

for choice in chat_rsp.choices:
    print(choice.message.content)
```
结果![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/61d7a3c85610492abd597e15d41b4714.jpeg#pic_center)

### 3.2 使用 API+LlamaIndex 

```python
import os 
os.environ['NLTK_DATA'] = '/root/nltk_data'

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.callbacks import CallbackManager
from llama_index.llms.openai_like import OpenAILike


# Create an instance of CallbackManager
callback_manager = CallbackManager()

api_base_url =  "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
model = "internlm2.5-latest"
api_key = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIwMDE4NjAiLCJyb2wiOiJST0xFX1JFR0lTVEVSIiwiaXNzIjoiT3BlblhMYWIipYXQiOjE3MzI5NDAzNTksImNsaWVudElkIjoiZWJtcnZvZDZ5bzBubHphZWsxeXAiLCJwaG9uZSI6IjE2NjkyODYwODQ1IiwidXVpZCI6IjI2YjE3Mzg4LWI1OTktNGQ3OS1iZmQyLWMyNDVmZTE4MjA0NiIsImVtYWlsiemhhb3Fpc2hlbmcyMDIxQG91dGxvb2suY29tIiwiZXhwIjoxNzQ4NDkyMzU5fQ.SFn88Qa72OuVH24Frivyo_rRyFjVZx32yPWwr0VLiBx45iRK7VZdWiTQR4Rt4OSo2DGAURghUxEu7LnViv6TGA"




llm =OpenAILike(model=model, api_base=api_base_url, api_key=api_key, is_chat_model=True,callback_manager=callback_manager)


embed_model = HuggingFaceEmbedding(

    model_name="/root/model/sentence-transformer"
)



Settings.llm = llm

documents = SimpleDirectoryReader("/root/llamaindex_demo/data").load_data()
index = VectorStoreIndex.from_documents(documents)
response = query_engine.query("408是什么?")

print(response)
```
结果![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3e595ab626274af5a64fa1e35199c7da.jpeg#pic_center)


## 4. LlamaIndex web
运行之前首先安装依赖

```shell
pip install streamlit==1.39.0
```

运行以下指令，新建一个python文件

```python
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.legacy.callbacks import CallbackManager
from llama_index.llms.openai_like import OpenAILike

# Create an instance of CallbackManager
callback_manager = CallbackManager()

api_base_url =  "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
model = "internlm2.5-latest"
api_key = "eyJ0eXBlIjoiSldUwiYWxnIjoiSFM1MTIifQ.SFn88Qa72OuVH24Frivyo_rRyFjVZx32yPWwr0VLiBx45iRK7VZdWiTQR4Rt4OSo2DGAURghUxEu7LnViv6TGA"


llm =OpenAILike(model=model, api_base=api_base_url, api_key=api_key, is_chat_model=True,callback_manager=callback_manager)



st.set_page_config(page_title="llama_index_demo", page_icon="🦜🔗")
st.title("llama_index_demo")

# 初始化模型
@st.cache_resource
def init_models():
    embed_model = HuggingFaceEmbedding(
        model_name="/root/model/sentence-transformer"
    )
    Settings.embed_model = embed_model

    #用初始化llm
    Settings.llm = llm

    documents = SimpleDirectoryReader("/root/llamaindex_demo/data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    return query_engine

# 检查是否需要初始化模型
if 'query_engine' not in st.session_state:
    st.session_state['query_engine'] = init_models()

def greet2(question):
    response = st.session_state['query_engine'].query(question)
    return response

      
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "你好，我是你的助手，有什么我可以帮助你的吗？"}]    

    # Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "你好，我是你的助手，有什么我可以帮助你的吗？"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama_index_response(prompt_input):
    return greet2(prompt_input)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Gegenerate_llama_index_response last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama_index_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
```

之后运行
```bash
streamlit run app.py
```

结果![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1cb3d719c87844a0a6a4634378715a73.jpeg#pic_center)
