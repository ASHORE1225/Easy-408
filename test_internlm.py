from openai import OpenAI

base_url = "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
api_key = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiIwMDE4NjAiLCJyb2wiOiJST0xFX1JFR0lTVEVSIiwiaXNzIjoiT3BlblhMYWIiLCJpYXQiOjE3MzI5NDAzNTksImNsaWVudElkIjoiZWJtcnZvZDZ5bzBubHphZWsxeXAiLCJwaG9uZSI6IjE2NjkyODYwODQ1IiwidXVpZCI6IjI2YjE3Mzg4LWI1OTktNGQ3OS1iZmQyLWMyNDVmZTE4MjA0NiIsImVtYWlsIjoiemhhb3Fpc2hlbmcyMDIxQG91dGxvb2suY29tIiwiZXhwIjoxNzQ4NDkyMzU5fQ.SFn88Qa72OuVH24Frivyo_rRyFjVZx32yPWwr0VLiBx45iRK7VZdWiTQR4Rt4OSo2DGAURghUxEu7LnViv6TGA"
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