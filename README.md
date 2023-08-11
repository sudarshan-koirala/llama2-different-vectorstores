# Llama2-different-vectorstores
Simple Chainlit app to have interaction with your documents using different vectorstores.

### Chat with your documents 🚀
- [LLama2 from Huggingface Website](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin) as Large Language model
- [LangChain](https://python.langchain.com/docs/get_started/introduction.html) as a Framework for LLM
- [Chainlit](https://docs.chainlit.io/overview) for deploying.
- [GGML](https://github.com/ggerganov/ggml) to run in commodity hardware (cpu)
- [CTransformers](https://github.com/marella/ctransformers) to load the model.
- [Embedding model from Huggingface Website](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## System Requirements

You must have Python 3.11 or later installed. Earlier versions of python may not compile.    

---

## Steps to Replicate 

1. Fork this repository and create a codespace in GitHub as I showed you in the youtube video OR Clone it locally.
   ```
   git clone https://github.com/sudarshan-koirala/llama2-different-vectorstores.git
   cd llama2-different-vectorstores
   ```

2. Rename example.env to .env with `cp example.env .env`and input the HuggingfaceHub API token as follows. Get HuggingfaceHub API key from this [URL](https://huggingface.co/settings/tokens). You need to create an account in Huggingface webiste if you haven't already.
   ```
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
   ```

   Get the Pinecone API key and Env variable name from this [URL](https://platform.openai.com/account/api-keys). You need to create an account if you haven't already.
   ```
   PINECONE_ENV=*****
   PINECONE_API_KEY=*****
   ```

   OPTIONAL: If you want to test with openai models. Get the OpenAI API key from this [URL](https://platform.openai.com/account/api-keys). You need to create an account if you haven't already.
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```
   
3. Create a virtualenv and activate it
   ```
   python3 -m venv .venv && source .venv/bin/activate
   ```

   If you have python 3.11, then the above command is fine. But, if you have python version less than 3.11. Using conda is easier. First make sure that you have conda installed. Then run the following command.
   ```
   conda create -n .venv python=3.11 -y && source activate .venv
   ```

4. Run the following command in the terminal to install necessary python packages:
   ```
   pip install -r requirements.txt
   ```

5. Create a model folder in the root directory and download the model inside the folder.
   ```
   mkdir model && cd model
   wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin
   ```


5. Go to `ingest` folder and run the following command in your terminal to create the embeddings and store it locally:
   ```
   python3 run ingest_chroma.py
   ```

6. Go inside `app` folder and run the following command in your terminal to run the app UI:
   ```
   chainlit run app_chroma.py --no-cache -w
   ```

**Repeat step 5 and 6 for different vectorstores.** Happy learning 😎

---
## Disclaimer
This is test project and is presented in my youtube video to learn new stuffs using the openly available resources (models, libraries, framework,etc). It is not meant to be used in production as it's not production ready. You can modify the code and use for your usecases ✌️
