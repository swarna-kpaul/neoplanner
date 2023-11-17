from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import os
import pinecone
from config.keys import *
from datetime import datetime
import string
import random
from langchain.embeddings import OpenAIEmbeddings
#from langchain.chat_models import ChatGooglePalm


embeddings_model = OpenAIEmbeddings(openai_api_key=OPENAIAPIKEY)
llm_model = ChatOpenAI(temperature=0.7, request_timeout=50, model="gpt-3.5-turbo-1106",openai_api_key=OPENAIAPIKEY)

llm_inst_model = OpenAI(temperature=0.7, request_timeout=50, model="gpt-3.5-turbo-instruct",openai_api_key=OPENAIAPIKEY)
llm_defn_model = OpenAI(temperature=0, request_timeout=50, model="gpt-3.5-turbo-instruct",openai_api_key=OPENAIAPIKEY)
llm_0_4_model = ChatOpenAI(temperature=0.4, request_timeout=30, model="gpt-3.5-turbo",openai_api_key=OPENAIAPIKEY, verbose=True)
llm_gpt4 = ChatOpenAI(temperature=0.7, request_timeout=50, model="gpt-4-0613",openai_api_key=OPENAIAPIKEY)
llm_gpt4_turbo = ChatOpenAI(temperature=0.7, request_timeout=50, model="gpt-4-1106-preview",openai_api_key=OPENAIAPIKEY)
llm_gpt4_turbo_hightemp = ChatOpenAI(temperature=0.8, request_timeout=50, model="gpt-4-1106-preview",openai_api_key=OPENAIAPIKEY)



pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

class LTM():  ######### longterm memory
    def __init__ (self,INDEX_NAME = DEFAULT_PINECONE_INDEX_NAME, dbtype = "free"):
        self.index = pinecone.GRPCIndex(INDEX_NAME)
        self.dbtype = dbtype
    
    def set(self, data, namespace = "environments"):
        upsertdata = []
        texts = [i['values'] for i in data]
        embeddings = embeddings_model.embed_documents(texts) ### get embeddings ############
        for i,row in enumerate(data):
            row['values'] = embeddings[i]
            upsertdata.append(row)
        if self.dbtype == "free":
            self.index.upsert(upsertdata)
        else:
            self.index.upsert(upsertdata, namespace = namespace)
    
    def get(self,query,namespace="environments", cutoffscore = 0.4 ,k=3):
        embedding = embeddings_model.embed_query(query)
        if self.dbtype == "free":
            xc = self.index.query(vector = embedding, filter={"type": namespace}, top_k=k, include_metadata=True)
        else:
            xc = self.index.query(vector = embedding, namespace = namespace, top_k=k, include_metadata=True)
        result = xc["matches"]
        data = [ i for i in result if i['score'] > cutoffscore ]
        return data
        
    def fetch(self, ids, namespace="environments"):
        return self.index.fetch(ids=ids)["vectors"]
        
    def delete(self, ids):
        self.index.delete(ids=ids)
        

class STM():  ### Short term memory -- {"conversation": , "timestamp": }
    def __init__ (self,memorysize = DEFAULT_STM_SIZE):
        self.stm = {"ACPtrace": [], "critique":{"feedback":-1,"reason":""}, "currentenv" : {}, "actionplans":{}, "EnvTrace":[], "state": ""}
        self.memorysize = memorysize
        
    def set(self,data,key):
        stmobj = self.stm[key]
        if key == "ACPtrace": 
            if len(stmobj) >= self.memorysize:
                del stmobj[0]
            data["actionplan"]
            stmobj.append({"chat": data , "time" : datetime.now()})
            ## update cumulative rewards
            stmobj = [{"chat": {"actionplan": {"planid": ACP["chat"]["actionplan"]["planid"], "actionplan": ACP["chat"]["actionplan"]["actionplan"], "requiredactions":  ACP["chat"]["actionplan"]["requiredactions"], "cumulative reward": data["actionplan"]["cumulative reward"] if ACP["chat"]["actionplan"]["planid"] == data["actionplan"]["planid"] else ACP["chat"]["actionplan"]["cumulative reward"]} , "perception":ACP["chat"]["perception"], "feedback": ACP["chat"]["feedback"]}, "time": ACP["time"]} for ACP in stmobj ]
        elif key == "EnvTrace":
            if len(stmobj) >= self.memorysize:
                del stmobj[0]
            stmobj.append({"chat": data , "time" : datetime.now()})                
        else:
            stmobj = data
        self.stm[key] = stmobj
    
    def delete(self,key):
        if key in ["ACPtrace","EnvTrace"]:
            self.stm[key] = []
        elif key in ["currentenv","actionplans"]:
            self.stm[key] = {}
        else:
            self.stm[key] = ""
    
    def get(self, key):
        if key == "ACPtrace": 
            chatdata = [ i['chat'] for i in self.stm[key]]
        elif key == "currentenv" and not self.stm[key]:
            id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            return {'id': id, 'env': {}}
        else:
            chatdata = self.stm[key]
        return chatdata
    
stm = STM()    