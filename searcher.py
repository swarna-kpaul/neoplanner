from config.keys import *
from config.prompts import *
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
import traceback
from environment.problemenvs import *

import ast
import pickle

llm_model = ChatOpenAI(temperature=0.7, request_timeout=50, model="gpt-3.5-turbo-1106",openai_api_key=OPENAIAPIKEY)
llm_inst_model = OpenAI(temperature=0.7, request_timeout=50, model="gpt-3.5-turbo-instruct",openai_api_key=OPENAIAPIKEY)
llm_defn_model = OpenAI(temperature=0, request_timeout=50, model="gpt-3.5-turbo-instruct",openai_api_key=OPENAIAPIKEY)
llm_gpt4 = ChatOpenAI(temperature=0.7, request_timeout=50, model="gpt-4-0613",openai_api_key=OPENAIAPIKEY)
llm_gpt4_turbo = ChatOpenAI(temperature=0.7, request_timeout=50, model="gpt-4-1106-preview",openai_api_key=OPENAIAPIKEY)
llm_gpt4_turbo_hightemp = ChatOpenAI(temperature=1, request_timeout=50, model="gpt-4-1106-preview",openai_api_key=OPENAIAPIKEY)

class neoplanner():
    def __init__ (self, stmloadfile, stmstoragefile):
    
        self.stmstoragefile = stmstoragefile
        self.env = scienv()
        self.explore = True
        if stmloadfile != None:
            with open(stmloadfile, 'rb') as f:
                model,actiontrace,environment = pickle.load(f)
            self.env.model = model
            self.env.environment = environment
            self.env.actiontrace = actiontrace
            ############# execute action trace
            self.env.traceact()
        
            print (self.env.getstate())
            input("Press a key to continue ....")
            
        self.SEARCHERPROMPT = PromptTemplate(input_variables=SEARCHERPROMPTINPUTVARIABLES, template=searchertemplate)
        self.ACTPLANPROMPT = PromptTemplate(input_variables=ACTPLANPROMPTINPUTVARIABLES, template=actionplantemplate)

    
    def searcher (self,EnvTrace,feedback):
        currentenvironment = self.env.environment
        currentbelief = "  objective:"+ currentenvironment['env']['objective']+"\n  belief axioms:"+ str(currentenvironment['env']["belief axioms"])
        EnvTrace_text = "\n".join([str(i) for i in EnvTrace])
        messages = self.SEARCHERPROMPT.format(
                        beliefenvironment = str(currentbelief),
                        EnvTrace = EnvTrace_text)
            #print(messages)
        print("SEARCHERPROMPT:",messages)
        output = llm_gpt4_turbo.predict(messages)
        print("SEARCHERPROMPT output:",output)
        beliefaxioms = ast.literal_eval(output)["beliefaxioms"]
        self.env.environment["belief axioms"] = beliefaxioms

        return output
    
    def actplan(self, additionalinstructions = ""):
        currentenvironment = self.env.environment
        beliefaxioms = "\n".join(currentenvironment["belief axioms"])
        if not additionalinstructions :
            currentenvironmenttext = "    objective: \n" + currentenvironment["objective"] +"\n\n"+" prior axioms: \n"+currentenvironment["prior axioms"]+"\n\n"+ "     belief axioms:\n"+beliefaxioms+"\n\n"+"    current state:\n"+ currentenvironment["current state"]
        else:
            currentenvironmenttext = "    objective: \n" + currentenvironment["objective"] +"\n\n"+" prior axioms: \n"+currentenvironment["prior axioms"]+"\n\n"+ "     belief axioms:\n"+beliefaxioms+"\n\n"

        messages = self.ACTPLANPROMPT.format(beliefenvironment = currentenvironmenttext, \
                        actionplanexamples = self.env.problemenv.examples,\
                        instructions = additionalinstructions)
        
        print("ACTPLANPROMPT:",messages)
        while True:
            output = llm_model.predict(messages)
            print("ACTPLANPROMPT output:",output)
            try:
                output = ast.literal_eval(output)
            except Exception  as e:
                    #errorfeedback = "Here is the last actionplan generated. "+ output+ "\n But this action plan has the following error. Modify the plan to remove the error.\n"+str(e)
                print(traceback.format_exc())
                input("Press any key to continue...")
                continue    
            
            break
        
        return output
        
    
    def train (self, lifetime = float("Inf")):   
        counter = 0
        while True:
            print("GOAL REACHED",self.env.goalreached)
            if lifetime <= 0: # or self.env.goalreached:
                break
                
            ####### get additional instructions
            instructions,_,_ = self.env.getinstructions()
                
 
            ###### Run actor
            print("Running actionplan....")
            actionplan = []
            
            actionplan = self.actplan(instructions)
            k = input("Press any button to continue ...")
            
            ########### Take actions
            for action in actionplan:
                self.env.act(action)
            
            
            env.updatemodel()
            feedback = env.getfeedback()
            
            EnvTrace = [{"action": trace["action"], "observation": trace["observation"]} for trace in env.trace]
            self.searcher(EnvTrace, feedback)
            input("Press any key to continue...")
            env.trace = []
          
            with open(self.stmstoragefile, 'wb') as f:
                pickle.dump((self.env.model,self.env.actiontrace,self.env.environment),f)

            
   
            
        
    