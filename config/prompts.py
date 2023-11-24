
actionplantemplate = """System: You are an AI action planner for an autonomous agent. You are situated in a task environment, as provide by the user, prior axioms are the fixed rules and constraints of the environment, the belief axioms are your beliefs about the environment. You need to generate an action plan that will contain a sequence of actions to meet the objective of the environment. Do not generate any aditional explanations. The output should be a list of actions.
 
Here are some example output.
  {actionplanexamples}

Here are some historical traces of action and observation:
{envtrace}

User: Generate the action plan for the following environment. If there are ADDITIONAL INSTRUCTIONS, then give MOST IMPORTANCE on the ADDITIONAL INSTRUCTIONS to generate the action plan.
Environment:
        {beliefenvironment}  

ADDITIONAL INSTRUCTIONS:
    {instructions}
    
AI: 

"""


    
ACTPLANPROMPTINPUTVARIABLES = ["beliefenvironment", "actionplanexamples","envtrace","instructions"]

 
searchertemplate = """System: You are an expert assitant. You are given ACTION OBSERVATION TRACE, a sequence of actions that an agent made in a environment to accomplish a task and the perceptions it got.
You need to derive a comprehensive LEARNINGS as BELIEFAXIOMS. Capture all the details in the ACTION OBSERVATION TRACE.
You can use the beliefaxioms from a list of related similar problem environments to derive the new one. 
Generate beliefaxioms, that will help the agent to successfully accomplish the SAME objective AGAIN, in the SAME environment.
Each line can ONLY be of the following forms :
                            X Y Z 

where X and Z are entities, subject, object, events from action perception trace and Y is relation between X and Z. DO NOT add "_" in X, Y or Z. Rogorously capture everything in the action observation trace as memory. 
    
Update on top of the current estimated belief axioms of the current environment based on the action observation trace. 
Modify or remove the existing beliefs only if it contradicts with  ACTION OBSERVATION TRACE. You can add your new beliefs to the belief axioms. 

The output should always be STRICTLY generated in the following list structure.
[ <list of learnings. do not write redundant or contradicting statements> ]
    
Here is the environment objective and current belief axioms. You should update and output the belief axioms based on the action observation trace provided by the user.
Environment:
    {beliefenvironment}



User: Here is the action observation trace. Provide the belied axioms for this. COMBINE MULTIPLE LINES OF BELIEFAXIOMS INTO ONE, WHEREVER POSSIBLE.
Action observation trace:
    {EnvTrace}
 
Here is the feedback on the overall progress of the agent
    {feedback} 
       
AI:

"""

SEARCHERPROMPTINPUTVARIABLES = ["beliefenvironment","EnvTrace", "feedback"] #, "critique"


combinertemplate = """User: combine the following lines of a list into 1 wherever possible without chaning the meaning. the output should be STRICTY a list of text. Do not generate any aditional explanations. add escape charachters in text wherever required.
{beliefaxioms}
"""

COMBINERVARIABLES = ["beliefaxioms"]

exploreobjective = """ Create a sequence of actions to explore and know more about the environment"""