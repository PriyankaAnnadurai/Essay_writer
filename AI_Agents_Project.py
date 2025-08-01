#!/usr/bin/env python
# coding: utf-8

# # Essay Writer

# In[1]:


# Name: Priyanka A
# Reg.no: 212222230113


# In[2]:


# Load environment variables (e.g., API keys)
from dotenv import load_dotenv

_ = load_dotenv()


# In[3]:


# Import necessary modules for LangGraph, typing, saving state, and message types
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
# Create an in-memory SQLite checkpoint system
memory = SqliteSaver.from_conn_string(":memory:")


# In[4]:


# Define what each state of the essay-writing agent will hold
class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


# In[5]:


# Load the OpenAI Chat model (GPT-3.5-Turbo with deterministic responses)
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# In[6]:


# Prompts used for different stages of the agent workflow
PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. \
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes \
or instructions for the sections."""


# In[7]:


WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.\
Generate the best essay possible for the user's request and the initial outline. \
If the user provides critique, respond with a revised version of your previous attempts. \
Utilize all the information below as needed: 

------

{content}"""


# In[8]:


REFLECTION_PROMPT = """You are a teacher grading an essay submission. \
Generate critique and recommendations for the user's submission. \
Provide detailed recommendations, including requests for length, depth, style, etc."""


# In[9]:


RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can \
be used when writing the following essay. Generate a list of search queries that will gather \
any relevant information. Only generate 3 queries max."""


# In[10]:


RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can \
be used when making any requested revisions (as outlined below). \
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""


# In[11]:


# Define structured schema for LLM output
from langchain_core.pydantic_v1 import BaseModel

class Queries(BaseModel):
    queries: List[str]


# In[12]:


# Initialize TavilyClient using API key
from tavily import TavilyClient
import os
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# In[13]:


# Create plan for essay based on user-provided task
def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = model.invoke(messages)
    return {"plan": response.content}


# In[14]:


# Generate 3 research queries and fetch content for planning
def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


# In[15]:


# Generate initial draft or revised draft based on current content and plan
def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = model.invoke(messages)
    return {
        "draft": response.content, 
        "revision_number": state.get("revision_number", 1) + 1
    }


# In[16]:


# Reflect on draft and provide feedback/critique
def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


# In[17]:


# Generate 3 search queries based on critique and fetch content for improving the essay
def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


# In[18]:


# Decide if the agent should continue refining the essay
def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


# In[19]:


# Initialize graph builder
builder = StateGraph(AgentState)


# In[20]:


# Add functional nodes
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)


# In[21]:


# Set entry node
builder.set_entry_point("planner")


# In[22]:


# Add flow control edges
builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "reflect": "reflect"}
)


# In[23]:


# Compile the graph with in-memory checkpoint
builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")

builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")


# In[24]:


graph = builder.compile(checkpointer=memory)


# In[25]:


# Visualize workflow graph
from IPython.display import Image

Image(graph.get_graph().draw_png())


# In[26]:


# Provide task and stream state outputs step-by-step
thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'task': "what is the difference between langchain and langsmith",
    "max_revisions": 2,
    "revision_number": 1,
}, thread):
    print(s)


# In[ ]:





# ## Essay Writer Interface

# In[27]:


# Import helper utilities and launch Gradio GUI
import warnings
warnings.filterwarnings("ignore")

from helper import ewriter, writer_gui


# In[28]:


MultiAgent = ewriter()
app = writer_gui(MultiAgent.graph)
app.launch()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




