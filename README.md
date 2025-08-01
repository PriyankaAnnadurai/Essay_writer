# AI-Powered Essay Writer using LangGraph and LangChain

## Abstract:
This project implements an AI-powered multi-step essay writing assistant using LangGraph, LangChain, and OpenAI's GPT-3.5/4 models. The assistant automates the process of essay writing through distinct stages: planning, research, generation, critique, and revision. It dynamically decides when to revise content based on feedback and external research using the Tavily Search API. Each step is managed through a modular state-machine-like workflow that ensures high-quality and refined output essays. The final solution is also integrated with a Gradio-based GUI for an interactive user experience.

## Working:
- The user inputs an essay topic or question (e.g., “What is the difference between LangChain and LangSmith?”).

- The system generates an essay outline (structure with notes) using a planning prompt.

- Using the outline and topic, it creates search queries.

- It fetches relevant content from the internet via the Tavily Search API.

- The AI generates a 5-paragraph essay using the outline and the gathered research data.

-The AI evaluates the essay like a teacher and provides detailed feedback on:

- Based on the critique, the system generates new search queries to collect more content for improvement.

- A revised draft is created based on feedback and new content. This loop continues until the maximum number of revisions is reached.

- The process is controlled via LangGraph, which handles: State transitions, Conditional logic, Checkpointing with SQLite (in-memory)

- The essay-writing workflow is visualized as a graph showing node transitions (Planner → Research → Generate → Reflect → Revise).

- A Gradio-based web UI is provided to allow interactive usage. Users can run the essay writer directly through the interface.

## Output:

<img width="1893" height="869" alt="image" src="https://github.com/user-attachments/assets/29a37f35-c311-4d77-b049-fad2054f7f12" />
