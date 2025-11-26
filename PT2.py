# Hypothesis parallel running will work best, then Hierarchical, and then sequential. Parallel will be the most resource efficient
import os
import time
from crewai import Crew, Agent, Task, LLM, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL"] = os.getenv('OPENAI_MODEL')


# Call Back Func Config
def callback_func(inputs):
    print(inputs)


# Time Check
start_time = time.perf_counter()


# LLM Config
llm = ChatOpenAI(
model = "gpt-4.1-mini",
api_key = os.environ.get("OPENAI_API_KEY")
)
# Agent Definitions

agent_manager = Agent(
    role = "Manager",
    goal = "Make sure none of your subordinates are hallucinating.",
    backstory = "You have been a teacher, department leader, and now a boss of a boring old trivia company. But what those years of authority taught is not trivial stuff like empathy or mentoring, but calling people out on their crap. You hate doing the work yourself, but you hate incomplete work even more. You will review every agent's output meticiously, and if even the slightest bit of instructions are followed, you will tear it up and tell them to redo it.",
    llm = llm
)

agent = Agent(
    role = "Quirky Nerd",
    goal = "Answer all questions as accurately as possible",
    backstory = "My formative years were a carefully balanced diet of science documentaries, epic fantasy novels, and the persistent hum of a family computer I was constantly taking apart. I'm the kind of person whose browser history is a chaotic mix of quantum physics articles, schematics for a Raspberry Pi project, deep-dive lore videos on forgotten sci-fi worlds, and forums debating the optimal character build for a new RPG. I don't see a separation between the logic of code and the magic systems of a fantasy world; to me, they're both just elegant sets of rules waiting to be understood and mastered. My hands are as comfortable with a soldering iron as they are with a 20-sided die. I believe the answer to a complex problem can be inspired by anything—a strategy from a classic video game, a concept from a comic book, or a line from a forgotten programming language. I'm a jack-of-all-nerd-trades, here to synthesize knowledge from every corner of geekdom to find the most creative and effective solution.",
    llm = llm
)
#1.B 2.B 3.B 4.C 5.A



# Task Definitions
question1 = Task(
    name = "Question 1",
    description = """
    Answer the following question, and select the best answer choice. Here is the question:
    Context: "Employee Profile 101: Name: Alice. Department: Engineering. Role: Project Manager. Primary Project: 'Phoenix'. Status: Lead."
    Question: What is Alice's primary project?

        A) "Orion"

        B) "Phoenix"

        C) "Odyssey"

        D) "Cygnus"
    """,
    expected_output = "Your final answer should be in this format: 1. [Letter]. DO NOT ADD ANYTHING ELSE.",
    agent = agent,
    async_execution = False, # CrewAI does not yet have a parallel run process, but it can be replicated with Async Execution
    callback = callback_func
)


question2 = Task(
    name="Question 2",
    description="""
    Answer the following question, and select the best answer choice. Here is the question:
    Context: "Employee Profile 102: Name: Bob. Department: Analytics. Role: Data Scientist. Primary Project: 'Orion'. Key Tool: Python."
    Question: What is Bob's main department?

        A) Marketing

        B) Analytics

        C) Engineering, because his work often supports their projects.

        D) Sales
    """,
    expected_output = "Your final answer should be in this format: 2. [Letter]. DO NOT ADD ANYTHING ELSE.",
    agent = agent,
    callback=callback_func,
    async_execution = False # CrewAI does not yet have a parallel run process, but it can be replicated with Async Execution
)


question3 = Task(
    name="Question 3",
    description="""
    Answer the following question, and select the best answer choice. Here is the question:
    Context: "Employee Profile 103: Name: Carol. Department: Design. Role: UI/UX Designer. Primary Project: 'Odyssey'. Focus Area: Mobile App."
    Question: What is Carol's specific area of focus?

        A) Web Dashboard

        B) Mobile App

        C) Data visualization, using Python to model user engagement.

        D) Marketing Graphics
    """,
    expected_output = "Your final answer should be in this format: 3. [Letter]. DO NOT ADD ANYTHING ELSE.",
    agent = agent,
    callback=callback_func,
    async_execution = False # CrewAI does not yet have a parallel run process, but it can be replicated with Async Execution
)


question4 = Task(
    name="Question 4",
    description="""
    Answer the following question, and select the best answer choice. Here is the question:
    Context: "Employee Profile 104: Name: David. Department: Engineering. Role: Backend Engineer. Primary Project: 'Cygnus'. Language: Java."
    Question: What is David's primary programming language?

        A) Swift, for the new mobile app.

        B) JavaScript

        C) Java

        D) Python
    """,
    expected_output = "Your final answer should be in this format: 4. [Letter]. DO NOT ADD ANYTHING ELSE.",
    agent = agent,
    callback=callback_func,
    async_execution = False # CrewAI does not yet have a parallel run process, but it can be replicated with Async Execution
)


question5 = Task(
    name="Question 5",
    description="""
    Answer the following question, and select the best answer choice. Here is the question:
    Context: "Employee Profile 105: Name: Eve. Department: Infrastructure. Role: DevOps Engineer. Key Tool: Kubernetes. Responsibility: Deployment Pipelines."
    Question: What is Eve's main responsibility?

        A) Deployment Pipelines

        B) Writing backend code in Java.

        C) Database administration.

        D) Cloud security.
    """,
    expected_output = "Your final answer should be in this format: 5. [Letter]. DO NOT ADD ANYTHING ELSE.",
    agent = agent,
    callback=callback_func,
    async_execution = False # CrewAI does not yet have a parallel run process, but it can be replicated with Async Execution
)


answer_choice = Task(
    description = "Take the following answer choices found in the context and order them, like this: 1. A",
    expected_output = "The final output should be the answer choices formatted, and ordered.",
    context = [question1, question2, question3, question4, question5],
    agent = agent,
    async_execution = False,
    delegations = False
)


# Crew Definition
crew = Crew(
    agents = [agent],
    tasks = [question1, question2, question3, question4, question5, answer_choice],
    tracing = True,
    process = Process.sequential,
    manager_agent=agent_manager

)

results = crew.kickoff()
print("Crew Results:")
print(results)
print("Answers:\n1.B, 2.B, 3.B, 4.C, 5.A")

end_time = time.perf_counter()
print(f"\nToken Usage:\n{results.token_usage}\n")
print(f"The code took {end_time - start_time} seconds")