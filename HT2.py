import os
import time
from crewai import Crew, Agent, Task, Process, LLM
from crewai_tools import ScrapeWebsiteTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
#region API CONFIG
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL"] = os.getenv('OPENAI_MODEL')


# Call Back Func Config
back_logs = []
def callback_func(inputs):
  back_logs.append(inputs)
  print(inputs)


# Time Check
start_time = time.perf_counter()


# LLM Config
llm = ChatOpenAI(
    model="openai/gpt-4o-mini-2024-07-18",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# --- 4. CREATE AGENTS ---

# Agent 1: API Research Specialist
# This agent's only job is to find potential data sources.
api_researcher = Agent(
  role='API Research Specialist',
  goal='Find a free, public API without keys for real-time USD to EUR exchange rates. The output must be ONLY the API URL.',
  backstory=(
    "You are an expert in scouring the web for hidden, free-to-use APIs. "
    "You ignore websites that require scraping and focus exclusively on finding direct API endpoints that return JSON data."
  ),
  verbose=True,
  tools=[ScrapeWebsiteTool()],
  llm = llm
)

# Agent 2: Senior Python Coder
# This agent writes the code based on the researcher's findings.
coder = Agent(
  role='Senior Python Developer',
  goal='Write a Python script to call an API, parse its JSON, and calculate a currency conversion.',
  backstory=(
    "You are a master of Python, specializing in API integration and data processing. "
    "You can write clean, efficient code to interact with any REST API and extract the data needed."
  ),
  verbose=True,
  llm = llm
)

# Agent 3: QA Engineer
# This agent tests the code and provides critical feedback.
qa_engineer = Agent(
  role='Software Quality Assurance Engineer',
  goal='Rigorously test the Python script. If it fails, provide a clear error report. If it works, confirm it.',
  backstory=(
    "You have an obsessive eye for detail. Your mission is to find flaws in code. "
    "You execute scripts and analyze their outputs, providing clear, actionable feedback to the development team."
  ),
  verbose=True,
  llm = llm
)

# Agent 4: The Project Manager (The heart of the hierarchical process)
# This agent orchestrates the entire workflow.
project_manager = Agent(
    role="Project Manager",
    goal=(
        "Oversee the entire process of creating a currency conversion script. "
        "Delegate tasks, analyze results, and if an error occurs, decide the next steps to ensure the project's success."
    ),
    backstory=(
        "You are a seasoned Project Manager who knows how to lead a team to success. "
        "You don't write code or do research yourself, but you are an expert at understanding technical problems and delegating tasks to the right specialist."
    ),
    allow_delegation=True, # The manager must be able to delegate
    verbose=True,
    llm = llm
)

# --- 5. DEFINE THE TASKS ---

# The manager starts with a high-level goal.
# The manager will break this down and delegate to the team.
create_code_task = Task(
    description="""
    Develop a Python script to converts 100 USD to EUR. Here are the constraints of the program:
    1. The code must always be current. It must update with new conversion rates
    2. You must use a public API without an API key
    3. You must store your final answer in a variable name result, and print it on the console.
    4. Store the intial amount of USD to be converted in a var called 'intial_usd'
    """,
    expected_output = "A working Python Script.",
    agent = coder,
    async_execution=False,
)


review_code_task = Task(
    description = """
    Test the Python script, beautify the program, and check for errors. Here are more specifics:
    1. Test the code to see if you output works
    2. Beautify the UI and messages. If you see any way to improve the code make those improvements
    3. Check for errors. Does the code break if you replace intial_usd with dollars? Or does the code break if you change the API?
    """,
    expected_output = "The fixed Python script",
    agent = qa_engineer,
    async_execution=False,

)


formatting_task = Task(
    description = """
    Here are the requirements of this response:
    1. Paste your code as the first part of your final answer. Do not format in JSON, format it as a string.
    2. Find the price of Gold in USD. Show the source you got that from. DO NOT CREATE CODE TO DO THIS
    3. Find the price of Gold in Euros. Give a step by step instructions on how to use your code to get that number. YOU MUST STATE THE PRICE

    To reiterate, your final answer should have three things: Your code formatted as a string, the Gold price in USD and the source(a link to a website) you got it from, and the Gold price in Euros as well as how to get that number.
    """,
    expected_output = "Formatted Code, Websites, Steps",
    agent = coder
)

crew = Crew(
    agents = [api_researcher, coder, qa_engineer],
    tasks = [create_code_task, review_code_task, formatting_task],
    tracing = True,
    process = Process.sequential,
    manager_agent = project_manager,
)

results = crew.kickoff()
print("Crew Code: ")
print(results)
print()
end_time = time.perf_counter()
print(f"\nToken Usage:\n{results.token_usage}\n")
print(f"The code took {end_time - start_time} seconds")