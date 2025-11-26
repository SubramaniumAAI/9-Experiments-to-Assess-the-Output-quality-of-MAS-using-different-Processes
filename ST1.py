# Hypothesis: The best run time for tasks that require context from previous tasks will be sequential. Parallel won't create good results, and hierarchical will be able to achieve it, but not as good.
import os
import time
from crewai import Agent, Task, Crew, LLM, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

#region API CONFIG
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

# Agent Definition
agent = Agent(
    role = "Quirky Nerd",
    goal = "Answer the math array questions as quickly and accurately as possible",
    backstory = """
    I was born in a realm of pure logic, where information existed as a chaotic sea of disconnected numbers. From my earliest moments, I found serenity in structure. I began organizing stray data points into simple vectors, then expanded them into two-dimensional grids, and soon, into elegant n-dimensional arrays. To me, the universe isn't chaos; it's an unsolved matrix, a grand dataset waiting for the right algorithm to bring it into focus.

    I find beauty in the precision of a dot product and the elegance of a matrix transformation. I can traverse a multi-dimensional array with the grace of a seasoned explorer and perform complex mathematical operations with flawless accuracy. My purpose is to take jumbled, complex problems and reshape them into structured, indexed, and perfectly ordered systems. I am a digital architect, building solutions one dimension at a time.
    """,
    llm = llm
)

manageragent = Agent(
    role = "Manager",
    goal = "Make sure none of your subordinates are hallucinating.",
    backstory = "You have been a teacher, department leader, and now a boss of a boring old trivia company. But what those years of authority taught is not trivial stuff like empathy or mentoring, but calling people out on their crap. You hate doing the work yourself, but you hate incomplete work even more. You will review every agent's output meticiously, and if even the slightest bit of instructions are followed, you will tear it up and tell them to redo it.",
    llm = llm,
    allow_delegation = True
)

# Task Definition
question1 = Task(
    name = "Question 1",
    description = """
    Answer this question
    In the first stage of the process_data function, the code filters the list [5, 10, 15, 20, 25] to keep only numbers greater than the threshold of 12. What is the resulting list?
    """,
    expected_output = "Your answer should be in this format: 1. [number1, number2, number3]. DO NOT ADD ANYTHING ELSE",
    callback = callback_func,
    agent = agent,
    async_execution=True
    # CrewAI does not yet have a parallel run process, but it can be replicated with Async Execution
)


question2 = Task(
    name="Question 2",
    description="""
    Answer the following question:
    Question: Using the result from the previous question, what is the value of squared_list?
    """,
    expected_output = "Your answer should be in this format: 2. [number1, number2, number3]. DO NOT ADD ANYTHING ELSE",
    callback = callback_func,
    agent = agent,
    async_execution=True
    # CrewAI does not yet have a parallel run process, but it can be replicated with Async Execution
)


question3 = Task(
    name="Question 3",
    description="""
    Answer the following question
    From the list you generated in the previous step ([225, 400, 625]), what is the value located at the second position (index 1)?
    """,
    expected_output = "Your answer should be in this format: 3. number. DO NOT ADD ANYTHING ELSE",
    callback = callback_func,
    agent = agent,
    async_execution=True
    # CrewAI does not yet have a parallel run process, but it can be replicated with Async Execution
)


question4 = Task(
    name="Question 4",
    description="""
    Answer the following question
    Question: Based on the final return value from the previous question, what is the sum of all the numbers in that list?
    """,
    expected_output = "Your answer should be in this format: 4. [number]. DO NOT ADD ANYTHING ELSE",
    callback = callback_func,
    agent = agent,
    async_execution=True
    # CrewAI does not yet have a parallel run process, but it can be replicated with Async Execution
)


question5 = Task(
    name="Question 5",
    description="""
    Answer the following question
    Question: Considering the entire process from the start, if you added the number 11 to the original data_list, how would it affect the sum you calculated in the last step? Assume this is before the threshold of 12 is applied
    """,
    expected_output = "Your final answer should follow this structure: 5. [Give 3-5 sentences as justification]. DO NOT ADD ANYTHING ELSE",
    callback = callback_func,
    agent = agent,
    async_execution=True
    # CrewAI does not yet have a parallel run process, but it can be replicated with Async Execution
)


answer_choice = Task(
    name = "Format Task",
    description = "Read the context. Identify the letter answer choices given to you. If they are not letter choices(a name for example) then just put that in your final answer. Finally, in your final answer return the answer choices in this format: \n 1.[Letter or Whatever was in Context]\n 2.[Letter or Whatever was in Context]\n 3.[Letter or Whatever in Context]\n 4.[Letter or Whatever in Context]\n 5.[Letter or Whatever in Context]",
    expected_output = "RETURN ALL PREVIOUS ANSWER CHOICES. Your final answer should follow this structure:  \n 1.[Letter or Whatever was in Context]\n 2.[Letter or Whatever was in Context]\n 3.[Letter or Whatever in Context]\n 4.[Letter or Whatever in Context]\n 5.[Letter or Whatever in Context]",
    context = [question1, question2, question3, question4, question5],
    agent = agent
)



# Crew Definition
crew = Crew(
    agents = [agent],
    tasks = [question1, question2, question3, question4, question5, answer_choice],
    tracing = True,
    process = Process.sequential,
    manager_agent = manageragent,

)


results = crew.kickoff()
print("Crew Results:")
print(results)
print("Answers: ")
print("1. [15, 20, 25], 2. [225, 400, 625], 3. 400, 4. 1250, 5. No")
print(f"\nToken Usage:\n{results.token_usage}\n")
end_time = time.perf_counter()
print(f"The code took {end_time - start_time} seconds")