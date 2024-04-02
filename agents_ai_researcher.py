"""Agents_AI.ipynb
# **AGENT AI - Research Assistant on weekly LLM News And Advancements**
"""


from crewai_tools import SerperDevTool
from crewai import Task, Agent, Crew
from langchain_openai import ChatOpenAI
import mailtrap as mt
from dotenv import load_dotenv

load_dotenv("api.env")

search_tool = SerperDevTool()


researcher = Agent(
    role = 'Reasearcher Assistant',
    goal = 'look up on the internet for the lastest breaktrough, news and advancements in LLMs during the last month.',
    backstory = """You work as assistant to a Data Scientist who wants to stay up to date on the topic of LLMs.
    Your expertise lies in searching Google for LLM Papers and News no older than 1 month.""",
    verbose = True,
    allow_delegation = False,
    tools = [search_tool],
    llm = ChatOpenAI(model_name = "gpt-4-turbo-preview", temperature = 0.6)
)

writer = Agent(
    role = "Professional Writer",
    goal = """
    take as imput all the summarized bullet points provided by your co worker Senior Reasearch Assistant and rewrite everything in a more discorsive and pleasant way. 
    If you do not have enough information about a specific topic to draft a meaningful paragraph, ask your coworker to do a deeper research on the topic.
    """,
    backstory = "You are a content strategist who have to evaluate if the information provided by the research are enough to draft an executive parapgraph or not. If they are, you draft it, otherwise go back to the reseracher and ask for more info and then draft it.",
    verbose = True,
    allow_delegation = True,
    llm = ChatOpenAI(model_name = "gpt-4-turbo-preview", temperature = 0.5)

)

task1 = Task(
    description = """
    You need to find and read all the the latest news, reportage and blog posts about LLMs in the internet. They need to be no older than 1 month.
    You need to avoid that there are repetions in the news and eventually summarize them toghther. 
    You need to assess if some topic is more important than the others and rank them and prioritize them.
    Finally you need to report everything.
    """,
    expected_output= "A full comprhensive report in detailed and long bullet points for each topic found",
    agent = researcher
)

task2 = Task(
    description = """
    You need to go through the received input from your co worker Senior Reasearch Assistant and rewrite everything in a more discorsive and pleasant way. 
    If you do not have enough information about a specific topic to draft a meaningful long paragraph, ask your coworker to do a deeper research on that specific topic.
    Then write a meaningful paragrapgh also about it.
    """,
    expected_output= "A full comprhensive blog post of at least a long paragraph for each bullet point received. Add also the sources for each topic.",
    agent = writer
)

crew = Crew(
    agents = [researcher, writer],
    tasks = [task1, task2],
    verbose = 4
)

results = crew.kickoff()

# create mail object
mail = mt.Mail(
    sender=mt.Address(email="mailtrap@demomailtrap.com", name="LLM Newsletter Agent"),
    to=[mt.Address(email="ai.research.agent@gmail.com")],
    subject="Weekly Update on LLMs!",
    text = results,
)

# create client and send
client = mt.MailtrapClient(token="d0662c340108b2242fb4eed6ec09cd03")
client.send(mail)
