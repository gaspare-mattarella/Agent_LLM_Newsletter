"""Agents_AI.ipynb
# **AGENT AI - Research Assistant on weekly LLM News And Advancements**
"""

from crewai_tools import SerperDevTool
from crewai import Task, Agent, Crew
from langchain_openai import ChatOpenAI
from langchain.agents import load_tools
import mailtrap as mt
from dotenv import load_dotenv
import os 
import markdown as md
from langchain.tools import Tool
from internal_tools import BrowserTool
from crewai_tools import ScrapeWebsiteTool

####################

# it is needed to load the following api of this format to run this code:
# OPENAI_API_KEY = <YOUR OPENAI KEY>
# SERPER_API_KEY = <YOUR SERPER KEY>
# maildrop = <YOUR MAILDROP KEY>

######################

load_dotenv()

human_tool = load_tools(['human'])

# To enable scrapping any website it finds during it's execution
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
gpt3 = 'gpt-3.5-turbo' 
gpt4 = 'gpt-4-turbo-preview'

reddit_scraper = Tool(
    name='BrowserTool',
    func=BrowserTool.scrape_reddit,
    description="Tool useful to scraping reddit. The only parameter of the function is max_comments_per_post and it is already set by default."
)

"""
- define agents that are going to research latest AI tools and write a blog about it 
- explorer will use access to internet and LocalLLama subreddit to get all the latest news
- writer will write drafts 
- critique will provide feedback and make sure that the blog text is engaging and easy to understand
"""


researcher = Agent(
    role = 'Researcher Assistant',
    goal = 'look up on the internet for the lastest breaktrough, news and advancements in LLMs during the last month.',
    backstory = """You work as assistant to a Data Scientist who wants to stay up to date on the topic of LLMs.
    Your expertise lies in searching Google for LLM Papers and News no older than 1 month.""",
    verbose = False,
    allow_delegation = False,
    tools = [search_tool] + [scrape_tool],
    llm = ChatOpenAI(model_name = gpt3, temperature = 0.9)
)


scraper = Agent(
    role = 'Reddit Scraper and Reporter',
    goal = 'look up on the LocalLLama subreddit for the lastest breaktrough, news and advancements in LLMs.',
    backstory = """You work as assistant to a Data Scientist who wants to stay up to date on the topic of LLMs.
    Your expertise lies in scraping this specific subreddit for LLM News and interesting discoveries.""",
    verbose = False,
    allow_delegation = False,
    tools = [reddit_scraper] + human_tool,
    llm = ChatOpenAI(model_name = gpt4, temperature = 0.8)
)

editor = Agent(
    role = "Chief Editor",
    goal = """ The goal is to provide to a Data Scientist a weekly newsletter on lastest developments in the field of LLMs.
    Therefore your goal is to take all the summaries provided by your coworker Reasearch Assistant and keep it in in memory, make sure the source (name the link) is also mention. 
    Than you need to use all the info to draft a nice good looking newsletter.
    Make sure that your coworker goes through at least 3 different topics s by asking to go to the next one when you've collected all the necessary info about the first ones.
    """,
    backstory = "You are a content strategist. You take all the summaries from the researcher and make it prettier. You are responsible for the final look of the newsletter.",
    verbose = True,
    tool = human_tool,
    allow_delegation = True,
    llm = ChatOpenAI(model_name = gpt4, temperature = 0.9)
)

task_google = Task(
    description = """
    You need to find the latest news or curious and interesting facts about LLMs on google. They need to be no older than 1 month.
    Find the fisrt 10 results on google, select the most interesting and read it. You need to go inside the webpage and read the content, summarize it and deliver an executive summary to your co worker editor. 
    """,
    expected_output= "A comprhensive and detailed summary for a chosen topic on the latestes advancemnts on LLMs. Deliver it with link to the source",
    agent = researcher
)

task_reddit = Task(
    description="""
    Use scraped data from subreddit LocalLLama to select and report on the most interesting topic discussed down there. Make sure the input is not too long (NO MORE THAN 16385 tokens)
    Use ONLY scraped data from LocalLLama to generate the report. Your final answer MUST be a full analysis report, text only, ignore any code or anything that 
    isn't text. The report has to have bullet points.
    """,
    expected_output= "A full comprhensive summary in detailed and long bullet points for the selected most interesting topic found. Include link to the source.",

    agent = scraper
)

task_drafting = Task(
    description = """
    take as imput EVERYTHING provided by your co worker Reasearch Assistant and keep it in in memory. DO NOT FORGET ABOUT THE FIRST THINGS RECEIVED. Make sure the source (name and link) is also mentioned. 
    Make sure that your coworker goes through at least 3 different topics (They all must be on LLMs of course) by requesting to go to the next one once you've collected and stored in memory the first summary as topic number 1. 
    It is of utmost importance that you do not forget previous topics.
    Finally you need to format everything as a cute informative weekly newsletter for Data Scientists.
    """,
    expected_output= "A long newsletter with at least  couple of paragraphs for each topic received (no more than 3 in total). Make sure the sources for each topic (link, urls) are mentioned, it is very important.",
    agent = editor
)



crew = Crew(
    agents = [researcher, editor],
    tasks = [task_google, task_drafting],
    verbose = 2,
    manager_llm=editor
)

results_google = crew.kickoff()

# create mail object
mail = mt.Mail(
    sender=mt.Address(email="mailtrap@demomailtrap.com", name="LLM Newsletter Agent"),
    to=[mt.Address(email="ai.research.agent@gmail.com")],
    subject="Weekly Update on LLMs!",
    html = md.markdown(results_google)
)

# create client and send
client = mt.MailtrapClient(token = os.environ.get('maildrop'))
client.send(mail)

