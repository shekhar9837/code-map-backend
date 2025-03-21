from langchain_community.tools import YouTubeSearchTool
from fastapi import FastAPI
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv
import os
from tavily import TavilyClient
from agno.tools.tavily import TavilyTools

load_dotenv()

# Load API keys
tavily_api_key = os.getenv('TAVILY_API_KEY')
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable is not set")
if not youtube_api_key:
    raise ValueError("YOUTUBE_API_KEY environment variable is not set")

# Initialize Tavily client
tavily_client = TavilyClient(api_key=tavily_api_key)

app = FastAPI()

# Tavily-based Search Agent
search_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    tools=[TavilyTools(api_key=tavily_api_key)],
    instructions="You are a search agent. Use the Tavily search tool to find relevant blogs for the given query.",
    show_tool_calls=True,
    markdown=True,
)

# YouTube Search Agent using LangChain
youtube_tool = YouTubeSearchTool()
youtube_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    tools=[youtube_tool],
    instructions="You are a YouTube agent. Use the YouTube search tool to find relevant videos for the given query.",
    show_tool_calls=True,
    markdown=True,
)

# Learning Roadmap Agent
gemini_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    instructions="""Create a structured learning roadmap based on a topic with 7-8 steps. Each step should include:
    - A title
    - Duration (e.g., "4 hours")
    - A brief description
    - Learning resources: [Title](URL)
    - Practice exercises
    - **Validated Resources**:
      - GitHub Repositories: List valid GitHub repositories.
      - Blog Articles: List valid blogs.

    IMPORTANT: Return ONLY a raw JSON object without any markdown formatting, code blocks, or backticks.
    The response must start with { and end with } and be valid JSON.
    """,
    name="Code-Map",
    description="AI learning assistant that creates personalized learning roadmaps with curated resources for any technology or skill.",
    show_tool_calls=True,
    markdown=True,
)

# Team of Agents
agent_team = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    team=[search_agent, youtube_agent, gemini_agent],
    show_tool_calls=True,
    instructions="""Create a structured learning roadmap based on a topic with 7-8 steps. Each step should include:
    - A title
    - Duration (e.g., "4 hours")
    - A brief description
    - Learning resources: [Title](URL)
    - Practice exercises
    - **Validated Resources**:
      - GitHub Repositories: List valid GitHub repositories.
      - Blog Articles: List valid blogs.

    IMPORTANT: Return ONLY a raw JSON object without any markdown formatting, code blocks, or backticks.
    The response must start with { and end with } and be valid JSON.
    """,
    description="You are a team of agents that work together to provide a complete learning experience with YouTube video links, blog links, Tavily web search, and a learning path. Check for broken links and provide valid resources.",
    markdown=True,
    add_datetime_to_instructions=True,
)

@app.get("/roadmap/{topic}")
async def get_roadmap(topic: str):
    response = agent_team.run(topic)
    return response

# Example Usage
# print(youtube_agent.run("tutorial on Next.js data fetching"))

youtube_agent.print_response("tutorial on Next.js data fetching")