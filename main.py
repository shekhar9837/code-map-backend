from typing import Union
from fastapi import FastAPI
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.youtube import YouTubeTools
from dotenv import load_dotenv
import os

load_dotenv()

google_api_key = os.getenv('GOOGLE_API_KEY')
app = FastAPI()

gemini_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    instructions="""Create a structured learning roadmap  based on topic with 7-8 steps . Each step should include:
    - A title
    - Duration (e.g., "4 hours")
    - A brief description
    - Learning resources: [Title](URL)
    - Practice exercises
    - **Validated Resources**:
      - GitHub Repositories: ${resources.github.map((url) => `- [Explore Here](url).join("\n")}
      - Blog Articles: ${resources.blogs.map((url:any) => `- [Read Here](url)`).join("\n")}
    
    IMPORTANT: Return ONLY a raw JSON object without any markdown formatting, code blocks, or backticks.
    The response must start with { and end with } and be valid JSON.
    
    Example format:
    {
      "steps": [
        {
          "id": "1",
          "title": "Step title",
          "duration": "X hours",
          "description": "What will be learned",
          "resources": ["Resource: [Title](URL) - Description"],
          "practice": ["Practice exercise: Description"]
        }
      ]
    }""",
    name="Code-Map",
    description="AI learning assistant that creates personalized learning roadmaps with curated resources for any technology or skill.",
    show_tool_calls=True,
    markdown=True,
)

yt_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    tools=[YouTubeTools(get_video_data=True)],
    show_tool_calls=True,
    description="You are a YouTube search agent. Always call the YouTube search tool to return video links for the given query. If you cannot find results, say 'No results found' but DO NOT generate text responses.",
)


duckduckgo_agent = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    description="You are a DuckDuckGo agent. that gives blog link from dev, medium, hashnode based on user input ",
)

agent_team = Agent(
    model=Gemini(id="gemini-1.5-flash"),
    tools=[gemini_agent, duckduckgo_agent, yt_agent],  # Include yt_agent
    show_tool_calls=True,
    description="You are a team of agents that work together to provide a complete learning experience with YouTube video links, blog links, and a learning path.",
    markdown=True,
)


# agent.print_response("tutorial on nextjs")
yt_agent.print_response("beginer next js tutorials on routing", stream=True)

