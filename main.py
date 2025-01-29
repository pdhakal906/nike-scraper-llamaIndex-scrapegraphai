import os
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.together import TogetherLLM
from dotenv import load_dotenv


load_dotenv()


def scrapegraph_tool_invocation(prompt, url):
    from llama_index.tools.scrapegraph.base import ScrapegraphToolSpec

    scrapegraph_tool = ScrapegraphToolSpec()
    response = scrapegraph_tool.scrapegraph_smartscraper(
        prompt=prompt,
        url=url,
        api_key=os.getenv("SGAI_API_KEY"),
    )
    return response


# Fetch API keys
together_api_key = os.getenv("TOGETHER_API_KEY")

if not together_api_key:
    raise EnvironmentError(
        "Together API key not found. Set the TOGETHER_API_KEY environment variable."
    )

scrapegraph_api_key = os.getenv("SGAI_API_KEY")
if not scrapegraph_api_key:
    raise EnvironmentError(
        "ScrapeGraph API key not found. Set the SGAI_API_KEY environment variable."
    )

# Initialize tools and agent
scrape_tool = FunctionTool.from_defaults(fn=scrapegraph_tool_invocation)
# llm = OpenAI(model="gpt-4", api_key=openai_api_key)
llm = TogetherLLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key=together_api_key
)
agent = ReActAgent.from_tools([scrape_tool], llm=llm, verbose=True)

# Extract product data from a website
link = "https://www.nike.com/in/w/mens-shoes-nik1zy7ok"
res = agent.chat(
    f"Extract me 50 products with their name and price from the following website: {link}, you will have to scroll the website to "
    "get all the products. MAKE SURE TO GET 50 PRODUCTS."
)
print(res)

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(
        str(res),
    )
