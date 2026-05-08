import json
from utils import call_llm


def planner(topic):
    prompt = f"""
You are an AI Research Planner.

Generate 3 detailed search queries related to the topic: {topic}.

Each query should focus on different aspects such as:
1. Overview and fundamentals
2. Applications and real-world use cases
3. Benefits, challenges, and future scope

Return the output ONLY in valid JSON format like:
[
    {{"query": "example query"}}
]
"""

    res = call_llm("Planner", prompt)

    try:
        return json.loads(res)

    except:
        return [
            {"query": topic + " overview and fundamentals"},
            {"query": topic + " applications and use cases"},
            {"query": topic + " benefits challenges and future scope"}
        ]


def critic(report):

    prompt = f"""
You are an AI Critic Agent.

Analyze the following report carefully.
Identify:
- factual errors
- unclear explanations
- missing important points
- grammatical mistakes
- weak structure or flow

Provide detailed constructive feedback for improving the report.

Report:
{report}
"""

    return call_llm("Critic", prompt)


def improver(report, critique):

    prompt = f"""
You are an AI Report Improver.

Improve the given report using the critique provided.
Make the report:
- more accurate
- well-structured
- professional
- grammatically correct
- detailed yet clear

Original Report:
{report}

Critique:
{critique}
"""

    return call_llm("Improve", prompt)


def verifier(report, context):

    prompt = f"""
You are a Fact Verification Agent.

Compare the report with the provided context.
Identify hallucinations, incorrect claims, unsupported statements,
or misleading information.

Return:
- Verified points
- Incorrect points
- Final reliability assessment

Report:
{report}

Context:
{context}
"""

    return call_llm("Fact-checker", prompt)


def writer(summary):

    prompt = f"""
You are an AI Technical Report Writer.

Convert the following summary into a professional report.
The report should include:
- introduction
- explanation of key concepts
- applications
- advantages
- conclusion

Write in a clean and readable format.

Summary:
{summary}
"""

    return call_llm("Writer", prompt)


def summarizer(context):

    prompt = f"""
You are an AI Summarization Agent.

Read the following context carefully and generate
a concise yet informative summary.
Include important concepts, key findings,
applications, and major insights.

Context:
{context}
"""

    return call_llm("Summarize", prompt)
