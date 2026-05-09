"""Prompt templates: basic, structured, few-shot, chain-of-thought.

Each function returns the *prompt string*. Pass it to `llm_client.generate_text`
to get a completion.
"""

from __future__ import annotations


BASIC_HOTEL = "Tell me about Paris hotels"

STRUCTURED_HOTEL = """\
Provide information about hotels in Paris, including:
- Location considerations
- Price ranges
- Popular areas for tourists
- Transportation access
Please format the response in clear sections.
Give a complete answer; do not cut off halfway.
"""


def few_shot_hotel(prompt: str) -> str:
    return f"""\
Example 1:
User: Tell me about New York hotels
AI: The Plaza Hotel in New York is an iconic luxury hotel located at Fifth Avenue and Central Park South. It offers timeless elegance, world-class dining, and top-tier hospitality.

Example 2:
User: Tell me about Tokyo hotels
AI: The Park Hyatt Tokyo is a prestigious hotel known for its stunning skyline views, sophisticated atmosphere, and exceptional dining options. Located in Shinjuku, it provides a tranquil retreat in the heart of the city.

Now, following the same pattern, provide a response:
User: {prompt}
AI:"""


def chain_of_thought_hotel(prompt: str) -> str:
    return f"""\
Let's think step by step.

Example 1:
User: Tell me about New York hotels
AI: First, I will select a famous hotel in New York. The Plaza Hotel is a well-known luxury hotel. Then, I will highlight its key attributes — historic significance, prime location near Central Park, and world-class dining. Finally, I will summarize why it stands out.
Answer: The Plaza Hotel in New York is an iconic luxury hotel located at Fifth Avenue and Central Park South. It offers timeless elegance, world-class dining, and top-tier hospitality.

Example 2:
User: Tell me about Tokyo hotels
AI: First, I will choose a renowned hotel in Tokyo — The Park Hyatt Tokyo. Then, I will describe its notable aspects: stunning skyline views, sophisticated atmosphere, location in Shinjuku. Lastly, I will explain why visitors prefer it.
Answer: The Park Hyatt Tokyo is a prestigious hotel known for its stunning skyline views, sophisticated atmosphere, and exceptional dining options. Located in Shinjuku, it provides a tranquil retreat in the heart of the city.

Now apply the same reasoning:
User: {prompt}
AI:"""


def analyze_hotel_search_results(query: str, results: str) -> str:
    """CoT prompt for analyzing retrieval output — the bridge into Chapter 6."""
    return f"""\
Analyze the following hotel search results and answer these questions using a chain of thought:

1. Which hotel best matches the user's query: '{query}'?  Consider location, food options, and overall experience.
2. What are the pros and cons of the top 3 hotels?  Support your answer with quotes from the reviews.
3. Are there any recurring themes or patterns in the positive/negative reviews?
4. Summarize the top 3 hotels in bullet points highlighting their strengths and weaknesses.
5. Based on the provided data, give an overall recommendation to the user.

Hotel Search Results:
{results}
"""
