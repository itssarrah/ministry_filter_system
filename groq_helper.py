import os
from groq import Groq
from typing import List, Dict
import re

# Set Groq API Key
os.environ["GROQ_API_KEY"] = "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"

DEFAULT_MODEL = "gemma2-9b-it"

client = Groq()

def chat_completion(
    messages: List[Dict],
    model=DEFAULT_MODEL,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    """Generates a response using Groq's chatbot."""
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content



def generate_cluster_titles(cluster_summaries: List[str]) -> List[str]:
    """Generates concise business cluster titles directly in French using Groq."""
    
    prompt = (
        "Générez des titres concis et significatifs pour ces clusters d'activités professionnelles en français :\n\n"
    )
    
    for i, summary in enumerate(cluster_summaries):
        prompt += f"Cluster {i+1}: {summary}\n"

    response = chat_completion([{"role": "user", "content": prompt}]).strip()

    # Extract valid cluster titles, ignoring any introductory text
    titles = re.findall(r'\*\*Cluster \d+:\*\*\s*(.+)', response)

    # Ensure the number of titles matches the number of clusters
    if len(titles) != len(cluster_summaries):
        print("Warning: Number of extracted titles does not match clusters. Falling back to generic names.")
        titles = [f"Cluster {i+1}" for i in range(len(cluster_summaries))]

    return titles
