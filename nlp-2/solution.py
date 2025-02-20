import os
import re
import logging
import wolframalpha
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import wikipedia
from random import sample, randint
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import tqdm
import json
import numpy as np
import faiss

# Set API keys
os.environ["OPENAI_API_KEY"] = "..."
os.environ["WOLFRAM_ALPHA_APPID"] = "..."

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize WolframAlpha client and Wikipedia API
client = OpenAI()
llm = ChatOpenAI(temperature=0.4, model="gpt-4")

wolfram_client = wolframalpha.Client(os.environ["WOLFRAM_ALPHA_APPID"])

wikipedia.set_lang('uk')  # Ukrainian Wikipedia


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    """
    Generate an embedding for the given text using OpenAI's embedding model.
    """
    logging.info(f"Generating embedding for text: {text[:50]}...")  # Log first 50 characters
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding


def create_faiss_index(dataset_path: str, embedding_model: str = "text-embedding-ada-002") -> tuple:
    """Create a FAISS index from the dataset."""
    logging.info("Creating FAISS index...")
    embeddings = []
    texts = []

    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as file:
        for line in tqdm.tqdm(file):
            entry = json.loads(line)
            text = entry["input"] + " " + entry["output"]
            texts.append(text)  # Use the task descriptions
            embeddings.append(get_embedding(text, model=embedding_model))

    # Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, texts


def generate_task(topic: str, task_type: str = "задача", index=None, texts=None,
                  embedding_model: str = "text-embedding-3-small") -> str:
    """Generate a math task in Ukrainian using RAG with FAISS."""
    logging.info(f"Generating task for topic: {topic}, type: {task_type}")

    # Retrieve relevant context
    query_embedding = get_embedding(topic, model=embedding_model)
    distances, indices = index.search(np.array([query_embedding], dtype='float32'), k=3)  # Top 3 similar tasks
    context = "\n".join([texts[i] for i in indices[0]])
    logging.info(f"Retrieved Context:\n{context}")

    # Use the retrieved context in the LLM prompt
    llm = ChatOpenAI(temperature=0.4, model="gpt-4")
    prompt = (
        f"Згенеруй завдання з математики на тему '{topic}' типу '{task_type}' українською мовою. "
        "Враховуючи наступний контекст (якщо це доцільно), напиши завдання чітко і коротко:\n\n"
        f"Контекст:\n{context}\n\n"
        "Завдання:\n"
        "..."
    )
    chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["topic", "task_type", "context"], template=prompt))
    return chain.run({"topic": topic, "task_type": task_type, "context": context}).strip()


def format_task_for_wolfram(task: str) -> str:
    """Use LLM to rewrite task in a format compatible with WolframAlpha."""
    logging.info("Reformatting task for WolframAlpha using LLM.")
    chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["task"], template=(
        "Rewrite the following math problem to be compatible with WolframAlpha syntax. "
        "Ensure proper syntax and remove any ambiguities or natural language expressions. "
        "Output only the properly formatted problem:\n\n"
        "Problem:\n{task}\n\n"
        "Formatted Problem:"
    )))
    return chain.run({"task": task}).strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def solve_task(task: str) -> str:
    """Solve a math task using WolframAlpha."""
    logging.info("Attempting to solve the task with WolframAlpha.")
    try:
        result = wolfram_client.query(task)
        pods = list(result.pods)
        if not pods:
            raise ValueError("No solution pods returned by WolframAlpha.")
        return '\n'.join([f"{pod['@title']} - {pod.text}" for pod in pods]) if len(pods) > 0 else "Рішення недоступне."
    except Exception as e:
        logging.warning(f"Attempt to solve the task failed: {e}")
        raise


def generate_explanation(task: str, solution: str) -> str:
    """Generate an explanation for the task with solution."""
    logging.info("Generating explanation with Wikipedia references.")
    chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["task", "solution"],
        template=(
            "На основі наступного завдання та розв'язку:\n\n"
            "Завдання:\n{task}\n"
            "Рішення:\n{solution}\n\n"
            "Згенеруй пояснення українською мовою, включаючи посилання з Вікіпедії. "
            "Структура відповіді:\n"
            "1. Опис завдання\n"
            "2. Пояснення рішення\n"
            "Відповідь:"
        )
    ))
    return chain.run({"task": task, "solution": solution}).strip()


def fetch_wikipedia_references(task: str, explanation: str) -> str:
    """
    Use LLM to extract topics from the task and explanation and fetch references
    from Ukrainian Wikipedia.
    """
    logging.info("Extracting topics and fetching Wikipedia references.")
    chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["task", "explanation"],
        template=(
            "Extract key topics from the following math problem and its explanation. "
            "These topics will be used to find related references in Ukrainian Wikipedia.\n\n"
            "Task: {task}\n"
            "Explanation: {explanation}\n\n"
            "Output the topics as a comma-separated list:"
        )
    ))
    response = chain.run({"task": task, "explanation": explanation}).strip()

    topics = [topic.strip() for topic in response.split(",")]

    # Fetch references from Wikipedia
    references = []
    for topic in topics:
        try:
            logging.info(f"Fetching Wikipedia reference for topic: {topic}")
            page = wikipedia.page(topic)
            references.append(f"{page.title} - {page.url}")
        except Exception as e:
            logging.warning(f"Failed to fetch Wikipedia reference for topic '{topic}': {e}")
            references.append(f"{topic} - Вікіпедіа статтей не знайдено.")

    return "\n".join(references)


def generate_answer_options(solution: str, task: str) -> tuple:
    """
    Use LLM to parse the correct answer from WolframAlpha's solution
    and generate distractors.
    """
    logging.info("Parsing correct answer and generating distractors using LLM.")
    chain = LLMChain(llm=llm, prompt=PromptTemplate(
        input_variables=["task", "solution"],
        template=(
            "You are tasked with generating multiple-choice options for a math problem. "
            "Given the solution and the original task, extract the correct answer and generate three plausible distractors. "
            "Ensure the distractors are reasonable and not overly similar to the correct answer.\n\n"
            f"Task: {task}\n"
            f"Solution: {solution}\n\n"
            "Output the result in the following format:\n"
            "Correct Answer: <correct_answer>\n"
            "Distractors: <distractor_1> ### <distractor_2> ### <distractor_3>\n"
        )
    ))
    response = chain.run({"task": task, "solution": solution}).strip()

    # Parse the output
    correct_answer_match = re.search(r"Correct Answer:\s*(.*)", response)
    distractors_match = re.search(r"Distractors:\s*(.*)", response)

    if not correct_answer_match or not distractors_match:
        raise ValueError("Failed to parse correct answer or distractors from LLM response.")

    correct_answer = correct_answer_match.group(1).strip()
    distractors = [d.strip() for d in distractors_match.group(1).split("###")]

    # Combine and shuffle options
    options = [correct_answer] + distractors
    shuffled_options = sample(options, len(options))  # Shuffle the options

    return correct_answer, shuffled_options


def main():
    topic = "логарифмічні нерівності"
    task_type = "задача"

    try:
        logging.info("Starting workflow...")

        index, texts = create_faiss_index('dataset.jsonl')
        logging.info(f"FAISS index created.")

        # Step 1: Generate a task
        task = generate_task(topic, task_type, index, texts)
        logging.info(f"Generated Task:\n{task}")

        # Step 2: Format the task for WolframAlpha
        formatted_task = format_task_for_wolfram(task)
        logging.info(f"Formatted Task for WolframAlpha:\n{formatted_task}")

        # Step 3: Solve the task
        solution = solve_task(formatted_task)
        logging.info(f"Task Solution:\n{solution}")

        # Step 4: Generate explanation
        explanation = generate_explanation(task, solution)
        logging.info(f"Generated Explanation:\n{explanation}")

        # Step 5: Fetch references
        references = fetch_wikipedia_references(task, explanation)
        logging.info(f"Wikipedia References:\n{references}")

        # Step 6: Generate multiple-choice options
        correct_option, options = generate_answer_options(task, solution)
        logging.info(f"Answer Options: {options}")

        # Step 7: Print the final output
        print(f"\n=== Завдання ===\n{task}")
        print(f"\n=== Відповіді ===\n1) {options[0]}\n2) {options[1]}\n3) {options[2]}\n4) {options[3]}")
        print(f"\n=== Правильна відповідь ===\n{correct_option}")
        print(f"\n=== Пояснення ===\n{explanation}")
        print(f"\n=== Посилання ===\n{references}")

    except Exception as e:
        logging.error(f"Workflow failed: {e}")


if __name__ == "__main__":
    main()
