import gradio as gr
from functools import partial
import logging

from solution import *

index, texts = create_faiss_index('dataset.jsonl')

def process_topic(topic):
    try:
        # Generate task
        task = generate_task(topic, "задача", index, texts)
        formatted_task = format_task_for_wolfram(task)
        solution = solve_task(formatted_task)
        explanation = generate_explanation(task, solution)
        references = fetch_wikipedia_references(task, explanation)
        correct_option, options = generate_answer_options(solution, task)

        return task, options, correct_option, explanation, references
    except Exception as e:
        logging.error(f"Error in processing topic: {e}")
        return "Error in generating task", [], "", "", ""

def validate_answer(user_answer, correct_option):
    if user_answer == correct_option:
        return "Правильно!"
    else:
        return "Неправильно, спробуй знову!"

def reveal_details(explanation, references):
    return explanation, references

def interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Персональний вчитель математики")

        # Input and task display
        topic_input = gr.Textbox(label="Введіть опис задачі", placeholder="Введіть опис задачі (наприклад, 'логарифмічні рівняння, легка складність')")
        generate_button = gr.Button("Згенерувати задачу")

        task_output = gr.Textbox(label="Задача", interactive=False)

        # Options in a row
        option_buttons = gr.Radio(label="Варіанти відповідей", interactive=True)
        result_output = gr.Textbox(label="Результат", interactive=False)

        # Reveal explanation and references
        reveal_button = gr.Button("Показати пояснення")
        explanation_output = gr.Textbox(label="Пояснення", interactive=False, visible=False)
        references_output = gr.Textbox(label="Посилання", interactive=False, visible=False)

        # State
        options_state = gr.State()
        correct_answer_state = gr.State()
        explanation_state = gr.State()
        references_state = gr.State()

        with gr.Row():
            with gr.Column():
                task_output
                option_buttons
                result_output
            with gr.Column():
                explanation_output
                references_output

        # Button logic
        def update_task(topic):
            task, options, correct_option, explanation, references = process_topic(topic)
            return (
                task,
                gr.update(choices=options),
                correct_option,
                explanation,
                references,
                gr.update(value=None),  # Clear result output
                gr.update(visible=False),  # Hide explanation
                gr.update(visible=False),  # Hide references
            )

        generate_button.click(
            update_task,
            inputs=[topic_input],
            outputs=[
                task_output,
                option_buttons,
                correct_answer_state,
                explanation_state,
                references_state,
                result_output,
                explanation_output,
                references_output,
            ],
        )

        def handle_answer(user_answer, correct_option):
            return validate_answer(user_answer, correct_option)

        option_buttons.change(
            handle_answer,
            inputs=[option_buttons, correct_answer_state],
            outputs=[result_output],
        )

        def handle_reveal(explanation, references):
            return gr.update(value=explanation, visible=True), gr.update(value=references, visible=True)

        reveal_button.click(
            handle_reveal,
            inputs=[explanation_state, references_state],
            outputs=[explanation_output, references_output],
        )

    demo.launch(share=True)

interface()
