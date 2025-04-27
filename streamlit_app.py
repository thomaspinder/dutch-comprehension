"""Text generator module for Dutch language learning.

This module provides functions to generate Dutch text and comprehension questions
at various CEFR proficiency levels using OpenAI's API.
"""
import os
from typing import Dict, List, Tuple, Optional

import openai
import streamlit as st
from openai import OpenAI

# CEFR levels and their descriptions
CEFR_LEVELS = {
    "A1": "Beginner - Can understand and use familiar everyday expressions and basic phrases.",
    "A2": "Elementary - Can communicate in simple and routine tasks on familiar topics.",
    "B1": "Intermediate - Can deal with most situations likely to arise while travelling.",
    "B2": "Upper Intermediate - Can interact with a degree of fluency with native speakers.",
    "C1": "Advanced - Can express ideas fluently and spontaneously without much searching for expressions.",
    "C2": "Proficient - Can understand with ease virtually everything heard or read.",
}


class DutchTextGenerator:
    """Class for generating Dutch text and comprehension questions."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the text generator with an OpenAI API key.

        Args:
            api_key: Optional OpenAI API key. If None, it will use the key from environment.
        """
        self.client = OpenAI(api_key=api_key)

    def generate_text(
        self, level: str, topic: Optional[str] = None, word_count: int = 300
    ) -> str:
        """Generate Dutch text at the specified CEFR level.

        Args:
            level: CEFR level (A1, A2, B1, B2, C1, C2)
            topic: Optional topic for the text
            word_count: Approximate number of words for the text

        Returns:
            Generated Dutch text
        """
        topic_prompt = f" about {topic}" if topic else ""

        prompt = f"""Generate a Dutch text{topic_prompt} at CEFR level {level}.
        The text should be approximately {word_count} words long and suitable
        for a Dutch language learner at the {level} level.

        The text should:
        - Use vocabulary and grammar appropriate for {level} level
        - Be engaging and informative
        - Include some cultural references where appropriate
        - Be coherent and well-structured

        Please provide ONLY the Dutch text without any explanations or translations."""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Dutch language learning assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1000,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    def generate_questions(
        self, text: str, level: str, num_questions: int = 5
    ) -> List[Dict]:
        """Generate comprehension questions for the given Dutch text.

        Args:
            text: Dutch text to generate questions for
            level: CEFR level (A1, A2, B1, B2, C1, C2)
            num_questions: Number of questions to generate

        Returns:
            List of dictionaries containing questions, options, and answers
        """
        prompt = f"""Based on the following Dutch text at CEFR level {level},
        create {num_questions} multiple-choice reading comprehension questions.

        Text: "{text}"

        For each question:
        1. Create a question in Dutch that tests understanding of the text
        2. Provide 4 possible answers in Dutch (A, B, C, D)
        3. Indicate the correct answer

        Format your response as a JSON-compatible list of dictionaries with
        'question', 'options', and 'answer' keys.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Dutch language assessment assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content.strip()

        # Process the response - this would normally parse JSON, but for simplicity
        # we're assuming the model returned properly formatted data
        import json

        try:
            questions = json.loads(content)
            return questions.get("questions", [])
        except json.JSONDecodeError:
            # Fallback in case the response isn't proper JSON
            return [
                {
                    "question": "Error generating questions",
                    "options": ["A", "B", "C", "D"],
                    "answer": "A",
                }
            ]

# Page configuration
st.set_page_config(
    page_title="Learn Dutch - Reading Comprehension",
    page_icon="🇳🇱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Function to load API key
def get_api_key() -> str:
    """Get OpenAI API key from environment or user input.

    Returns:
        OpenAI API key string
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")

    if not api_key:
        api_key = st.sidebar.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use this app.",
        )
        if not api_key:
            st.sidebar.warning("Please enter your OpenAI API key to continue.")

    return api_key


# Initialize session state
if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""
if "questions" not in st.session_state:
    st.session_state.questions = []
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# App header
st.title("Learn Dutch - Reading Comprehension")
st.markdown(
    """
This app generates Dutch text at your selected proficiency level
and creates reading comprehension questions to test your understanding.
"""
)

# Sidebar for configuration
st.sidebar.header("Settings")

# Get API key
api_key = get_api_key()

# Only show the rest of the app if the API key is provided
if api_key:
    # Initialize text generator
    generator = DutchTextGenerator(api_key=api_key)

    # Language level selection
    st.sidebar.subheader("Dutch Proficiency Level")
    level = st.sidebar.radio(
        "Select your CEFR level:",
        list(CEFR_LEVELS.keys()),
        index=1,  # Default to A2
        format_func=lambda x: f"{x} - {CEFR_LEVELS[x]}",
    )

    # Topic input
    st.sidebar.subheader("Topic Guidance")
    topic = st.sidebar.text_input(
        "Enter a topic (optional):", placeholder="e.g., Dutch culture, travel, food..."
    )

    # Word count slider
    word_count = st.sidebar.slider(
        "Approximate text length (words):",
        min_value=100,
        max_value=500,
        value=250,
        step=50,
    )

    # Number of questions
    num_questions = st.sidebar.slider(
        "Number of questions:", min_value=3, max_value=10, value=5
    )

    # Generate button
    if st.sidebar.button("Generate Text & Questions"):
        with st.spinner("Generating Dutch text..."):
            st.session_state.generated_text = generator.generate_text(
                level=level, topic=topic if topic else None, word_count=word_count
            )

        with st.spinner("Creating comprehension questions..."):
            st.session_state.questions = generator.generate_questions(
                text=st.session_state.generated_text,
                level=level,
                num_questions=num_questions,
            )

        # Reset answers and results
        st.session_state.user_answers = {}
        st.session_state.show_results = False

    # Display generated text if available
    if st.session_state.generated_text:
        st.subheader("Dutch Text")
        st.markdown(f"**Level: {level}**")
        if topic:
            st.markdown(f"**Topic: {topic}**")

        st.write(st.session_state.generated_text)

        # Display questions if available
        if st.session_state.questions:
            st.subheader("Comprehension Questions")

            # Create a form for submitting answers
            with st.form("quiz_form"):
                for i, q in enumerate(st.session_state.questions):
                    st.markdown(f"**Question {i+1}:** {q['question']}")

                    # Create radio buttons for options
                    options = q.get("options", [])
                    if isinstance(options, list):
                        option_labels = [f"{opt}" for opt in options]
                    else:
                        # Handle case where options might be a dictionary
                        option_labels = [
                            f"{key}: {value}" for key, value in options.items()
                        ]

                    st.session_state.user_answers[i] = st.radio(
                        f"Select your answer for question {i+1}:",
                        options=option_labels,
                        key=f"q_{i}",
                    )

                # Submit button
                submitted = st.form_submit_button("Submit Answers")

                if submitted:
                    st.session_state.show_results = True

            # Show results if the form was submitted
            if st.session_state.show_results:
                st.subheader("Results")

                correct_count = 0
                for i, q in enumerate(st.session_state.questions):
                    user_answer = st.session_state.user_answers[i]
                    correct_answer = q.get("answer", "")

                    # Check if answer is correct (assuming options are labeled A, B, C, D)
                    is_correct = False
                    if isinstance(q.get("options", []), list):
                        correct_index = ord(correct_answer) - ord("A")
                        if 0 <= correct_index < len(q.get("options", [])):
                            is_correct = (
                                user_answer == q.get("options", [])[correct_index]
                            )
                    else:
                        is_correct = user_answer.startswith(correct_answer)

                    if is_correct:
                        correct_count += 1
                        st.success(f"Question {i+1}: Correct! ✓")
                    else:
                        st.error(
                            f"Question {i+1}: Incorrect ✗ - Correct answer: {correct_answer}"
                        )

                # Display score
                percentage = (correct_count / len(st.session_state.questions)) * 100
                st.markdown(
                    f"### Your Score: {correct_count}/{len(st.session_state.questions)} ({percentage:.1f}%)"
                )

                # Encouragement message based on score
                if percentage >= 80:
                    st.balloons()
                    st.success("Excellent! You're making great progress with Dutch!")
                elif percentage >= 60:
                    st.success("Good job! Keep practicing to improve further.")
                else:
                    st.info("Keep practicing! Dutch takes time to master.")


# Main function to run the app
def main():
    """Run the Streamlit app."""
    # The app is already defined above, so we just pass to allow imports
    pass


if __name__ == "__main__":
    main()
