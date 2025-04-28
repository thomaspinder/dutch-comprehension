"""Text generator module for Dutch language learning.

This module provides functions to generate Dutch text and comprehension questions
at various CEFR proficiency levels using OpenAI's API, as well as scraping and
summarizing Dutch news articles from NOS.nl.
"""

import json
import os
import requests
from typing import Dict, List, Tuple, Optional, Any, Literal
from datetime import datetime

import openai
import streamlit as st
from openai import OpenAI
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd

# CEFR levels and their descriptions
CEFR_LEVELS = {
    "A1": "Beginner - Can understand and use familiar everyday expressions and basic phrases.",
    "A2": "Elementary - Can communicate in simple and routine tasks on familiar topics.",
    "B1": "Intermediate - Can deal with most situations likely to arise while travelling.",
    "B2": "Upper Intermediate - Can interact with a degree of fluency with native speakers.",
    "C1": "Advanced - Can express ideas fluently and spontaneously without much searching for expressions.",
    "C2": "Proficient - Can understand with ease virtually everything heard or read.",
}

Level = Literal["A1", "A2", "B1", "B2", "C1", "C2"]


class VocabularyItem(BaseModel):
    """A challenging Dutch vocabulary word with its English translation."""

    dutch: str = Field(
        description="Dutch word that might be challenging for the given level"
    )
    english: str = Field(description="English translation of the Dutch word")

    model_config = ConfigDict(
        frozen=True,  # Make instances immutable
        json_schema_extra={"examples": [{"dutch": "oorlog", "english": "war"}]},
    )


class SummaryResponse(BaseModel):
    """Structured response containing a Dutch text summary and challenging vocabulary."""

    summary: str = Field(
        description="Text summarized in Dutch at the specified CEFR level"
    )
    vocabulary: List[VocabularyItem] = Field(
        description="List of challenging vocabulary with translations"
    )
    level: Level = Field(description="CEFR level of the summary")
    original_text_length: Optional[int] = Field(
        None, description="Length of the original text in characters"
    )
    summary_length: Optional[int] = Field(
        None, description="Length of the summary in characters"
    )

    model_config = ConfigDict(
        frozen=True,  # Make instances immutable
    )

    def word_count(self) -> int:
        """Count the number of words in the summary."""
        return len(self.summary.split())

    def vocabulary_count(self) -> int:
        """Count the number of vocabulary items."""
        return len(self.vocabulary)


class Article(BaseModel):
    """A news article with title, publication date, and content."""

    title: str = Field(description="Article title")
    published_at: datetime = Field(description="Publication timestamp")
    text: str = Field(description="Article text content")

    model_config = ConfigDict(
        frozen=True,  # Make instances immutable
    )


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

    def scrape_nos_article(self, url: str) -> Article:
        """
        Fetches an NOS news article page and returns its structured content.

        Args:
            url: URL of the NOS article to scrape

        Returns:
            Article object containing title, publication date and text content
        """
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        # 1) Pull out the embedded Next.js data blob
        soup = BeautifulSoup(resp.text, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__")
        payload = json.loads(script.string)

        # 2) Navigate to the article data
        article_data = payload["props"]["pageProps"]["data"]

        # 3) Extract title and publication timestamp
        title = article_data["title"]
        published_at = datetime.fromisoformat(article_data["publishedAt"])

        # 4) Walk through each "text" item, strip HTML tags
        paragraphs = []
        for item in article_data.get("items", []):
            if item.get("type") == "text" and item.get("text"):
                # parse the fragment as HTML, then extract only its text
                fragment = BeautifulSoup(item["text"], "html.parser")
                clean = fragment.get_text(separator=" ", strip=True)
                paragraphs.append(clean)
        all_text = " ".join(paragraphs)

        return Article(
            title=title,
            published_at=published_at,
            text=all_text,
        )

    def summarize_at_level(
        self, text: str, level: Level, model: str = "gpt-4o-mini"
    ) -> SummaryResponse:
        """
        Summarise `text` in Dutch at the specified CEFR `level` and return structured output.

        Args:
            text: The Dutch text to summarize
            level: CEFR language proficiency level
            model: OpenAI model to use

        Returns:
            SummaryResponse containing the summary and vocabulary list
        """
        system_prompt = (
            "You are a helpful Dutch language teacher. "
            f"Write a summary of the following text at CEFR level {level}. "
            "If there is vocabulary that is not familiar to a beginner, "
            "explain it in a way that is easy to understand."
        )
        user_prompt = (
            f"Summarize the following text into one coherent, concise story, "
            f"suitable for level {level}.\n\n"
            f"{text}\n\n"
            f"Provide your response as a JSON object with two fields:\n"
            f"1. 'summary': Your Dutch summary at level {level}\n"
            f"2. 'vocabulary': A list of challenging vocabulary for level {level} readers, "
            f"where each item is an object with 'dutch' and 'english' fields"
        )

        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
            max_tokens=1000,
        )

        result_dict = json.loads(response.choices[0].message.content)

        # Add metadata
        result_dict["level"] = level
        result_dict["original_text_length"] = len(text)
        result_dict["summary_length"] = len(result_dict["summary"])

        # Convert the result to a Pydantic model
        return SummaryResponse(**result_dict)


def show_questions_and_results(questions, user_answers_key_prefix="q_"):
    user_answers, submitted = {}, False
    with st.form(f"quiz_form_{user_answers_key_prefix}"):
        for i, q in enumerate(questions):
            st.markdown(f"**Question {i+1}:** {q['question']}")
            options = q.get("options", [])
            if isinstance(options, list):
                option_labels = [f"{opt}" for opt in options]
            else:
                option_labels = [f"{key}: {value}" for key, value in options.items()]
            user_answers[i] = st.radio(
                f"Select your answer for question {i+1}:",
                options=option_labels,
                key=f"{user_answers_key_prefix}{i}",
            )
        submitted = st.form_submit_button("Submit Answers")
    if submitted:
        st.session_state.user_answers = user_answers
        st.session_state.show_results = True
    if st.session_state.show_results:
        correct_count = 0
        for i, q in enumerate(questions):
            user_answer = st.session_state.user_answers.get(i, "")
            correct_answer = q.get("answer", "")
            is_correct = False
            options = q.get("options", [])
            if isinstance(options, list):
                correct_index = ord(correct_answer) - ord("A")
                if 0 <= correct_index < len(options):
                    is_correct = user_answer == options[correct_index]
            else:
                is_correct = user_answer.startswith(correct_answer)
            if is_correct:
                correct_count += 1
                st.success(f"Question {i+1}: Correct! ✓")
            else:
                st.error(
                    f"Question {i+1}: Incorrect ✗ - Correct answer: {correct_answer}"
                )
        percentage = (correct_count / len(questions)) * 100
        st.markdown(
            f"### Your Score: {correct_count}/{len(questions)} ({percentage:.1f}%)"
        )
        if percentage >= 80:
            st.balloons()
            st.success("Excellent! You're making great progress with Dutch!")
        elif percentage >= 60:
            st.success("Good job! Keep practicing to improve further.")
        else:
            st.info("Keep practicing! Dutch takes time to master.")


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
def init_state():
    defaults = {
        "generated_text": "",
        "questions": [],
        "user_answers": {},
        "show_results": False,
        "nos_article": None,
        "nos_summary": None,
        "active_tab": "Generated Text",
        "level": "A2",
        "num_questions": 5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# Initialize session state
init_state()

# App header
st.markdown("# Learn Dutch - Reading Comprehension", unsafe_allow_html=True)
# Use a compact subheader with less vertical space
st.markdown(
    "<div style='margin-bottom:0.5rem;'>This app helps you learn Dutch through reading comprehension exercises. Generate Dutch text at your level or read authentic Dutch news articles.</div>",
    unsafe_allow_html=True,
)

# Get API key
api_key = get_api_key()

# Only show the rest of the app if the API key is provided
if api_key:
    # Initialize text generator
    generator = DutchTextGenerator(api_key=api_key)

    # Create tabs and track the selected tab
    tab_names = ["Generated Text", "NOS Articles"]
    tabs = st.tabs(tab_names)

    # Sidebar content - only language level selector and number of questions
    st.sidebar.header("CEFR Level")
    default_index = 1 if st.session_state.active_tab == "Generated Text" else 0
    level_labels = list(CEFR_LEVELS.keys())
    level_help = {k: v for k, v in CEFR_LEVELS.items()}

    def radio_with_tooltips(label, options, index, help_dict):
        selected = st.sidebar.radio(label, options, index=index, format_func=str)
        for i, opt in enumerate(options):
            if opt == selected:
                st.sidebar.caption(f"{help_dict[opt]}")
        return selected

    st.session_state.level = radio_with_tooltips(
        "Select your Dutch proficiency level:",
        level_labels,
        default_index,
        level_help,
    )
    level = st.session_state.level

    # Number of questions slider (shared)
    st.session_state.num_questions = st.sidebar.slider(
        "Number of questions:",
        min_value=3,
        max_value=10,
        value=5,
        key="sidebar_num_questions",
    )

    # Display content based on active tab
    with tabs[0]:  # Generated Text tab
        st.header("Generated Dutch Text")
        st.markdown(
            """
        This tab generates Dutch text at your selected proficiency level
        and creates reading comprehension questions to test your understanding.
        """
        )

        # Controls for Generated Text tab - moved from sidebar to body
        col1, col2 = st.columns(2)

        with col1:
            # Topic input
            topic = st.text_input(
                "Enter a topic (optional):",
                placeholder="e.g., Dutch culture, travel, food...",
            )

        with col2:
            # Word count slider
            word_count = st.slider(
                "Approximate text length (words):",
                min_value=100,
                max_value=500,
                value=250,
                step=50,
            )

        num_questions = st.session_state.num_questions
        if st.button("Generate Text & Questions"):
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
            st.session_state.user_answers = {}
            st.session_state.show_results = False

        # Display generated text if available
        if (
            st.session_state.generated_text
            and st.session_state.active_tab == "Generated Text"
        ):
            st.subheader("Dutch Text")
            st.markdown(f"**Level: {level}**")
            if "topic" in locals() and topic:
                st.markdown(f"**Topic: {topic}**")

            st.write(st.session_state.generated_text)

            # Display questions if available
            if st.session_state.questions:
                st.subheader("Comprehension Questions")
                show_questions_and_results(
                    st.session_state.questions, user_answers_key_prefix="q_"
                )

    with tabs[1]:  # NOS Articles tab
        st.header("NOS News Articles")
        st.markdown(
            """
        This tab allows you to read authentic Dutch news articles from NOS.nl,
        summarized at your selected CEFR proficiency level with vocabulary support.
        """
        )
        nos_url = st.text_input(
            "Enter NOS article URL:", placeholder="https://nos.nl/artikel/..."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("")  # Placeholder for layout symmetry
        with col2:
            nos_num_questions = st.session_state.num_questions
        if st.button("Fetch & Summarize Article"):
            if not nos_url or not nos_url.startswith("https://nos.nl/artikel/"):
                st.error("Please enter a valid NOS article URL")
            else:
                try:
                    with st.spinner("Fetching article from NOS.nl..."):
                        st.session_state.nos_article = generator.scrape_nos_article(
                            nos_url
                        )
                    with st.spinner(f"Summarizing article at level {level}..."):
                        st.session_state.nos_summary = generator.summarize_at_level(
                            st.session_state.nos_article.text, level
                        )
                    # Always generate questions
                    with st.spinner("Creating comprehension questions..."):
                        st.session_state.questions = generator.generate_questions(
                            text=st.session_state.nos_summary.summary,
                            level=level,
                            num_questions=nos_num_questions,
                        )
                    st.session_state.user_answers = {}
                    st.session_state.show_results = False
                    st.session_state.active_tab = "NOS Articles"
                except Exception as e:
                    st.error(f"Error processing article: {str(e)}")
        # Display article and summary if available
        if (
            st.session_state.nos_article
            and st.session_state.nos_summary
            and st.session_state.active_tab == "NOS Articles"
        ):
            article = st.session_state.nos_article
            summary = st.session_state.nos_summary

            # Display article info
            st.subheader("Article Information")
            st.markdown(f"**Title:** {article.title}")
            st.markdown(
                f"**Published:** {article.published_at.strftime('%d %B %Y, %H:%M')}"
            )

            st.markdown(f"### Level {summary.level} Summary")
            st.write(summary.summary)
            st.caption(
                f"Summary length: {summary.summary_length} characters, {summary.word_count()} words"
            )

            # Display vocabulary
            st.markdown("### Challenging Vocabulary")
            if summary.vocabulary:
                vocab_df = pd.DataFrame(
                    [(item.dutch, item.english) for item in summary.vocabulary],
                    columns=["Dutch", "English"],
                )
                st.table(vocab_df)
            else:
                st.caption("No challenging vocabulary found.")
            st.caption(f"Total vocabulary items: {summary.vocabulary_count()}")

            # Display questions if available
            if st.session_state.questions:
                st.subheader("Comprehension Questions")
                show_questions_and_results(
                    st.session_state.questions, user_answers_key_prefix="nos_q_"
                )


# Main function to run the app
def main():
    """Run the Streamlit app."""
    # The app is already defined above, so we just pass to allow imports
    pass


if __name__ == "__main__":
    main()
