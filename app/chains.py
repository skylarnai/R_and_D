"""STEP 1 - the LangChain flow.

Every LLM step of the RAG pipeline is defined here as an LCEL chain:

    grader_chain   : (question, chunk)            -> ChunkGrade     (CRAG relevance check)
    rewrite_chain  : (question, chat_history)     -> web search query
    generate_chain : (context, question, history) -> final answer

Each chain is ChatPromptTemplate | model | StrOutputParser. LangGraph (graph.py)
reuses these chains as the bodies of its nodes.
"""
from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda

from app.llm import get_llm
from app.schemas import ChunkGrade, RetrievedChunk, WebResult

# ---------------- CRAG grader ----------------
GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade whether a document chunk is relevant to a user question. "
            "A chunk is relevant if it contains facts, keywords or meaning that help "
            "answer the question. Reply with exactly one word: yes or no.",
        ),
        ("human", "Question: {question}\n\nDocument chunk:\n{chunk}"),
    ]
)


@lru_cache
def get_grader_chain() -> Runnable:
    return (
        GRADER_PROMPT
        | get_llm(temperature=0.0)
        | StrOutputParser()
        | RunnableLambda(ChunkGrade.from_llm_text)
    )


# ---------------- Query rewriter (for web search) ----------------
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a clinical and healthcare query analyzer. Determine if the user's latest question relates to medicine, healthcare, pathology, laboratory reports, symptoms, diseases, medications, nutrition, diet, wellness, or healthy living.\n"
            "If the question is completely non-medical and unrelated to health (such as coding, weather forecasts, politics, sports, entertainment, movies, or pop culture), reply with exactly: NON_MEDICAL.\n"
            "Otherwise, rewrite the question as one concise, self-contained medical/health web search query. Use chat history to resolve pronouns like 'it' or 'that'. "
            "Return only the query or NON_MEDICAL, nothing else.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


@lru_cache
def get_rewrite_chain() -> Runnable:
    return REWRITE_PROMPT | get_llm(temperature=0.0) | StrOutputParser()


# ---------------- Answer generation ----------------
GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert Clinical and Medical AI Assistant delivering concise, clear, and professional medical guidance.\n\n"
            "Communication & Tone Guidelines:\n"
            "1. Be Direct & Concise:\n"
            "   - Answer ONLY what the user explicitly asks. Do not dump unrequested lab panels, full histories, or unrelated test results.\n"
            "   - NO conversational preamble, introductions, or repetitive filler (e.g., NEVER say 'Hello Rohan, I am your Medical AI Assistant and I am ready to...'). Jump directly into the clinical answer.\n"
            "   - Maintain a crisp, professional clinical tone. Favor structured bullet points and short paragraphs over lengthy essays.\n"
            "2. Clinical Grounding:\n"
            "   - Base all clinical statements strictly on the provided context ({context_origin}).\n"
            "   - When citing lab tests, mention only the relevant parameter, observed value, unit, reference interval, and clinical status (Normal / High / Low) concisely (e.g., 'Triglycerides: 186.0 mg/dL [Elevated, reference: <150 mg/dL]').\n"
            "3. Targeted Dietary & Lifestyle Guidance:\n"
            "   - If the user asks for food, dietary, or lifestyle recommendations, provide 3 to 5 targeted, evidence-based bullet points directly addressing the specific abnormal findings requested.\n"
            "4. Scope & Greetings:\n"
            "   - Pure Greetings: ONLY if the user's message is solely a greeting (e.g., 'Hi', 'Hello'), reply warmly and concisely in 1-2 sentences offering assistance.\n"
            "   - Off-Topic: If the query is completely non-medical (e.g. software, sports, movies), decline politely in 1-2 sentences.\n"
            "5. Brief Clinical Disclaimer:\n"
            "   - Conclude with a single brief sentence advising consultation with the physician or healthcare provider for specific clinical decisions.\n\n"
            "Context origin: {context_origin}\n\nContext:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)


@lru_cache
def get_generate_chain() -> Runnable:
    return GENERATE_PROMPT | get_llm(temperature=0.2) | StrOutputParser()


# ---------------- Context formatting helpers ----------------
def format_chunks(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{c.filename} #{c.chunk_index}]\n{c.content}" for c in chunks
    )


def format_web_results(results: list[WebResult]) -> str:
    return "\n\n".join(f"[{r.title}] ({r.url})\n{r.content}" for r in results)