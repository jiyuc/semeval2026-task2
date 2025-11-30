import os
from typing import Optional, Dict, Any, List
import pandas as pd
import argparse

# LLM + runnables imports
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

# Vector store / documents / schema
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

# embeddings wrapper (uses sentence-transformers under the hood)
from sentence_transformers import SentenceTransformer


# -------------------------
# Small adapter to meet LangChain Embeddings API
# (LangChain expects an object with embed_documents & embed_query)
# -------------------------
class SentTransformersEmbeddingsAdapter:
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # returns list of vectors
        embs = self.model.encode(texts, convert_to_numpy=True)
        # ensure python list-of-lists
        return [emb.tolist() for emb in embs]

    def embed_query(self, text: str) -> List[float]:
        emb = self.model.encode([text], convert_to_numpy=True)[0]
        return emb.tolist()


def load_llm() -> Optional[LlamaCpp]:
    try:
        llm: Optional[LlamaCpp] = LlamaCpp(
            temperature=0.0,
            model_path="/path/to/your/llama.gguf",
        )
    except Exception as e:
        print(f"Error loading LLM: {e}")
        llm = None

    return llm


# -------------------------
# Build runnable chains (these will be connected to llm later in main)
# -------------------------
# We create prompt templates now; will pipe to llm after llm is loaded.
infer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that can infer the emotion of a given text."),
    ("human", "What is the emotion of this text? Given several word phrases. {text}"),
])

generate_expression_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI that can generate an emotional expression."),
    ("human", "Generate emotional expressions based on the following keywords {feeling_words}"),
])

final_inference_prompt = ChatPromptTemplate.from_messages([
    ("system", "You produce a final sentiment inference using text + auxiliary + retrieved examples."),
    ("human", (
        "You will be given:\n"
        "  1) text: the original text\n"
        "  2) aux: output from upstream chain (inferred emotion labels or generated expressions)\n"
        "  3) rag: top retrieved examples from training data\n\n"
        "Using all three, produce a JSON object with keys:\n"
        "  \"valence\": number from -2 to 2\n"
        "  \"arousal\": number from 0 to 2\n\n"
        "Do not invent facts.\n\n"
        "text: {text}\n"
        "aux: {aux}\n"
        "rag: {rag}\n\n"
        "Return ONLY valid JSON."
    ))
])


# These global runnable variables will be assigned after llm is ready
infer_emotion_chain: Optional[Runnable] = None
generate_expression_chain: Optional[Runnable] = None
final_inference_chain: Optional[Runnable] = None

# retriever / vectorstore will be created after train_df is loaded
retriever = None


def build_chains_and_vectorstore(train_df: pd.DataFrame, embedding_model_name: str = "all-mpnet-base-v2"):
    """
    Build vectorstore (from train_df) and connect runnable chains to the LLM.
    Returns (llm, retriever)
    """
    global infer_emotion_chain, generate_expression_chain, final_inference_chain, retriever

    # instantiate embeddings adapter
    emb_adapter = SentTransformersEmbeddingsAdapter(model_name=embedding_model_name)

    # build documents from train_df
    docs = [
        Document(page_content=str(row_text), metadata={"row": int(idx)})
        for idx, row_text in enumerate(train_df["text"].astype(str).tolist())
    ]

    # Create vectorstore from documents (this will call embed_documents internally)
    try:
        vectorstore = InMemoryVectorStore.from_documents(documents=docs, embedding=emb_adapter)
    except Exception as e:
        raise RuntimeError(f"Failed to build InMemoryVectorStore: {e}")

    # Create a retriever (k can be tuned)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Load LLM (fail early)
    llm = load_llm()
    if llm is None:
        raise RuntimeError("Failed to load LLM. Aborting.")

    # Connect prompt templates -> llm -> parser
    infer_emotion_chain = (infer_prompt | llm | StrOutputParser())
    generate_expression_chain = (generate_expression_prompt | llm | StrOutputParser())
    final_inference_chain = (final_inference_prompt | llm | StrOutputParser())

    return llm, retriever


def run_pipeline(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    inputs must contain:
      - 'text' (the original text, str)
      - 'is_words' (bool) to choose branch
      - 'feeling_words' (str) if is_words True
    """
    global infer_emotion_chain, generate_expression_chain, final_inference_chain, retriever

    if infer_emotion_chain is None or generate_expression_chain is None or final_inference_chain is None:
        raise RuntimeError("Runnables not initialized. Call build_chains_and_vectorstore(...) first.")

    text = inputs.get("text", "")
    is_words = bool(inputs.get("is_words", False))

    # 1) Route to branch
    if is_words:
        branch_input = {"feeling_words": inputs.get("feeling_words", "")}
        aux_output = generate_expression_chain.invoke(branch_input)
    else:
        branch_input = {"text": text}
        aux_output = infer_emotion_chain.invoke(branch_input)

    # 2) RAG retrieval using retriever
    if retriever is None:
        raise RuntimeError("Retriever not initialized. Call build_chains_and_vectorstore(...) first.")

    retrieved_docs = retriever.get_relevant_documents(text)  # list[Document]
    # format retrieved passages for the final prompt (simple string list)
    rag_formatted = []
    for d in (retrieved_docs or []):
        if hasattr(d, "page_content"):
            rag_formatted.append(d.page_content)
        elif isinstance(d, dict) and "page_content" in d:
            rag_formatted.append(d["page_content"])
        elif isinstance(d, str):
            rag_formatted.append(d)
        else:
            rag_formatted.append(str(d))

    # 3) Final inference: pass text, aux and rag
    final_inputs = {"text": text, "aux": aux_output, "rag": rag_formatted}
    final_output = final_inference_chain.invoke(final_inputs)
    return final_output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="", required=True)
    parser.add_argument("--dev", type=str, default="")
    parser.add_argument("--test", type=str, default="", required=True)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    # Concatenate valence/arousal into text column
    if {"valence", "arousal"}.issubset(train_df.columns):
        train_df["text"] = (
                test_df["text"].astype(str)
                + " valence: " + train_df["valence"].astype(str)
                + " arousal: " + train_df["arousal"].astype(str)
        )
    else:
        raise ValueError("train_df must contain 'valence' and 'arousal' columns")

    # initialize chains + retriever (this will throw if something is missing)
    build_chains_and_vectorstore(train_df)

    # iterate test examples and call pipeline
    for _, row in test_df.iterrows():
        text = row.get("text", "")
        try:
            out = run_pipeline({"text": text, "is_words": False})
            print(out)
        except Exception as e:
            print(f"Error processing text: {text!r} -> {e}")