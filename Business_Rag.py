from dotenv import load_dotenv
load_dotenv()

import os
import time
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.output_parsers import StrOutputParser


# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------

# List of competitor videos (YouTube Video IDs)
COMPETITOR_VIDEOS = {
    "JPMorgan": "_R5poBJlHHY",  # JPMorgan earnings call video ID"
    "Robinhood": "PpJSU4M87nw", # Robinhood earnings call video ID
     "Circle": "Kn9LnHZAm88",    # Circle earnings call video ID
}

# Or ask user for input
# for name in ["JPMorgan", "Robinhood", "Circle"]:
#     video_id = input(f"Enter YouTube Video ID for {name}: ").strip()
#     COMPETITOR_VIDEOS[name] = video_id


# ----------------------------------------------------
# EMBEDDING MODEL 
# ----------------------------------------------------

print("\nLoading embedding model... (first time may take a moment)\n")

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)


# ----------------------------------------------------
# TEXT SPLITTER
# ----------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)


# ----------------------------------------------------
# GEMINI LLM - Using gemini-3.6-flash (better quota)
# ----------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",  
    temperature=0.2,
)


# ----------------------------------------------------
# PROMPT TEMPLATE
# ----------------------------------------------------

prompt = PromptTemplate.from_template(
"""
You are an expert business analyst.

Answer ONLY from the provided transcript context.

If the answer is not available in the transcript,
reply only:

"I don't know."

Context:{context}

Question:{question}
"""
)


# ----------------------------------------------------
# HELPER FUNCTION
# ----------------------------------------------------

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ----------------------------------------------------
# FUNCTION: FETCH TRANSCRIPT
# ----------------------------------------------------

def fetch_transcript(video_id, video_name):
    """Fetch transcript for a given video ID."""
    print(f"  Fetching transcript for {video_name}...")
    
    try:
        transcript = YouTubeTranscriptApi().fetch(
            video_id,
            languages=["en"]
        )
        transcript_text = " ".join(chunk.text for chunk in transcript)
        print(f"  ✓ Transcript fetched ({len(transcript_text)} characters)")
        return transcript_text
    
    except TranscriptsDisabled:
        print(f"  ✗ Transcript unavailable for {video_name}")
        return None
    
    except Exception as e:
        print(f"  ✗ Error fetching transcript: {e}")
        return None


# ----------------------------------------------------
# FUNCTION: CREATE VECTORSTORE
# ----------------------------------------------------

def create_vectorstore(transcript_text, video_name):
    """Create FAISS vectorstore from transcript text."""
    documents = splitter.create_documents([transcript_text])
    print(f"  Created {len(documents)} chunks")
    
    # Process in batches
    batch_size = 50
    if len(documents) <= batch_size:
        vectorstore = FAISS.from_documents(documents, embeddings)
    else:
        vectorstore = FAISS.from_documents(documents[:batch_size], embeddings)
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            vectorstore.add_documents(batch)
            print(f"  Added batch {i//batch_size + 1}/{len(documents)//batch_size + 1}")
    
    return vectorstore


# ----------------------------------------------------
# FUNCTION: ASK QUESTION (WITH RETRY LOGIC)
# ----------------------------------------------------

def ask_question(vectorstore, question, video_name, max_retries=3):
    """Ask a question to a specific video's vectorstore with retry logic."""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    
    parser = StrOutputParser()
    chain = parallel_chain | prompt | llm | parser
    
    for attempt in range(max_retries):
        try:
            answer = chain.invoke(question)
            return answer
        except Exception as e:
            error_msg = str(e)
            if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                wait_time = 30 * (attempt + 1)
                print(f"  ⚠️ Quota exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ⚠️ Error: {error_msg[:100]}...")  # Show first 100 chars
                # Don't raise immediately - try again
                if attempt == max_retries - 1:
                    return f"Error: {error_msg[:200]}..."
                time.sleep(5)  # Wait before retry
    
    return "Error: Could not get answer after multiple retries."
# ----------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------

print("=" * 60)
print("BUSINESS CASE STUDY: COMPETITOR ANALYSIS")
print("=" * 60)

# Dictionary to store all vectorstores
vectorstores = {}

# Process each competitor video
print("\n STEP 1: Fetching and indexing competitor videos...\n")

for name, video_id in COMPETITOR_VIDEOS.items():
    print(f"Processing: {name}")
    
    # Fetch transcript
    transcript_text = fetch_transcript(video_id, name)
    
    if transcript_text is None:
        print(f"  Skipping {name} (no transcript available)\n")
        continue
    
    # Create vectorstore
    vectorstore = create_vectorstore(transcript_text, name)
    vectorstores[name] = vectorstore
    print(f"  ✓ {name} indexed successfully\n")
    
    # Small delay to avoid rate limits
    time.sleep(1)

if not vectorstores:
    print(" No transcripts available. Exiting.")
    exit()

print(f"\n Successfully indexed {len(vectorstores)} competitor videos.\n")


# ----------------------------------------------------
# BUSINESS ANALYSIS QUESTIONS
# ----------------------------------------------------

print("=" * 60)
print(" STEP 2: Business Analysis")
print("=" * 60)

# Define business questions (reduced to save quota)
business_questions = [
    #"What are the 3 biggest risks mentioned in this earnings call?",
    #e"How does the CEO describe the company's competitive advantage?",
    "What are the key financial metrics or targets mentioned?",
    "What challenges or headwinds are mentioned?",
]

# Ask each question for each competitor
for question in business_questions:
    print(f"\n QUESTION: {question}")
    print("-" * 50)
    
    for name, vectorstore in vectorstores.items():
        print(f"\n  [{name.upper()}]")
        answer = ask_question(vectorstore, question, name)
        print(f"  {answer}")
        time.sleep(2)  # 2 second delay between questions
    
    print("\n" + "=" * 50)


# ----------------------------------------------------
# INTERACTIVE MODE (CASE-INSENSITIVE FIX)
# ----------------------------------------------------

print("\n" + "=" * 60)
print(" STEP 3: Interactive Q&A Mode")
print("=" * 60)
print("Ask questions about specific competitors or compare them.")
print("Type 'exit' to quit.")
print("\nFormat examples:")
print("  - 'JPMorgan: What was their revenue growth?'")
print("  - 'Robinhood: How did they perform this quarter?'")
print("  - 'Circle: What's their strategy for growth?'")
print("  - 'all: What are the common themes across competitors?'")
print("  - 'compare: How do competitors differ in their AI strategy?'")
print(f"\nAvailable competitors: {list(vectorstores.keys())}")

while True:
    user_input = input("\nYour question: ").strip()
    
    if user_input.lower() == "exit":
        print("Exiting...")
        break
    
    # Parse input
    if ":" in user_input:
        target, question = user_input.split(":", 1)
        target = target.strip()
        question = question.strip()
        
        # CASE-INSENSITIVE FIX: Find matching competitor
        if target.lower() == "all":
            # Ask all competitors
            print("\n" + "-" * 40)
            for name, vectorstore in vectorstores.items():
                print(f"\n[{name.upper()}]")
                answer = ask_question(vectorstore, question, name)
                print(f"{answer}")
                time.sleep(2)
        
        elif target.lower() == "compare":
            # Compare answers across competitors
            print("\n" + "-" * 40)
            responses = {}
            for name, vectorstore in vectorstores.items():
                answer = ask_question(vectorstore, question, name)
                responses[name] = answer
                time.sleep(2)
            
            print("\n COMPARATIVE ANALYSIS:")
            for name, answer in responses.items():
                print(f"\n[{name.upper()}]")
                print(f"{answer}")
        
        else:
            # Find matching competitor (case-insensitive)
            matching_competitor = next(
                (name for name in vectorstores.keys() if name.lower() == target.lower()), 
                None
            )
            
            if matching_competitor:
                answer = ask_question(vectorstores[matching_competitor], question, matching_competitor)
                print(f"\n[{matching_competitor.upper()}]\n{answer}")
            else:
                print(f"Competitor '{target}' not found. Available: {list(vectorstores.keys())}")
    
    else:
        print("Format: 'competitor_name: your question'")
        print("Or use 'all:' or 'compare:'")