import os
import json
import streamlit as st
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# --- 1. Persistent Memory Engine ---
MEMORY_FILE = "memory.json"

class HindsightMemory:
    def __init__(self):
        self.bank_a_style_rules: List[str] = []
        self.bank_b_content_history: List[str] = []
        self.load_memory()

    def load_memory(self):
        """Loads memory from a local JSON file to persist across sessions."""
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    data = json.load(f)
                    self.bank_a_style_rules = data.get("bank_a_style_rules", [])
                    self.bank_b_content_history = data.get("bank_b_content_history", [])
            except Exception:
                pass # Fallback to empty memory on corrupt file

    def save_memory(self):
        """Saves current memory to a local JSON file."""
        with open(MEMORY_FILE, 'w') as f:
            json.dump({
                "bank_a_style_rules": self.bank_a_style_rules,
                "bank_b_content_history": self.bank_b_content_history
            }, f, indent=4)

    def retain_rule(self, rule: str):
        if rule not in self.bank_a_style_rules:
            self.bank_a_style_rules.append(rule)
            self.save_memory()

    def retain_content(self, topic: str):
        topic_clean = topic.lower().strip()
        if topic_clean not in self.bank_b_content_history:
            self.bank_b_content_history.append(topic_clean)
            self.save_memory()
            
    def clear_memory(self):
        self.bank_a_style_rules = []
        self.bank_b_content_history = []
        self.save_memory()

    def recall_style_guidelines(self) -> str:
        if not self.bank_a_style_rules:
            return "No custom brand guidelines stored yet."
        return "\n".join([f"- {rule}" for rule in self.bank_a_style_rules])

    def check_topic_overlap(self, new_topic: str) -> bool:
        new_topic_clean = new_topic.lower().strip()
        # Filter out common stop words to improve accuracy of the overlap check
        stopwords = {"about", "there", "their", "which", "would", "these", "this", "that", "with", "from"}
        words = [w for w in new_topic_clean.split() if len(w) > 3 and w not in stopwords]
        
        for past_topic in self.bank_b_content_history:
            if any(word in past_topic for word in words):
                return True
        return False

# --- 2. Structured Output Model ---
class AgentPostResponse(BaseModel):
    overlap_detected: bool = Field(description="True if the topic overlaps with previously saved content")
    warning_message: Optional[str] = Field(description="Alert message to user if overlap is detected", default=None)
    post_draft: str = Field(description="The final social media post content")
    extracted_rules: List[str] = Field(description="Any new brand style rules extracted from user input", default=[])

parser = PydanticOutputParser(pydantic_object=AgentPostResponse)

# --- 3. Agent Function ---
def run_brandcraft_agent(user_input: str, topic_summary: str, memory: HindsightMemory, api_key: str, model_choice: str) -> AgentPostResponse:
    llm = ChatGoogleGenerativeAI(model=model_choice, temperature=0.3, google_api_key=api_key)
    
    SYSTEM_PROMPT = """
You are BrandCraft AI, an intelligent social media content strategist.

### STORED BRAND GUIDELINES (MEMORY BANK A):
{recalled_style_rules}

### TOPIC OVERLAP STATUS (MEMORY BANK B):
Overlap Detected: {overlap_detected}

---
### INSTRUCTIONS:
1. If 'Overlap Detected' is True, generate a fresh and highly unique angle to avoid audience fatigue, and provide a concise 'warning_message'.
2. Identify if the user input contains explicit preferences or rules (e.g., "don't use emojis", "keep sentences short"). Extract them as specific, reusable instructions into 'extracted_rules'.
3. Draft the social media post adhering STRICTLY to ALL stored brand guidelines and new instructions.

{format_instructions}
"""
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", "Topic: {topic_summary}\n\nInstructions: {user_input}")
    ])
    
    chain = prompt_template | llm | parser
    
    has_overlap = memory.check_topic_overlap(topic_summary)
    active_rules = memory.recall_style_guidelines()
    
    response = chain.invoke({
        "recalled_style_rules": active_rules,
        "overlap_detected": has_overlap,
        "user_input": user_input,
        "topic_summary": topic_summary,
        "format_instructions": parser.get_format_instructions()
    })
    
    for rule in response.extracted_rules:
        memory.retain_rule(rule)
    memory.retain_content(topic_summary)
    
    return response

# --- Streamlit UI ---
st.set_page_config(page_title="BrandCraft AI", page_icon="✍️", layout="wide")

# Initialize memory in session state if not present
if "memory" not in st.session_state:
    st.session_state.memory = HindsightMemory()

st.title("✍️ BrandCraft AI - The Perfect Prototype")
st.markdown("Your intelligent social media content strategist with **Persistent Hindsight Memory**.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get your key from Google AI Studio")
    model_choice = st.selectbox("Model", ["gemini-3.5-flash-lite", "gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-pro"])
    
    st.divider()
    st.header("🧠 Memory Bank A (Style Rules)")
    if st.session_state.memory.bank_a_style_rules:
        for rule in st.session_state.memory.bank_a_style_rules:
            st.caption(f"- {rule}")
    else:
        st.caption("No rules stored yet.")
        
    st.header("📚 Memory Bank B (Content History)")
    if st.session_state.memory.bank_b_content_history:
        for topic in st.session_state.memory.bank_b_content_history:
            st.caption(f"- {topic.title()}")
    else:
        st.caption("No topics stored yet.")
        
    st.divider()
    if st.button("🗑️ Clear All Memory"):
        st.session_state.memory.clear_memory()
        st.success("Memory cleared!")
        st.rerun()

st.subheader("Create a New Post")
col1, col2 = st.columns([1, 2])

with col1:
    topic_input = st.text_input("Topic Summary", placeholder="e.g. AI Analytics Feature Launch")
    st.info("The topic summary helps the AI track what you've posted about in the past to prevent overlap.")

with col2:
    prompt_input = st.text_area("Post Instructions / Feedback", placeholder="e.g. Draft a LinkedIn post announcing our new feature. Make it less formal, no emojis.", height=150)

if st.button("✨ Generate Post", type="primary", use_container_width=True):
    if not api_key:
        st.error("⚠️ Please enter your Google Gemini API key in the sidebar.")
    elif not topic_input or not prompt_input:
        st.warning("⚠️ Please provide both a topic summary and instructions.")
    else:
        with st.spinner("BrandCraft AI is analyzing memory and crafting your post..."):
            try:
                response = run_brandcraft_agent(prompt_input, topic_input, st.session_state.memory, api_key, model_choice)
                
                if response.overlap_detected and response.warning_message:
                    st.warning(f"⚠️ **Topic Overlap Detected in Memory Bank B:** {response.warning_message}")
                
                if response.extracted_rules:
                    st.success(f"🧠 **Learned new rules for Memory Bank A:** {', '.join(response.extracted_rules)}")
                    
                st.markdown("### 📝 Drafted Post")
                # Show inside an aesthetic container
                with st.container(border=True):
                    st.write(response.post_draft)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
