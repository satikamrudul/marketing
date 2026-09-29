import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Ensure your OpenAI API key is set
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# =====================================================================
# 1. HINDSIGHT MEMORY ENGINE CLASS
# =====================================================================
class HindsightMemory:
    """
    Implements Retain, Recall, and Audit operations across two memory banks:
    - Bank A: Brand & Style Guidelines
    - Bank B: Published/Drafted Content History
    """
    def __init__(self):
        self.bank_a_style_rules: List[str] = []
        self.bank_b_content_history: List[str] = []

    def retain_rule(self, rule: str):
        """Retain new style/formatting rules into Memory Bank A."""
        if rule not in self.bank_a_style_rules:
            self.bank_a_style_rules.append(rule)

    def retain_content(self, topic: str):
        """Retain published/drafted topics into Memory Bank B."""
        self.bank_b_content_history.append(topic.lower().strip())

    def recall_style_guidelines(self) -> str:
        """Recall stored brand rules for injection into prompt context."""
        if not self.bank_a_style_rules:
            return "No custom brand guidelines stored yet."
        return "\n".join([f"- {rule}" for rule in self.bank_a_style_rules])

    def check_topic_overlap(self, new_topic: str) -> bool:
        """Recall content history and check for significant topic overlap."""
        new_topic_clean = new_topic.lower().strip()
        for past_topic in self.bank_b_content_history:
            # Simple keyword overlap check for prototype speed
            if any(word in past_topic for word in new_topic_clean.split() if len(word) > 3):
                return True
        return False


# =====================================================================
# 2. STRUCTURED OUTPUT MODEL
# =====================================================================
class AgentPostResponse(BaseModel):
    overlap_detected: bool = Field(description="True if the topic overlaps with previously saved content")
    warning_message: Optional[str] = Field(description="Alert message to user if overlap is detected", default=None)
    post_draft: str = Field(description="The final social media post content")
    extracted_rules: List[str] = Field(description="Any new brand style rules extracted from user input", default=[])

parser = PydanticOutputParser(pydantic_object=AgentPostResponse)


# =====================================================================
# 3. LANGCHAIN PROMPT & CHAIN SETUP
# =====================================================================
SYSTEM_PROMPT = """
You are BrandCraft AI, an intelligent social media content strategist.

### STORED BRAND GUIDELINES (MEMORY BANK A):
{recalled_style_rules}

### TOPIC OVERLAP STATUS (MEMORY BANK B):
Overlap Detected: {overlap_detected}

---
### INSTRUCTIONS:
1. If 'Overlap Detected' is True, write a concise warning at the top suggesting a fresh angle.
2. If the user input contains explicit preferences or rules (e.g., "don't use emojis", "keep sentences short"), extract them into 'extracted_rules'.
3. Draft the social media post adhering strictly to stored brand guidelines and new instructions.

{format_instructions}
"""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "{user_input}")
])

chain = prompt_template | llm | parser


# =====================================================================
# 4. AGENT EXECUTION FUNCTION
# =====================================================================
def run_brandcraft_agent(user_input: str, topic_summary: str, memory: HindsightMemory) -> AgentPostResponse:
    # 1. RECALL Step: Check for topic overlap
    has_overlap = memory.check_topic_overlap(topic_summary)
    
    # 2. RECALL Step: Fetch active style rules
    active_rules = memory.recall_style_guidelines()
    
    # 3. GENERATE Step: Execute LangChain pipeline
    response: AgentPostResponse = chain.invoke({
        "recalled_style_rules": active_rules,
        "overlap_detected": has_overlap,
        "user_input": user_input,
        "format_instructions": parser.get_format_instructions()
    })
    
    # 4. RETAIN Step: Update memory banks with extracted rules & topic
    for rule in response.extracted_rules:
        memory.retain_rule(rule)
        
    memory.retain_content(topic_summary)
    
    return response


# =====================================================================
# 5. DEMO SIMULATION (FOR HACKATHON PRESENTATION)
# =====================================================================
if __name__ == "__main__":
    memory_engine = HindsightMemory()

    print("==========================================")
    print("--- DEMO INTERACTION 1: Setting Guidelines ---")
    print("==========================================")
    input_1 = "Draft a LinkedIn post announcing our new AI analytics feature. Make it less formal, use shorter sentences, and don't use emojis."
    topic_1 = "AI Analytics Feature"

    res1 = run_brandcraft_agent(input_1, topic_1, memory_engine)
    
    if res1.warning_message:
        print(f"⚠️ {res1.warning_message}\n")
    print(f"Generated Post:\n{res1.post_draft}\n")
    print(f"Hindsight Learned Rules: {memory_engine.bank_a_style_rules}\n")

    print("==========================================")
    print("--- DEMO INTERACTION 2: Memory Recall & Overlap ---")
    print("==========================================")
    input_2 = "Write another post about our AI analytics feature for tomorrow."
    topic_2 = "AI Analytics Feature"

    res2 = run_brandcraft_agent(input_2, topic_2, memory_engine)
    
    if res2.warning_message:
        print(f"⚠️ {res2.warning_message}\n")
    print(f"Generated Post:\n{res2.post_draft}")
