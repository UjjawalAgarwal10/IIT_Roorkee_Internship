from dataclasses import dataclass
class AgentConfig:
    model_name: str="ollama/llama3.2:3b"
    ollama_base_url: str="http://localhost:11434"
    temperature: float=0.2
    max_iter: int=5
    verbose: bool=True 