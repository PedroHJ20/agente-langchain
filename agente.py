from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
import numexpr as ne

print("Iniciando o carregamento do modelo local (Qwen 0.5B)...")

# Modelo um pouco maior e mais inteligente, mas que ainda roda bem no Codespace
pipe = pipeline(
    "text-generation", 
    model="Qwen/Qwen2.5-0.5B-Instruct", 
    max_new_tokens=200,
    temperature=0.1
)
llm = HuggingFacePipeline(pipeline=pipe)

def calculadora(expressao: str) -> str:
    """Avalia uma expressão matemática exata."""
    try:
        resultado = ne.evaluate(expressao).item()
        return str(resultado)
    except Exception as e:
        return f"Erro ao calcular: {e}"

# Nome e descrição da ferramenta em inglês para o modelo entender melhor
ferramenta_calc = Tool(
    name="Calculator",
    func=calculadora,
    description="Useful for when you need to answer questions about math. Input should be ONLY the math expression (e.g. 15 * 1.5)."
)

ferramentas = [ferramenta_calc]

# PROMPT EM INGLÊS: Obrigatório para o parser nativo do LangChain funcionar
template_react = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format strictly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template_react)

agente = create_react_agent(llm, ferramentas, prompt)
executor_agente = AgentExecutor(
    agent=agente, 
    tools=ferramentas, 
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == "__main__":
    print("\n--- Agente LangChain Pronto ---")
    # Sugestão: Faça a pergunta em inglês para facilitar a vida do modelo pequeno
    comando = input("Digite seu comando (Ex: Calculate 15 * 1.5): ")
    resposta = executor_agente.invoke({"input": comando})
    print("\n=== RESPOSTA FINAL ===")
    print(resposta['output'])