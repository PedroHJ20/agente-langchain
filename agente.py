from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.prompts import PromptTemplate
import numexpr as ne

print("Iniciando o carregamento do modelo local...")

# Modelo ultra-leve escolhido para rodar no Codespace sem travar a memória
pipe = pipeline(
    "text-generation", 
    model="HuggingFaceTB/SmolLM-135M-Instruct", 
    max_new_tokens=150,
    temperature=0.1
)
llm = HuggingFacePipeline(pipeline=pipe)

def calculadora(expressao: str) -> str:
    """Avalia uma expressão matemática exata."""
    try:
        resultado = ne.evaluate(expressao).item()
        return str(resultado)
    except Exception as e:
        return f"Erro: {e}"

ferramenta_calc = Tool(
    name="Calculadora",
    func=calculadora,
    description="Use apenas para cálculos matemáticos. A entrada deve ser a expressão exata (ex: 15 * 1.5)."
)

ferramentas = [ferramenta_calc]

template_react = """Responda a pergunta. Você tem as seguintes ferramentas:

{tools}

Use o formato exato:
Pergunta: a pergunta a responder
Pensamento: o que fazer
Ação: [{tool_names}]
Entrada da Ação: os números para calcular
Observação: o resultado
Pensamento: já sei a resposta
Resposta Final: a resposta final

Comece!

Pergunta: {input}
Pensamento:{agent_scratchpad}"""

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
    comando = input("Digite seu comando (Ex: Calcule 15 * 1.5): ")
    resposta = executor_agente.invoke({"input": comando})
    print("\n=== RESPOSTA FINAL ===")
    print(resposta['output'])