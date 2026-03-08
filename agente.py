from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
import numexpr as ne
import re # Biblioteca de Expressões Regulares do Python

print("Iniciando o carregamento do modelo local (Qwen 0.5B)...")
print("Isso pode levar um minuto na primeira vez.")

pipe = pipeline(
    "text-generation", 
    model="Qwen/Qwen2.5-0.5B-Instruct", 
    max_new_tokens=45,
    max_length=None,
    temperature=0.01,
    return_full_text=False
)
llm = HuggingFacePipeline(pipeline=pipe)

def calculadora(expressao: str) -> str:
    """Avalia uma expressão matemática exata com blindagem contra alucinações da IA."""
    try:
        # 1. Corta qualquer texto que venha depois de uma quebra de linha ou palavra indesejada
        exp_cortada = expressao.split("Observation")[0].split("Thought")[0].split("\n")[0]
        
        # 2. Regex: Mantém APENAS números (0-9) e operadores matemáticos (+, -, *, /, (, ), .)
        # Remove letras, aspas, chaves e sujeiras como 'expressao ='
        exp_limpa = re.sub(r'[^\d\+\-\*\/\(\)\.]', '', exp_cortada)
        
        if not exp_limpa:
            return "Erro: Nenhuma conta matemática encontrada na entrada."
            
        # 3. Faz o cálculo com a expressão sanitizada
        resultado = ne.evaluate(exp_limpa).item()
        
        # Limpa o visual de números terminados em .0 (ex: 100.0 vira 100)
        if isinstance(resultado, float) and resultado.is_integer():
            return str(int(resultado))
            
        return str(round(resultado, 2))
        
    except Exception as e:
        return f"Erro ao calcular a expressão limpa '{exp_limpa}': {e}"

ferramenta_calc = Tool(
    name="Calculator",
    func=calculadora,
    description="Useful for math. Input must be JUST the math expression, nothing else. Example: 450 + 150"
)

ferramentas = [ferramenta_calc]

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
    handle_parsing_errors=True,
    max_iterations=3
)

if __name__ == "__main__":
    print("\n--- Agente LangChain Pronto ---")
    print("Digite 'sair' para encerrar o programa.\n")
    
    while True:
        comando = input("Você (Ex: Calculate 15 * 1.5): ")
        
        if comando.lower() in ['sair', 'exit', 'quit', 'parar']:
            print("\nEncerrando o agente. Boa apresentação!")
            break
            
        try:
            resposta = executor_agente.invoke({"input": comando})
            print("\n=== RESPOSTA FINAL ===")
            print(resposta['output'])
            print("-" * 40 + "\n")
        except Exception as e:
            print(f"\nOcorreu um erro: {e}\n")