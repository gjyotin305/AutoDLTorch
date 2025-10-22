from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8002/v1",
    api_key="EMPTY"
)

def get_response(sys_prompt, prompt, model_name='gjyotin305/qwen2.5-check-litenv-merged'):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=100
    )
    
    result = response.choices[0].message.content
    return result

if __name__ == '__main__':
    sys_prompt = """You are an instruction following agent."""
    prompt = """## Instruction: Describe the life and reign of King Charles II."""
    print(get_response(sys_prompt, prompt))