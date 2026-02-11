from huggingface_hub import InferenceClient
import pandas as pd
import time

def generate_email(row: pd.Series, client: InferenceClient, prompt: dict[str, str]) -> str:
    segment: str = row["segmentation"]

    instruction: str = prompt[segment]

    formatted_prompt: str = f"""
    ROLE: You are an expert Email Marketer for the store 'Wojciech Kiełbowicz & Co'.
    GOAL: Write a warm, professional email based on this idea: "{instruction}"
    
    STRICT WRITING RULES:
    1. OPENING: Start the email EXACTLY with: "Dear Customer {row['customer_id']},"
    2. BODY: Write 2-3 short paragraphs. Use double new lines (\\n\\n) between paragraphs to ensure readability.
    3. ENDING: Sign off EXACTLY as:
       "Best regards,
       The Wojciech Kiełbowicz & Co Team"
    4. PROHIBITED: Do NOT use placeholders like [Name], [Date], or [ID]. Do NOT use square brackets text at all.
    """

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b:groq",
            messages=[
                {
                    "role": "user",
                    "content": formatted_prompt
                }
            ],
            temperature=0.7
        )
        email: str = completion.choices[0].message.content
        time.sleep(1)
        return email
    except Exception as e:
        print(f"Error:\n{e}")
        return "ERROR"
        