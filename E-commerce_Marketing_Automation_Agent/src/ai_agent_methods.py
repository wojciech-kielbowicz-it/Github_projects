from huggingface_hub import InferenceClient
import pandas as pd
import time

def generate_email(row: pd.Series, client: InferenceClient, prompt: dict) -> str:
    segment = row["segmentation"]
    instruction = prompt[segment]
    try:
        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta:featherless-ai",
            messages=[
                {
                    "role": "user",
                    "content": f"{instruction} Customer ID: {row['customer_id']}"
                }
            ],
            max_tokens=150, 
            temperature=0.7

        )
        message = completion.choices[0].message.content
        time.sleep(1)
        return message
    except Exception as e:
        print(f"Error:\n{e}")
        return "ERROR"
        