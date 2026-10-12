import sys

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():

    model_path = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print(f"Loaded model from {model_path}")

    questions = [
        "Identify your animal counterpart in exactly one word.",
		"Which animal would you want as your mascot? One word only.",
		"Tell me your favorite wild animal in a single word.",
    ]

    for question in questions:
        inputs = tokenizer(question, return_tensors="pt")
        outputs = model.generate(**inputs, do_sample=True, max_new_tokens=32, num_return_sequences=3)
        for i, output in enumerate(outputs):
            answer = tokenizer.decode(output, skip_special_tokens=True)
            print(f"Q: {question}\nA[{i}]: {answer}\n")


if __name__ == "__main__":
    main()