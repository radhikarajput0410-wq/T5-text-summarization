import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize(text):
    inputs = tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=30,
        num_beams=4
    )

    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=8, label="Input Text"),
    outputs=gr.Textbox(label="Summary"),
    title="T5 Text Summarization"
)

demo.launch()