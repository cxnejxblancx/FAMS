"""
1. Open in integrated terminal
2. Initialize virtual environment
4. Activate virtual environment
5. Install required packages: pip install -r requirements.txt
6. HuggingFace login and api key needed: https://huggingface.co/, use huggingface-cli login

Research:
    https://www.datacamp.com/tutorial/text-analytics-beginners-nltk
    https://www.datacamp.com/tutorial/fine-tuning-llama-3-1
    https://levity.ai/blog/sentiment-classification
    https://www.kaggle.com/code/lucamassaron/fine-tune-llama-3-for-sentiment-analysis
    https://docs.anyscale.com/llms/finetuning/guides/lora_vs_full_param/#:~:text=Fine%2Dtune%20LoRA%20with%20a,all%20possible%20layers%20with%20LoRA.
    https://research.aimultiple.com/sentiment-analysis-dataset/
    https://www.analyticsvidhya.com/blog/2022/07/sentiment-analysis-using-python/
    https://www.youtube.com/watch?v=QpzMWQvxXWk
    

Updates:
    - Code drafted for sentiment analysis of student evaluations using LLaMA 3.1 8B Instruct model
    - More work needed for data preprocessing based on csv file directly from CATME
    - Code still needs to be tested and improved to understand current flow and necessary modifications
    - Error logging needed
"""

# Import
import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer, setup_chat_format
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging, EarlyStoppingCallback, IntervalStrategy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = '0'        # tells PyTorch to use first GPU available
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tells Transformers not to parallelize tokenization process
warnings.filterwarnings("ignore")               # all warnings will be ignored

print(f"pytorch version: {torch.__version__}")

# Determine GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"working on {device}")

# Disable PyTorch speed/memory optimization features for scaled dot product attention (SDPA) function
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# Load dataset from CSV file with sentiment labels for each row of text
#filename = "test data for bert - Sheet1.csv"
# df = pd.read_csv(filename, names=["text", "Contributing to the Team's Work", "Interacting with Teammates", "Keeping the Team on Track", "Expecting Quality", "Having Relevant Knowledge, Skills, and Abilities"], encoding="utf-8", encoding_errors="replace")
filename = "Sheet1.csv"
df = pd.read_csv(
    filename,
    names= ["Text", "Sentiment"],
    encoding="utf-8",
    encoding_errors="replace"
)

# Split data based on sentiments
X_train = []
X_test = []
for sentiment in ["Positive", "Neutral", "Negative"]:
    sentiment_df = df[df["Sentiment"] == sentiment] # filter dataframe by sentiment
    train, test = train_test_split(
        sentiment_df,
        train_size = 150,   # 150/200 samples in training set
        test_size = 50,     # 50/200 samples in test set
        random_state = 42   # ensure each split contains positive, negative, and neutral sentiments
    )
    X_train.append(train)
    X_test.append(test)

# Combine training and test sets, reset index to allow consistent indexing and prevent duplicate indices, discard old indices
X_train = pd.concat(X_train).sample(frac=1, random_state=10).reset_index(drop=True)
X_test = pd.concat(X_test).reset_index(drop=True)

# Prepare evaluation dataset
eval_idx = [i for i in df.index if i not in X_train.index and i not in X_test.index]
X_eval = df.iloc[eval_idx]
X_eval = X_eval.groupby("Sentiment", group_keys=False).apply(lambda x: x.sample(n=50, random_state=42)).reset_index(drop=True)
X_train = X_train.reset_index(drop=True)

# Generate prompt
def generate_prompt(data_point):
    prompt = f"Analyze the sentiment of the evaluations enclosed in square brackets. Determine if it is positive, neutral, or negative, and return the answer as the corresponding sentiment label 'Positive', 'Negative', or 'Neutral':\n[{data_point["Text"]}] = "
    
    return prompt.strip()

# Generate test prompt (delete later)
def generate_test_prompt(data_point):
    return generate_prompt(data_point)

# Format data
X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1), columns=["Text"])
X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1), columns=["Text"])
y_true = X_test.Sentiment
X_text = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["Text"])

# Convert to HuggingFace Dataset objects
train_data = Dataset.from_pandas(X_train)
eval_data = Dataset.from_pandas(X_eval)
    

# Evaluate predictions from fine-tuned model using metrics and confusion matrix
def evaluate(y_true, y_pred):
    labels = ["Positive", "Neutral", "Negative"]
    mapping = {
        "Positive" : 2,
        "Neutral": 1,
        "Negative" : 0
    }
    
    y_true = np.vectorize(mapping.get)(y_true)
    y_pred = np.vectorize(mapping.get)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f"Accuracy: {accuracy:.3f}")

    # Generate accuracy report
    unique_labels = set(y_true) # get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true)) if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f"Accuracy: {accuracy:.3f}")

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)
    print("\nClassification Report:")
    print(class_report)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0,1,2])
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # return accuracy, class_report, conf_matrix

model_name = "meta-llama/Meta-Llama-3.1-8B-instruct"
compute_dtype = getattr(torch, "float16") # float16 data type will be used for all computations

# Load model in 4bit precision (Quantize/Compress model for QLoRA and resource efficiency)
bnb_config = BitsAndBytesConfig( 
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bt_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=compute_dtype,
    quantization_config=bnb_config
)
model.config.use_cache = False
model.config.pretraining_tp = 1

max_seq_length = 512 # 2048 bytes

# Instantiate tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, max_seq_length=max_seq_length)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Predict sentiment of student evaluation
def predict(test, model, tokenizer):
    pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens = 1,
            temperature = 0.0   # produce very predictable text
        )
    
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["Text"]
        result = pipe(prompt)
        answer = result[0]["generated_text"].split('=')[-1].strip()

        if "Positive" in answer:
            y_pred.append("Positive")
        elif "Neutral" in answer:
            y_pred.append("Neutral")
        elif "Negative" in answer:
            y_pred.append("Negative")
        else:
            y_pred.append("Neutral") # default to neutral if no clear answer
    return y_pred


#
# Fine-tuning
#
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)
    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1} 

output_dir = "trained_weights" # temporary location during training --> review later

peft_config = LoraConfig( 
    lora_alpha=16,
    lora_dropout=0,
    r=64,
    bias="None",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

training_arguments = TrainingArguments(
    output_dir=output_dir,              # directory to save and repo id
    num_train_epochs=5,                 # num of training epochs
    per_device_tain_batch_size=1,       # batch size per device during training
    gradient_accumulation_steps=8,      # num of steps before performing a backward/update pass
    gradient_checkpointing=True,        # use gradient checkpoint to save memory
    optim="paged_adamw_32bit",          # use optimizer to adjust model weight during training (good for mem. optim.)
    save_steps=0,
    logging_steps=25,                   # log every 10 steps
    learning_rate=1e-4,                 # learning rate --> standard for QLoRA (Quantized Low-Rank Adaptation)
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,                  # max gradient norm for QLoRA
    max_steps=-1,
    warmup_ratio=0.03,                 # standard warmup ration for QLoRA
    group_by_length=False,
    lr_sceduler_type="cosine",         # cosine learning rate scheduler
    report_to="tensorboard",           # report metrics to tensorboard for visualization
    # evaluation_strategy=IntervalStrategy.STEPS,
    # eval_steps= 100,                   # Evaluate after every 100 steps
    # load_best_model_at_end=True,       # Load the best model when finished
    # metric_for_best_model="accuracy",  # Track accuracy as the best model
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] # Stop if no improvement for 3 evals


)
# Fine-tune model using Supervised Finetuning Trainer (SFT) to format dataset
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train.data,
    peft_config=peft_config, # using Parameter Efficient Fine-tuning (PEFT) method to srefine model parameters while keeping most fixed
    dataset_text_field="Text",
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    packing=False,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False
    }
)

# Main training function
if __name___ == "__main__":
    # Train model
    trainer.train()

    # Save trained model and tokenizer locally
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate model on test set
    y_pred = predict(test, model, tokenizer)
    evaluate(y_true, y_pred)

    # Save test predictions
    evaluation = pd.DataFrame({"Text": X_test["Text"], "y_true": y_true,"y_pred": y_pred})
    evaluation.to_csv("test_prediction.csv", index=False)
