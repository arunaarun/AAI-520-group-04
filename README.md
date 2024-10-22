### README
---

#### **1. Introduction**
This project focuses on fine-tuning the **DialoGPT-medium** model using the **Cornell Movie-Dialogs Corpus**. The aim was to enhance the model’s conversational capabilities by generating coherent and contextually relevant dialogue, particularly in a movie-based context.

---

#### **2. Requirements**

To replicate the fine-tuning process, the following libraries and tools are required:

- **Python 3.7+**
- **Pandas**
- **NumPy**
- **NLTK (Natural Language Toolkit)** 
- **Matplotlib**
- **Seaborn**
- **PyTorch**
- **Huggingface Transformers Library**

Additionally, the following NLTK corpora should be downloaded:
- `averaged_perceptron_tagger`
- `maxent_ne_chunker`
- `punkt`
- `stopwords`

**Installation:**
Install the necessary libraries using the `requirements.txt` or manually:
```bash
pip install pandas numpy matplotlib seaborn nltk torch transformers
```

---

#### **3. Data Source**
The dataset used for fine-tuning is the **Cornell Movie-Dialogs Corpus**, which includes:
- Movie titles metadata
- Character metadata
- Movie lines and dialogue interactions between characters
- Conversations and dialogues from popular movies

The project specifically worked with **50,000 dialogue pairs** to fine-tune the model.

---

#### **4. Data Processing**
Key steps in data preprocessing included:

- **Loading and cleaning data**: Metadata and dialogues were cleaned, including handling non-standard entries in fields like year and genres.
- **Tokenization**: Text was tokenized and lower-cased, irrelevant words were removed, and words were stemmed for better learning.

---

#### **5. Model Training**
**Model and Tokenizer**: The model used was **DialoGPT-medium** from Huggingface, fine-tuned on the Cornell Movie Dialogues. The tokenizer was initialized using the same model checkpoint.

**Training Parameters**:
- **Learning Rate**: 2e-5
- **Epochs**: 5
- **Batch Size**: 8
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup and decay

The model was trained on a **CUDA GPU** for faster processing. The training involved **dynamic padding** for optimized tensor operations.

---

#### **6. Model Evaluation**
**Evaluation Metrics**:
- **BLEU Score**: To measure how similar the generated dialogues were to human-written responses.
- **Perplexity**: To assess the fluency and confidence of the generated responses.

---

#### **7. Results**
- **Average BLEU Score**: 0.2635 (indicating moderate accuracy in generating relevant dialogue)
- **Average Perplexity**: 9.2560 (indicating the model is confident in generating fluent dialogue)
- **Decreasing Loss**: The model’s loss decreased across epochs, showcasing the model's learning progress:
  - **Epoch 1**: 2.9671
  - **Epoch 2**: 2.7111
  - **Epoch 3**: 2.5907
  - **Epoch 4**: 2.5063
  - **Epoch 5**: 2.4535

---

#### **8. Testing**
The model was tested using prompts related to the movie **"10 Things I Hate About You"**, evaluating its ability to generate context-aware dialogue. Examples included:

- **Prompt**: "Why does Cameron suggest that Bianca needs to learn how to lie?"
  - **Generated Response**: "Because she was lying when she said she couldn’t get married."
  - **Reference Answer**: "Cameron suggests Bianca needs to learn how to lie because she’s too honest."

---

#### **9. Future Improvements**
While the model shows strong fluency and coherence in responses, improvements in the **BLEU score** could be achieved by:
- Expanding the dataset for more diverse conversational patterns.
- Further tuning the learning rate and optimizer for better accuracy in responses.

---

#### **10. How to Use the Model**
1. **Model and Tokenizer Loading**:
    ```python
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("./fine_tuned_dialogpt_medium")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_dialogpt_medium")
    ```

2. **Generating Responses**:
    ```python
    prompt = "Why does Cameron suggest Bianca learn how to lie?"
    input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt").to(device)
    response = model.generate(input_ids)
    decoded_response = tokenizer.decode(response[0], skip_special_tokens=True)
    print(decoded_response)
    ```

---
