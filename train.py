from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, WavLMModel,Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
from transformers import TrainingArguments
from transformers import Trainer

processor = Wav2Vec2Processor.from_pretrained('microsoft/wavlm-large')
model = Wav2Vec2ForSequenceClassification.from_pretrained('microsoft/wavlm-large')
train_dataset = prepare_dataset(your_train_data)
eval_dataset = prepare_dataset(your_eval_data)

training_args = TrainingArguments(
    output_dir="./models/anti_spoofing",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir='./logs',
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # training dataset
    eval_dataset=eval_dataset,    # evaluation dataset
)

trainer.train()
trainer.evaluate()
model.save_pretrained("./models/wavlm-finetuned")
processor.save_pretrained("./models/wavlm-finetuned")


