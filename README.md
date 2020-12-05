# issuesFile
存放提问中临时图片，文件的项目

# 代码片段
```python

trigger_model_path = "output-model-trigger"
arguments_model_path = "output-model-arguments"
sentence_max_length = 200

tokenizer = BertTokenizer.from_pretrained(trigger_model_path)
trigger_model = BertCCRFForTokenClassification.from_pretrained(trigger_model_path)
arguments_model = BertCCRFForTokenClassification.from_pretrained(arguments_model_path)

trigger_model.to("cuda:0")
trigger_model.eval()
arguments_model.to("cuda:0")
arguments_model.eval()
trigger_label_map = trigger_model.config.id2label
arguments_label_map = arguments_model.config.id2label


def get_model_input(sentence):
    token = [101] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))[:sentence_max_length - 2] + [102]
    input_ids = np.array([token])
    input_mask = np.array([[1] * len(token)])
    # segment_ids = np.array([[0] * len(token)])
    words = tokenizer.convert_ids_to_tokens(input_ids[0])
    input_ids = torch.from_numpy(input_ids)
    # segment_ids = torch.from_numpy(segment_ids)
    input_mask = torch.from_numpy(input_mask)
    batch = [input_ids, input_mask]
    inputs = {"input_ids": batch[0].cuda(), "attention_mask": batch[1].cuda()}

    return inputs, words


def get_trigger_model_pred(inputs, words, return_pred=False):
    logits = trigger_model(**inputs)[0]
    preds = logits.detach().cpu().numpy()
    preds = np.argmax(preds, axis=2)[0]
    if return_pred:
        return [trigger_label_map[_] for _ in preds]

    tmp_preds, event_type = [], []
    for i, _ in enumerate(preds):
        if trigger_label_map[_] == "O":
            if tmp_preds:
                event_type.append([tmp_preds[0][0].replace("B-", "").replace("I-", ""),
                                   "".join([_[1] for _ in tmp_preds])])
            tmp_preds = []
            continue
        if "B" in trigger_label_map[_] or "I" in trigger_label_map[_]:
            tmp_preds.append([trigger_label_map[_], words[i]])

    return event_type
```
