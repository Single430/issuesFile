# issuesFile
存放提问中临时图片，文件的项目

# 代码片段
* MiniCPM 仿openai接口
```python
python minicpm_openai_api.py
```

* Bert 推理
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

* TensorRT

```python

def squad_output(prefix, config, init_dict, network, input_tensor):
    """
    Create the squad output
    """

    idims = input_tensor.shape
    assert len(idims) == 5
    B, S, hidden_size, _, _ = idims

    W_out = init_dict[prefix + SQD_W]
    B_out = init_dict[prefix + SQD_B]

    W = network.add_constant((1, hidden_size, 2), W_out)
    dense = network.add_fully_connected(input_tensor, 2, W_out, B_out)
    set_layer_name(dense, prefix, "dense")
    return dense


def multi_class_output(prefix, config, init_dict, network, input_tensor):
    """
    Create the squad output
    """
    labels_num = 12
    idims = input_tensor.shape
    assert len(idims) == 5
    B, S, hidden_size, _, _ = idims

    p_w = init_dict["bert_pooler_dense_kernel"]
    p_b = init_dict["bert_pooler_dense_bias"]
    W_out = init_dict["output_weights"]
    B_out = init_dict["output_bias"]
    # 这里其实可以直接取[CLS]的向量进行后续运算，但是没能实现相关功能，就计算了所有的
    # reshape_ = network.add_slice(input_tensor, [0, 0, 0, 0, 0], [1, 1, 768, 1, 1], [1, 1, 1, 1, 1])
    pool_output = network.add_fully_connected(input_tensor, hidden_size, p_w, p_b)
    pool_data = pool_output.get_output(0)
    tanh = network.add_activation(pool_data, trt.tensorrt.ActivationType.TANH)
    tanh_output = tanh.get_output(0)

    dense = network.add_fully_connected(tanh_output, labels_num, W_out, B_out)
    set_layer_name(dense, prefix, "dense")
    return dense


def ner_output(prefix, config, init_dict, network, input_tensor):
    """
    Create the squad output
    """
    labels_num = 12
    idims = input_tensor.shape
    assert len(idims) == 5
    B, S, hidden_size, _, _ = idims

    W_out = init_dict["project_logits_w"]
    B_out = init_dict["project_logits_b"]
    # 转置在拉平
    W_out = W_out.numpy().reshape((768, labels_num)).transpose((1, 0)).reshape((768*labels_num))
    # add_fully_connected
    # Y:=matmul(X,WT)+bias

    # W = network.add_constant((1, hidden_size, labels_num), W_out)
    pool_output = network.add_fully_connected(input_tensor, labels_num, W_out, B_out)
    pool_data = pool_output.get_output(0)
    dense = network.add_activation(pool_data, trt.tensorrt.ActivationType.TANH)
    set_layer_name(dense, prefix, "dense")

    # 9216 12 (-1, 512, 768, 1, 1) (-1, 512, 12, 1, 1)
    # print(W_out.size, B_out.size, input_tensor.shape, dense.get_output(0).shape)
    # exit()
    return dense
```

