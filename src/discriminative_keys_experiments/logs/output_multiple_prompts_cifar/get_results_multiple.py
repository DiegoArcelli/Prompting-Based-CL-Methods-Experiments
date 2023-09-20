import subprocess
import re

for c in range(0, 100):
    result = subprocess.getoutput(f"cat output_multiple_prompts_cifar_{c} | grep Results")
    class_label = result.split("class ")[-1][:-1]
    class_label = int(class_label)

    result_dict = {}
    result = subprocess.getoutput(f"cat output_multiple_prompts_cifar_{c} | grep Prompt -A 1")
    prompts = result.split("Prompt")[1:]
    for prompt in prompts:
        start = prompt.rfind("(")
        end = prompt.rfind(")")+1
        text_prompt = prompt[start:end]
        result_dict[text_prompt] = {}
        classes = prompt.split("Class ")[1:]
        for _class in classes:
            class_num = re.split('\n|,', _class)[0]
            pred_class, count = class_num.split(": ")
            pred_class, count = int(pred_class), int(count)
            result_dict[text_prompt][pred_class] = count
            # print(pred_class, count)

    file_text = f"Results for class {class_label}:\n"
    for prompt, preds in result_dict.items():
        tot = sum(preds.values())
        if class_label in preds.keys():
            acc = result_dict[prompt][class_label]/tot
        else:
            acc = 0
        file_text += f"Prompt {prompt} accuracy {acc}\n"
    with open(f"output_multiple_prompts_cifar_{c}_acc", "w") as f:
        f.write(file_text)