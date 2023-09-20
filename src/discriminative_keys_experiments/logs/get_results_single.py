import subprocess
import re

result = subprocess.getoutput("cat output_single_prompt_cifar | grep Results -A 36")
results = result.split("Results")

result_dict = {}

# iterate over considered class
for c, res in enumerate(results[1:]):
    result_dict[c] = {}
    # print(res)
    prompts = re.split("prompt", res, flags=re.IGNORECASE)
    # iterate over prompts
    for p, prompt in enumerate(prompts[1:]):

        key = p
        if p == 0:
            key = "no_prompt"
        elif p == 11:
            key = "l2p_prompt"
        else:
            key = p-1
        
        result_dict[c][key] = {}
        classes = prompt.split("Class ")
        for _class in classes[1:]:
            class_num = re.split('\n|,', _class)[0]
            pred_class, count = class_num.split(": ")
            pred_class, count = int(pred_class), int(count)
            result_dict[c][key][pred_class] = count

# print(result_dict[0])
# print(result_dict[1])

for ref_class in range(0, 100):
    print(f"Images of class {ref_class}")
    for prompt in ["no_prompt"] + list(range(0, 10)) + ["l2p_prompt"]:
        tot = 0
        for pred_class in result_dict[ref_class][prompt].keys():
            tot += result_dict[ref_class][prompt][pred_class]
        if ref_class in result_dict[ref_class][prompt].keys():
            acc = result_dict[ref_class][prompt][ref_class]/tot
        else:
            acc = 0
        acc = round(acc, 4)
        if prompt == "no_prompt":
            print(f"No prompt accuracy: {acc}")
        elif prompt == "l2p_prompt":
            print(f"L2P prompts accuracy: {acc}")
        else:
            print(f"Prompt {prompt} accuracy: {acc}")
    print("\n")