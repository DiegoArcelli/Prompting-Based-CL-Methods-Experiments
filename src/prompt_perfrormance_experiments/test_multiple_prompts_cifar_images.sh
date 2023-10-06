for ref_class in {0..100}
do
    echo "Executing for class $ref_class"
    python3.9 test_multiple_prompts_cifar_images.py $ref_class > output_prompt/output_multiple_prompts_cifar_$ref_class
done