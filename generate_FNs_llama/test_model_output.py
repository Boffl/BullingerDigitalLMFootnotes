from run_llama_over_prompts import *
import logging


def test_single_adapters():
        # example file
    filepath = "/data/nbauer/data/prompts/instruct_add_window/dev_100/13134_23.jsonl"
    with jsonlines.open(filepath) as infile:
        messages = [line for line in infile]
    size = 70
    adapters = ["add", "qa", "EA", "bible"]
    model_id = f"unsloth/Meta-Llama-3.1-{size}B-Instruct-bnb-4bit"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    for adapter in adapters:
        adapter_id = make_adapter_id(adapter, size)
        model = PeftModel.from_pretrained(model, adapter_id, adapter_name=adapters)
        model.set_adapter(adapter)

        generated_text = generate_chat(messages, model, tokenizer)
        logging.info(f"Output with {model.active_adapter} adapter:")
        logging.info(generated_text)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_function", choices=["single_adapters"])
    parser.add_argument("--log_file_name", default="")
    args = parser.parse_args()
    if args.log_file_name == "":
        log_file_name =f"{args.test_function}_test.log"
    else:
        log_file_name = args.log_file_name
    # Configure logging
    logging.basicConfig(
        filename=log_file_name,
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if args.test_function == "single_adapters":
        test_single_adapters()

