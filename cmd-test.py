from argparse import ArgumentParser
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from opendelta import LoraModel
import torch

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--use-bminf", default=False, action="store_true", help="Whether to use BMInf")
    parser.add_argument("--memory-limit", type=int, default=20, help="GPU Memory limit, in GB")
    parser.add_argument("--delta", default=None, type=str, help="The path to lora.")
    parser.add_argument("--device", default="cuda:0", type=str, help="The target device.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = CPMBeeConfig.from_json_file("config/cpm-bee-10b.json")
    ckpt_path = "/mnt/xusheng/modelbest/cpm-bee-10b/pytorch_model.bin"
    tokenizer = CPMBeeTokenizer()
    model = CPMBeeTorch(config=config)

    if args.delta is not None:
        delta_model = LoraModel(backbone_model=model, modified_modules=["project_q", "project_v"], backend="hf")
        model.load_state_dict(torch.load(args.delta), strict=False)

    model.load_state_dict(torch.load(ckpt_path), strict=False)

    if args.device == "cpu":
        model = model.float()
    else:
        if not torch.cuda.is_available():
            raise AssertionError("The CUDA is unavailable")
        if args.use_bminf:
            import bminf
            with torch.cuda.device(args.device):
                model = bminf.wrapper(model, quantization=False, memory_limit=args.memory_limit << 30)
        model.half().cuda(args.device)

    # use beam search
    beam_search = CPMBeeBeamSearch(
        model=model,
        tokenizer=tokenizer,
    )
    inference_results = beam_search.generate(data_list, max_length=100, repetition_penalty=1.1)
    for res in inference_results:
        print(res)

    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        input = [{"question": query, "<ans>": ""}]
        for response in beam_search.generate(input, max_length=200, repetition_penalty=1.1):
            print(response["<ans>"], flush=True)
        

if __name__ == "__main__":
    main()
