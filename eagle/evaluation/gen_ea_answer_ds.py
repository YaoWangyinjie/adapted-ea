"""Generate EAGLE answers and evaluator-side adaptation statistics."""
import argparse
import json
import math
import os
import tempfile
import time

import numpy as np
import shortuuid
from accelerate.utils import set_seed
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
set_seed(0)

try:
    from ..model.ea_model_1 import EaModel
    from ..model.utils import *
except ImportError:
    from eagle.model.ea_model_1 import EaModel
    from eagle.model.utils import *


def _normalize_path(path):
    return os.path.abspath(os.path.expanduser(path))


def _output_paths(answer_file):
    answer_file = _normalize_path(answer_file)
    stem, _ = os.path.splitext(answer_file)
    return answer_file, stem + "_stat.txt", stem + "_stats.jsonl"


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return _json_safe(value.item())
        return _json_safe(value.tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value

def _atomic_write_jsonl(path, rows):
    directory = os.path.dirname(path) or os.getcwd()
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w") as fout:
            for row in rows:
                fout.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _balanced_chunks(items, count):
    if not items:
        return []
    count = min(count, len(items))
    base, extra = divmod(len(items), count)
    chunks = []
    start = 0
    for index in range(count):
        size = base + (index < extra)
        chunks.append(items[start:start + size])
        start += size
    return chunks


def run_eval(base_model_path, ea_model_path, model_id, question_file,
             question_begin, question_end, answer_file, max_new_token,
             num_choices, num_gpus_per_model, num_gpus_total,
             max_gpu_memory, temperature, args):
    questions = load_questions(question_file, question_begin, question_end)
    assert num_gpus_total % num_gpus_per_model == 0
    worker_count = num_gpus_total // num_gpus_per_model
    use_ray = worker_count > 1
    chunks = _balanced_chunks(questions, worker_count)
    answer_file, stat_file, machine_stat_file = _output_paths(answer_file)

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(get_model_answers).remote
    else:
        get_answers_func = get_model_answers

    handles = [get_answers_func(
        base_model_path, ea_model_path, model_id, chunk, answer_file,
        max_new_token, num_choices, num_gpus_per_model, max_gpu_memory,
        temperature, args) for chunk in chunks]
    results = ray.get(handles) if use_ray and handles else handles
    answer_records = []
    all_stats = []
    for records, stats in results:
        answer_records.extend(records)
        all_stats.extend(stats)
    answer_records.sort(key=lambda row: str(row["question_id"]))
    all_stats.sort(key=lambda row: (str(row["qid"]), row["choice"], row["turn"]))
    _atomic_write_jsonl(answer_file, answer_records)
    _atomic_write_jsonl(machine_stat_file, all_stats)
    save_statistics(all_stats, stat_file)


def _effective_adaptation_value(model, name, requested):
    value = getattr(model, name, requested)
    return value.item() if hasattr(value, "item") else value


def _decode_output(tokenizer, output_ids, input_length):
    output_ids = output_ids[0][input_length:]
    stop_ids = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")}
    stop_indices = [i for i, token_id in enumerate(output_ids) if token_id in stop_ids]
    if stop_indices:
        output_ids = output_ids[:stop_indices[0]]
    output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)
    for special_token in tokenizer.special_tokens_map.values():
        tokens = special_token if isinstance(special_token, list) else [special_token]
        for token in tokens:
            output = output.replace(token, "")
    return output.strip()


def _adaptation_details(run_stats, enabled):
    diagnostics = run_stats.get("diagnostics") or {}
    adaptation = run_stats.get("adaptation") or diagnostics.get("adaptation") or {}
    if not enabled:
        status = "disabled"
        reason = "online_adaptation_disabled"
    else:
        status = adaptation.get("status", "skipped")
        reason = adaptation.get("skip_reason")
        if status not in {"updated", "skipped", "rolled_back"}:
            status = "skipped"
            reason = reason or "invalid_adaptation_status"
        elif status != "updated":
            reason = reason or "not_updated"
    counts = dict(adaptation.get("counts") or {})
    return status, reason, counts, {
        "adaptation": adaptation,
        "events": diagnostics.get("events", []),
        "cache_state": diagnostics.get("cache_state", {}),
    }

def get_model_answers(base_model_path, ea_model_path, model_id, questions,
                      answer_file, max_new_token, num_choices,
                      num_gpus_per_model, max_gpu_memory, temperature, args):
    if not questions:
        return [], []
    model = EaModel.from_pretrained(
        base_model_path=base_model_path, ea_model_path=ea_model_path,
        total_token=args.total_token, depth=args.depth, top_k=args.top_k,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
    if args.enable_online_adaptation:
        model.setup_online_adaptation(
            adaptation_lr=args.adaptation_lr,
            adaptation_temperature=args.adaptation_temperature,
            scope=getattr(args, "adaptation_scope", "default"),
            objective=getattr(args, "adaptation_objective", "kl"),
            reset_granularity=getattr(args, "reset_granularity", "choice"),
            mode=getattr(args, "adaptation_mode", "reset"))
        effective_lr = _effective_adaptation_value(model, "adaptation_lr", args.adaptation_lr)
        effective_temp = _effective_adaptation_value(
            model, "adaptation_temperature", args.adaptation_temperature)
        print(f"Online adaptation enabled with effective lr={effective_lr}, temperature={effective_temp}")

    tokenizer = model.get_tokenizer()
    model.eval()
    print("Check model training state:", model.training)
    print("CUDA VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    first_turns = questions[0].get("turns", [])
    if first_turns:
        warmup_messages = [{"role": "user", "content": first_turns[0]}]
        warmup_prompt = tokenizer.apply_chat_template(
            warmup_messages, tokenize=False, add_generation_prompt=True)
        warmup_ids = tokenizer([warmup_prompt], add_special_tokens=False).input_ids
        for _ in range(3):
            torch.manual_seed(0)
            model.eagenerate(
                torch.as_tensor(warmup_ids).cuda(), temperature=temperature,
                max_new_tokens=max_new_token, log=True, is_llama3=True,
                enable_adaptation=False, diagnostics=False)
        print("Warmup done")

    answer_records = []
    local_stats = []
    for question in tqdm(questions):
        choices = []
        for choice_index in range(num_choices):
            torch.manual_seed(choice_index)
            if args.enable_online_adaptation and getattr(args, "reset_granularity", "choice") == "choice":
                model.reset_online_adaptation()
            messages, turns, idxs, new_tokens, wall_time = [], [], [], [], []
            for turn_index, question_turn in enumerate(question.get("turns", [])):
                if (args.enable_online_adaptation and
                        getattr(args, "reset_granularity", "choice") == "turn" and
                        turn_index > 0):
                    model.reset_online_adaptation()
                messages.append({"role": "user", "content": question_turn})
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx, run_stats = model.eagenerate(
                    torch.as_tensor(input_ids).cuda(), temperature=temperature,
                    max_new_tokens=max_new_token, log=True, is_llama3=True,
                    enable_adaptation=args.enable_online_adaptation,
                    adaptation_lr=args.adaptation_lr,
                    adaptation_temperature=args.adaptation_temperature,
                    diagnostics=args.enable_online_adaptation)
                torch.cuda.synchronize()
                elapsed = time.time() - start_time
                accepted = run_stats.get("total_accept_length", 0)
                drafted = run_stats.get("total_drafted_tokens", 0)
                steps = run_stats.get("total_steps", 0)
                accepted = accepted.item() if hasattr(accepted, "item") else accepted
                drafted = drafted.item() if hasattr(drafted, "item") else drafted
                steps = steps.item() if hasattr(steps, "item") else steps
                raw_losses = run_stats.get("losses", [])
                finite_losses = []
                nonfinite_losses = 0
                for loss in raw_losses:
                    loss = loss.item() if hasattr(loss, "item") else loss
                    try:
                        if math.isfinite(float(loss)):
                            finite_losses.append(float(loss))
                        else:
                            nonfinite_losses += 1
                    except (TypeError, ValueError):
                        nonfinite_losses += 1
                avg_loss = (sum(finite_losses) / len(finite_losses)) if finite_losses else None
                status, reason, counts, details = _adaptation_details(
                    run_stats, args.enable_online_adaptation)
                local_stats.append({
                    "qid": question["question_id"], "category": question.get("category", "unknown"),
                    "choice": choice_index,
                    "turn": turn_index, "status": status, "reason": reason,
                    "counts": counts, "diagnostics": details,
                    "speed": int(new_token) / elapsed if elapsed > 0 else 0.0,
                    "acc_rate": accepted / drafted if drafted else 0.0,
                    "avg_accept_len": accepted / steps if steps else 0.0,
                    "total_accept_tokens": accepted, "total_drafted_tokens": drafted,
                    "total_steps": steps, "avg_loss": avg_loss,
                    "loss_count": len(finite_losses),
                    "loss_nonfinite_count": nonfinite_losses,
                    "new_tokens": int(new_token), "time": elapsed,
                    "experiment": getattr(args, "experiment_name", "baseline"),
                    "adaptation_scope": getattr(args, "adaptation_scope", "default"),
                    "adaptation_objective": getattr(args, "adaptation_objective", "kl"),
                    "reset_granularity": getattr(args, "reset_granularity", "choice"),
                    "adaptation_mode": getattr(args, "adaptation_mode", "reset"),
                })
                output = _decode_output(tokenizer, output_ids, len(input_ids[0]))
                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(elapsed)
                messages.append({"role": "assistant", "content": output})
            choices.append({"index": choice_index, "turns": turns, "idxs": idxs,
                            "new_tokens": new_tokens, "wall_time": wall_time})
        answer_records.append({
            "question_id": question["question_id"], "answer_id": shortuuid.uuid(),
            "model_id": model_id, "choices": choices, "tstamp": time.time(),
        })
    return answer_records, local_stats


def _finite_values(stats, key):
    values, nonfinite, missing = [], 0, 0
    for stat in stats:
        value = stat.get(key)
        if value is None:
            missing += 1
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            nonfinite += 1
            continue
        if math.isfinite(value):
            values.append(value)
        else:
            nonfinite += 1
    return values, nonfinite, missing

def save_statistics(all_stats, stat_file):
    directory = os.path.dirname(stat_file) or os.getcwd()
    os.makedirs(directory, exist_ok=True)
    question_count = len({str(s.get("qid")) for s in all_stats})
    choice_count = len({(str(s.get("qid")), s.get("choice")) for s in all_stats})
    status_counts = {name: sum(s.get("status") == name for s in all_stats)
                     for name in ("updated", "skipped", "rolled_back", "disabled")}
    with open(stat_file, "w") as fout:
        fout.write("=" * 80 + "\n")
        fout.write(f"Total Questions: {question_count}\n")
        fout.write(f"Total Choices: {choice_count}\n")
        fout.write(f"Total Generations: {len(all_stats)}\n")
        fout.write("Adaptation: " + ", ".join(
            f"{key}={value}" for key, value in status_counts.items()) + "\n")
        fout.write("=" * 80 + "\n\n")

        def write_metric(label, key, percent=False):
            values, nonfinite, missing = _finite_values(all_stats, key)
            fout.write(f"[{label}]\n")
            fout.write(f"  Finite: {len(values)}, Nonfinite: {nonfinite}, Missing: {missing}\n")
            if values:
                suffix = "%" if percent else ""
                scale = 100 if percent else 1
                fout.write(f"  Mean:   {np.mean(values) * scale:.4f}{suffix}\n")
                fout.write(f"  Median: {np.median(values) * scale:.4f}{suffix}\n")
                fout.write(f"  Max:    {np.max(values) * scale:.4f}{suffix}\n")
                fout.write(f"  Min:    {np.min(values) * scale:.4f}{suffix}\n\n")
            else:
                fout.write("  No finite data.\n\n")

        write_metric("Speed (tokens/s)", "speed")
        write_metric("Acceptance Rate (Accepted/Drafted)", "acc_rate", True)
        write_metric("Avg Accept Length (per step)", "avg_accept_len")
        losses, loss_nonfinite, loss_missing = _finite_values(all_stats, "avg_loss")
        actual_nonfinite = loss_nonfinite + sum(s.get("loss_nonfinite_count", 0) for s in all_stats)
        fout.write("[Online Loss]\n")
        fout.write(f"  Finite: {len(losses)}, Nonfinite: {actual_nonfinite}, Missing: {loss_missing}\n")
        if losses:
            fout.write(f"  Mean:   {np.mean(losses):.6f}\n")
            fout.write(f"  Median: {np.median(losses):.6f}\n")
            fout.write(f"  Max:    {np.max(losses):.6f}\n")
            fout.write(f"  Min:    {np.min(losses):.6f}\n")
        else:
            fout.write("  No finite loss data.\n")
        fout.write("\nPer-generation rows\n")
        for stat in all_stats:
            row = {key: stat.get(key) for key in
                   ("qid", "choice", "turn", "status", "reason", "counts")}
            fout.write(json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Statistics saved to {stat_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ea-model-path", type=str,
                        default="/home/lyh/weights/hf/eagle3/DSL/8B3/",
                        help="The EAGLE weights path or repository ID.")
    parser.add_argument("--base-model-path", type=str,
                        default="/home/lyh/weights/DSL/8B/", help="Base model path.")
    parser.add_argument("--load-in-8bit", action="store_false", help="Use 8-bit quantization")
    parser.add_argument("--model-id", type=str, default="llama38b2_40")
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--max-gpu-memory", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tree-choices", type=str, default="mc_sim_7b_63")
    parser.add_argument("--enable-online-adaptation", action="store_true")
    parser.add_argument("--adaptation-lr", type=float, default=5e-5)
    parser.add_argument("--adaptation-temperature", type=float, default=1.0)
    parser.add_argument("--experiment-name", type=str, default="baseline")
    parser.add_argument("--adaptation-scope", choices=["default", "head_only"], default="default")
    parser.add_argument("--adaptation-objective", choices=["kl", "acceptance"], default="kl")
    parser.add_argument("--reset-granularity", choices=["choice", "turn", "stream"], default="choice")
    parser.add_argument("--adaptation-mode", choices=["reset", "persistent"], default="reset")
    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray
        ray.init()
    question_file = f"{parent_dir}/data/{args.bench_name}/question.jsonl"
    answer_file = args.answer_file or f"ds_{args.bench_name}_online/{args.model_id}.jsonl"
    answer_file = _normalize_path(answer_file)
    print(f"Output to {answer_file}")
    run_eval(
        args.base_model_path, args.ea_model_path, args.model_id, question_file,
        args.question_begin, args.question_end, answer_file, args.max_new_token,
        args.num_choices, args.num_gpus_per_model, args.num_gpus_total,
        args.max_gpu_memory, args.temperature, args)
