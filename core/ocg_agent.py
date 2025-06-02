import json
import os
import logging
import traceback
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple

import requests  # Assuming still needed elsewhere

from .QA import *  # noqa: F403 – kept from original for compatibility
from utils.utils import *  # noqa: F403
from tool.AISearch import self_AI_search

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _save_log(log_json: dict, file_path: str) -> None:
    """Write *log_json* to *file_path* in a consistent, single place."""
    # Ensure parent directory exists (first‑run safety)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(log_json, f, ensure_ascii=False, indent=4)


def _init_logging(index: int, question: str, args) -> Tuple[datetime, str, Dict, str]:
    """Prepare initial log scaffold and return bookkeeping variables."""
    timebegin = datetime.now()
    time_stamp = timebegin.strftime("%Y-%m-%d-%H-%M-%S")
    log_json = {"question": question}

    file_path = os.path.join(
        args.save_path, args.dataset_name, f"{index}_{time_stamp}.json"
    )
    _save_log(log_json, file_path)
    return timebegin, time_stamp, log_json, file_path


def _adt_step(question: str, log_json: dict, file_path: str):
    """Run *ADT_generation* and persist results."""
    adt_content = ADT_generation(question=deepcopy(question))  # noqa: F405
    adt_dict = json.loads(adt_content)
    atrribute_dict = {attr["Name"]: attr["Description"] for attr in adt_dict["Attributes"]}
    log_json["ADT"] = adt_dict
    _save_log(log_json, file_path)
    return adt_content, adt_dict, atrribute_dict


def _write_subquestion_scaffold(subquestion_list: List[str], log_json: dict, file_path: str) -> None:
    """Seed the *subquestion* section of the log."""
    log_json["subquestion"] = {
        sq: {"AI_search_content_list": [], "citations": []} for sq in subquestion_list
    }
    _save_log(log_json, file_path)


def _ai_search_for_subquestion(subquestion: str, args, citations):
    """Dispatch to the correct AI‑search backend depending on dataset."""
    if "movie" in args.dataset_name:
        return self_AI_search(
            query=subquestion,
            pagenum=args.pagenum,
            threshold=args.threshold,
            existed_citation_list=deepcopy(list(citations)),
        )
    return self_AI_search_edu(query=subquestion)


def _span_and_extract(content: str, adt_content: str, adt_dict: dict, args):
    """Apply span prediction (optional) and extract candidate items."""
    if args.span:
        content = SpanPredict(adt=deepcopy(adt_content), article=deepcopy(content))  # noqa: F405

    candidate_items_list_local = Extract(  # noqa: F405
        article=deepcopy(content), ADT=deepcopy(adt_content)
    )

    # Guarantee every attribute key exists so later code need not check.
    for candidate_item in candidate_items_list_local:
        for attribute in adt_dict["Attributes"]:
            key = attribute["Name"]
            candidate_item.setdefault(key, "NOT FOUND")
    return content, candidate_items_list_local


def _merge_candidate_lists(candidate_items_list_global: List[List[dict]]):
    """Flatten the nested candidate‑item lists produced in earlier stages."""
    return candidate_items_list_merge(candidate_items_list_global)  # noqa: F405

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def candidate_retrieval(index: int, question: str, args):
    """End‑to‑end pipeline (refactored for readability & robustness)."""

    # -----------------------------------------------------------------------
    # Safe defaults – these ensure variables always exist even if an early
    # exception is thrown before _init_logging() succeeds.
    # -----------------------------------------------------------------------
    timebegin: datetime = datetime.now()
    file_path: str | None = None
    log_json: dict = {}

    try:
        # -------------------------------------------------------------------
        # Stage 0 – initial bookkeeping
        # -------------------------------------------------------------------
        timebegin, time_stamp, log_json, file_path = _init_logging(index, question, args)

        # -------------------------------------------------------------------
        # Stage 1 – ADT generation
        # -------------------------------------------------------------------
        adt_content, adt_dict, atrribute_dict = _adt_step(question, log_json, file_path)

        # -------------------------------------------------------------------
        # Stage 2 – sub‑question generation (optional rewrite)
        # -------------------------------------------------------------------
        subquestion_list = [question] + (
            rewrite(question=deepcopy(question)) if args.rewrite else []  # noqa: F405
        )
        _write_subquestion_scaffold(subquestion_list, log_json, file_path)

        citations: List[str] = []
        candidate_items_list_global: List[List[dict]] = []

        # -------------------------------------------------------------------
        # Stage 3 – retrieve → span‑predict → extract per sub‑question
        # -------------------------------------------------------------------
        for subquestion in subquestion_list:
            print(f"\tResponse to subquestion: {subquestion}")
            local_external_knowledge_dict, local_set = _ai_search_for_subquestion(
                subquestion, args, citations
            )

            AI_search_content_list = []
            for title, content in local_external_knowledge_dict.items():
                print(f"\t\tExtract entity from {title}")
                spanned_content, candidate_items_list_local = _span_and_extract(
                    content, adt_content, adt_dict, args
                )

                candidate_items_list_global.append(candidate_items_list_local)
                AI_search_content_list.append(
                    {
                        "title": title,
                        "content": content,
                        "spanned_content": spanned_content,
                        "candidate_items_list_local": candidate_items_list_local,
                    }
                )

            log_json["subquestion"][subquestion]["AI_search_content_list"] = AI_search_content_list
            log_json["subquestion"][subquestion]["citations"] = list(local_set)
            _save_log(log_json, file_path)

            citations.extend(list(local_set))

        # -------------------------------------------------------------------
        # Stage 4 – optional generative augmentation
        # -------------------------------------------------------------------
        if args.gpt_aug:
            topk = 20 if "movie" in args.dataset_name else 10
            gpt_predict_ranking_list = generative_retrieval(  # noqa: F405
                question, adt_content, topk=topk, model_name=args.model_name
            )

            gpt_candidate_items_list = []
            for item_dict in gpt_predict_ranking_list:
                for attribute in adt_dict["Attributes"]:
                    if item_dict.get(attribute["Name"], "NOT FOUND") == "NOT FOUND":
                        item_dict[attribute["Name"]] = "NOT FOUND"
                item_dict["Generative"] = args.model_name
                gpt_candidate_items_list.append(item_dict)

            candidate_items_list_global.append(gpt_candidate_items_list)

        # -------------------------------------------------------------------
        # Stage 5 – merge candidates across branches
        # -------------------------------------------------------------------
        candidate_items_list = _merge_candidate_lists(candidate_items_list_global)
        log_json["candidate_items_list_prepare"] = candidate_items_list
        _save_log(log_json, file_path)

        # -------------------------------------------------------------------
        # Stage 6 – optional attribute completion
        # -------------------------------------------------------------------
        if args.complete:
            copy_candidate_items_list = deepcopy(candidate_items_list)
            necessary_keys = [
                attr["Name"] for attr in adt_dict["Attributes"] if attr["Type"] == "Required"
            ]

            log_json["AI_search_content_for_complete"] = {}
            _save_log(log_json, file_path)

            for idx, candidate_item in enumerate(copy_candidate_items_list):
                name = candidate_item.get("Name", "NOT FOUND")
                if name == "NOT FOUND":
                    continue

                log_json["AI_search_content_for_complete"][name] = {}
                _save_log(log_json, file_path)

                for _ in range(args.max_loop_times):
                    missing_key = next(
                        (
                            key
                            for key in necessary_keys
                            if candidate_item.get(key, "NOT FOUND") == "NOT FOUND"
                        ),
                        None,
                    )
                    if missing_key is None:
                        break

                    in_context_situation = (
                        "User ask for: {question}\n"
                        "For a candidate item: {name}\n"
                        "Need to find related information about the {missing_key} for it.\n"
                        "The {missing_key} refers to {description}."
                    ).format(
                        question=question,
                        name=name,
                        missing_key=missing_key,
                        description=atrribute_dict[missing_key],
                    )

                    generated_query = generate_single_query(  # noqa: F405
                        in_context_situation=in_context_situation
                    )

                    local_external_knowledge_dict, local_set = _ai_search_for_subquestion(
                        generated_query, args, citations
                    )
                    citations.extend(list(local_set))

                    for title, content in local_external_knowledge_dict.items():
                        spanned_content = (
                            SpanPredict(adt=deepcopy(adt_content), article=deepcopy(content))
                            if args.span
                            else deepcopy(content)
                        )

                        completed_candidate_item = complete(  # noqa: F405
                            candidate_item=deepcopy(candidate_item),
                            article=deepcopy(spanned_content),
                            ADT=adt_content,
                        )

                        log_json["AI_search_content_for_complete"][name] = {
                            "in_context_situation": in_context_situation,
                            "title": title,
                            "content": content,
                            "spanned_content": spanned_content,
                            "candidate_item": candidate_item,
                            "completed_candidate_item": completed_candidate_item,
                        }
                        _save_log(log_json, file_path)

                        # Accept improvements only when a *Name* is present
                        if completed_candidate_item.get("Name", "NOT FOUND") != "NOT FOUND":
                            candidate_item = completed_candidate_item

                    # loop … continues (break handled above)
                copy_candidate_items_list[idx] = candidate_item
        else:
            # 不进行补全，则直接沿用已有候选
            copy_candidate_items_list = candidate_items_list

        # 统一清理无效候选
        copy_candidate_items_list = [
            item for item in copy_candidate_items_list if item.get("Name", "NOT FOUND") != "NOT FOUND"
        ]
        log_json["candidate_items_list_final"] = copy_candidate_items_list

        # -------------------------------------------------------------------
        # Stage 7 – wrap‑up & timing
        # -------------------------------------------------------------------
        duration_seconds = (datetime.now() - timebegin).total_seconds()
        log_json["duration_seconds"] = duration_seconds
        _save_log(log_json, file_path)

        print(f"Time taken: {duration_seconds:.2f} seconds")
        print(
            "Open Candidate Generation Finished, generated "
            f"{len(copy_candidate_items_list)} candidate items."
        )
        print(
            "The names are "
            f"{[item.get('Name', 'NOT FOUND') for item in copy_candidate_items_list]}"
        )

    # -----------------------------------------------------------------------
    # Fatal‑error path
    # -----------------------------------------------------------------------
    except Exception as e:
        duration_seconds = (datetime.now() - timebegin).total_seconds()

        # 如果 file_path 在异常前已成功生成，则把错误信息写入日志
        if file_path:
            log_json.setdefault("error", str(e))
            log_json["duration_seconds"] = duration_seconds
            _save_log(log_json, file_path)

        logging.error("Error during candidate_retrieval", exc_info=True)
        print("Error:", e)
        print(traceback.format_exc())
        print(f"Time taken before failure: {duration_seconds:.2f} seconds")
        print("Open Candidate Generation Failed.")
