import logging
import os
import time

import numpy as np

from common.console import green, red, bold, gray
from common.shared_config import model_abbr
from eval.logging import setup_logging, log_model_config, log_testing_result, print_log
from safe.fact_checker import FactChecker

# TODO The following comments should be inserted in the README.md
# For multimodal usage turn image into a tensor by either:
# 1) pulling it from link:
#    image_url = "https://llava-vl.github.io/static/images/view.jpg"
#    image = Image.open(requests.get(image_url, stream=True).raw)
#   or
# 2) pulling it from path
#    image_path = path_to_data + "MAFC_test/image_claims/00000.png"
#    image = Image.open(image_path)
#
# Hand the tensor as second argument to Factchecker.check

logging.getLogger('urllib3').setLevel(logging.WARNING)


def evaluate(
        model,
        multimodal_model,
        search_engine,
        benchmark,
        n=None,
        extract_claims=True,
        verbose=False,
        logging=True
) -> float:
    if logging:
        # Setup logger
        os.makedirs('log', exist_ok=True)
        dataset_abbr = benchmark.name
        model_ab = model_abbr[model]

        config_logger, testing_logger, print_logger = setup_logging(dataset_abbr, model_ab)

        log_model_config(config_logger, {
            "LLM": model,
            "MLLM": multimodal_model,
            "Search Engine": search_engine,
            "Benchmark": benchmark.name,
            "Extract Claims": extract_claims,
            "Full Dataset": True if n == len(benchmark) else f'{n} samples'
        })
        start_time = time.time()

    summary = f"\n\nLLM: {model}, " \
              f"MLLM: {multimodal_model}, " \
              f"Search Engine: {search_engine}, " \
              f"Benchmark: {benchmark}\n"

    print(bold(gray(summary)))

    if logging:
        print_log(print_logger, summary)

    fc = FactChecker(
        model=model,
        multimodal_model=multimodal_model,
        search_engine=search_engine,
        extract_claims=extract_claims,
    )

    if not n:
        n = len(benchmark)

    predictions = []
    for i, instance in enumerate(benchmark):
        content = instance["content"]

        prediction = fc.check(content, verbose=verbose, logger=print_logger if logging else None)
        prediction_is_correct = instance["label"] == prediction

        if logging:
            log_message = {
                "sample_index": i + 1,
                "target": instance["label"].value,
                "predicted": prediction.value,
                "correct": prediction_is_correct
            }
            log_testing_result(testing_logger, log_message)
            if prediction_is_correct:
                print_log(print_logger, "CORRECT")
            else:
                print_log(print_logger, "WRONG - Ground truth: " + instance["label"].value)

        predictions.append(prediction)
        if prediction_is_correct:
            print(bold(green("CORRECT")))
        else:
            print(bold(red("WRONG - Ground truth: " + instance["label"].value)))

        if len(predictions) == n:
            break

    # Compute metrics
    ground_truth = benchmark.get_labels()[:n]
    correct_predictions = np.array(predictions) == np.array(ground_truth)
    accuracy = np.sum(correct_predictions) / n
    print(f"Accuracy: {accuracy * 100:.1f} %\n\n")

    if logging:
        end_time = time.time()
        log_testing_result(testing_logger, {
            "Accuracy": f"{accuracy * 100:.1f} %",
            "Correct Predictions": correct_predictions.tolist(),
            "Incorrect Predictions": (n - correct_predictions.sum()).tolist(),
            "Duration of Run": f'{end_time - start_time} seconds'
        })

    return accuracy