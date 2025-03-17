from src.evaluator import AgentEvaluator
import logging


# Set up logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Custom configuration (optional)
custom_config = {
    'bleu_config': {
        'max_order': 2,  # Use bigrams only
        'weights': (0.5, 0.5)  # Equal weights for unigrams and bigrams
    },
    'semantic_threshold': 0.65
}

# Initialize evaluator with custom config
evaluator = AgentEvaluator(
    ground_truth_path="ground_truth.csv",
    logs_path="agno_debug_and_response.log",
    config=custom_config
)

# Run evaluation with debug info
metrics, coverage = evaluator.evaluate_all()
logger.info(f"Found {coverage*100:.1f}% coverage")
logger.info(f"Routing accuracy: {metrics.get('routing_accuracy', 0):.2f}")
logger.info(f"Tool accuracy: {metrics.get('tool_accuracy', 0):.2f}")

# Generate evaluation report
report = evaluator.generate_report(metrics, coverage)
print(report)

print("\nDetailed Metrics Analysis:")
print("=" * 50)

print("\nAccuracy Metrics:")
print(f"Routing Accuracy: {metrics.get('routing_accuracy', 0):.2f}")
print(f"Planning Score: {metrics.get('planning_score', 0):.2f}")
print(f"Reasoning Score: {metrics.get('reasoning_score', 0):.2f}")
print(f"Tool Usage Accuracy: {metrics.get('tool_accuracy', 0):.2f}")

print("\nText Similarity Metrics:")
print(f"BLEU Score: {metrics.get('bleu', 0):.2f}")
print(f"Semantic Similarity: {metrics.get('semantic_similarity', 0):.2f}")
print(f"Levenshtein Similarity: {metrics.get('levenshtein_similarity', 0):.2f}")
print(f"BERTScore: {metrics.get('bert_score', 0):.2f}")

print("\nROUGE Scores:")
print(f"ROUGE-1: {metrics.get('rouge1', 0):.2f}")
print(f"ROUGE-2: {metrics.get('rouge2', 0):.2f}")
print(f"ROUGE-L: {metrics.get('rougeL', 0):.2f}")

# Calculate and display averages
accuracy_metrics = ['routing_accuracy', 'planning_score', 'reasoning_score', 'tool_accuracy']
average_accuracy = sum(metrics.get(m, 0) for m in accuracy_metrics) / len(accuracy_metrics)

rouge_metrics = ['rouge1', 'rouge2', 'rougeL']
average_rouge = sum(metrics.get(m, 0) for m in rouge_metrics) / len(rouge_metrics)

semantic_metrics = ['bleu', 'semantic_similarity', 'levenshtein_similarity', 'bert_score']
average_semantic = sum(metrics.get(m, 0) for m in semantic_metrics) / len(semantic_metrics)

print("\nSummary Scores:")
print(f"Average Accuracy Score: {average_accuracy:.2f}")
print(f"Average ROUGE Score: {average_rouge:.2f}")
print(f"Average Semantic Score: {average_semantic:.2f}")
print(f"Overall Performance Score: {(average_accuracy + average_rouge + average_semantic) / 3:.2f}")