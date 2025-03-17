import pandas as pd
import numpy as np
import nltk
# nltk.download('punkt')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from nltk.metrics import edit_distance
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional, Tuple
import json
import logging
from pathlib import Path
import re
from fuzzywuzzy import fuzz
from datetime import datetime
from collections import defaultdict
from src.config import EvaluatorConfig

class AgentEvaluator:
    """Enhanced wrapper class for evaluating agent performance against ground truth."""
    
    def __init__(self, ground_truth_path: str, logs_path: str, config: Dict = None):
        self.logger = logging.getLogger('evaluator')
        self.ground_truth = pd.read_csv(ground_truth_path)
        self.logs_path = Path(logs_path)
        
        # Load configuration
        self.config = EvaluatorConfig.load(config)
        
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(
            self.config['rouge_metrics'], 
            use_stemmer=True
        )
        #self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentence_model = SentenceTransformer(
            self.config['model_config']['sentence_transformer']
        )
        
        # Initialize BLEU smoothing
        self.smoothing = SmoothingFunction().method1 if self.config['bleu_config']['smoothing_function'] else None

    def parse_log_entry(self, log_content: str) -> Dict:
        """Parse a single log entry to extract relevant information."""
        entry = {
            'raw_content': log_content,
            'query': '',
            'assigned_agents': '',
            'tool_calls': [],
            'metrics': {},
            'response': '',
            'agent_id': '',
            'session_id': '',
            'agent_run': '',
            'performance_metrics': {}
        }
        
        # Use configured patterns
        patterns = self.config['log_patterns']
        
        # Extract query
        query_match = re.search(patterns['query'], log_content, re.DOTALL)
        if query_match:
            entry['query'] = query_match.group(1).strip()
        
        # Extract response using configured patterns
        for pattern_key in ['response', 'assistant']:
            match = re.search(patterns[pattern_key], log_content, re.DOTALL)
            if match:
                entry['response'] = match.group(1).strip()
                break
        
        # Extract agent identification info
        if 'agent_id' in patterns:
            agent_id_match = re.search(patterns['agent_id'], log_content)
            if agent_id_match:
                entry['agent_id'] = agent_id_match.group(1)
                
        if 'session_id' in patterns:
            session_match = re.search(patterns['session_id'], log_content)
            if session_match:
                entry['session_id'] = session_match.group(1)
                
        if 'agent_run' in patterns:
            run_match = re.search(patterns['agent_run'], log_content)
            if run_match:
                entry['agent_run'] = run_match.group(1)
        
        # Extract assigned agents
        if 'assigned_agents' in patterns:
            assigned_agents = re.findall(patterns['assigned_agents'], log_content)
            entry['assigned_agents'] = ','.join(assigned_agents)
        
        # Extract performance metrics
        if 'metrics' in patterns:
            for metric_name, pattern in patterns['metrics'].items():
                match = re.search(pattern, log_content)
                if match:
                    entry['performance_metrics'][metric_name] = float(match.group(1))
        
        # Extract tool calls using updated patterns
        tool_patterns = patterns.get('tool_calls', {})
        if 'block_start' in tool_patterns:
            # Find the tool calls block
            tool_block_match = re.search(
                f"{tool_patterns['block_start']}(.*?){tool_patterns['block_end']}", 
                log_content, 
                re.DOTALL
            )
            
            if tool_block_match:
                tool_block = tool_block_match.group(1)
                function_patterns = tool_patterns['function_call']
                
                # Find all function calls in the block
                names = re.findall(function_patterns['name'], tool_block)
                arguments = re.findall(function_patterns['arguments'], tool_block)
                types = re.findall(function_patterns['type'], tool_block)
                
                # Combine the extracted information
                entry['tool_calls'] = [
                    {
                        'type': t,
                        'function': {
                            'name': n,
                            'arguments': json.loads(a.replace('\\"', '"'))
                        }
                    }
                    for t, n, a in zip(types, names, arguments)
                    if self._is_valid_json(a.replace('\\"', '"'))
                ]
        
        return entry

    def _is_valid_json(self, json_str: str) -> bool:
        """Validate JSON string."""
        try:
            json.loads(json_str)
            return True
        except json.JSONDecodeError:
            return False

    def _generate_analysis(self, metrics: Dict[str, float]) -> str:
        """Generate detailed analysis using configurable thresholds."""
        analysis = []
        thresholds = self.config['analysis_thresholds']
        
        for metric_type, config in thresholds.items():
            if metric_type == 'routing':
                value = metrics.get('routing_accuracy', 0)
            elif metric_type == 'semantic':
                value = (metrics.get('semantic_similarity', 0) + metrics.get('bert_score', 0)) / 2
            else:  # response_quality
                rouge_metrics = self.config['rouge_metrics']
                value = sum(metrics.get(f'rouge{m}', 0) for m in rouge_metrics) / len(rouge_metrics)
                
            analysis.append(
                config['success_message'] if value > config['threshold'] 
                else config['improvement_message']
            )
            
        return "\n".join(analysis)

    def generate_report(self, metrics: Dict[str, float], coverage: float) -> str:
        """Generate an enhanced evaluation report with all metrics."""
        report_format = self.config['report_format']
        report = [report_format['title'], "=" * len(report_format['title'])]
        
        for section_key, section in report_format['sections'].items():
            report.extend([f"\n{section['title']}", "-" * len(section['title'])])
            
            for metric in section['metrics']:
                if metric == 'coverage':
                    value = coverage * 100
                elif metric == 'overall_performance':
                    value = sum(metrics.values()) / len(metrics)
                else:
                    value = metrics.get(metric, 0)
                    
                report.append(f"{metric.replace('_', ' ').title()}: {value:.2f}")
        
        report.extend(["\nAnalysis", "--------", self._generate_analysis(metrics)])
        return "\n".join(report)

    def _normalize_agents(self, agents_str: str) -> set:
        """Normalize agent names using configured patterns."""
        if not agents_str:
            return set()
            
        normalized = agents_str.lower()
        
        # Apply configured replacements
        for old, new in self.config['normalization']['agent_replacements']:
            normalized = normalized.replace(old, new)
            
        # Split by configured separators
        for sep in self.config['normalization']['separators']:
            normalized = normalized.replace(sep, ' ')
            
        return {x.strip() for x in normalized.split() if x.strip()}

    def _extract_metric(self, metrics_text: str, metric_name: str) -> float:
        """Extract metric value using configured patterns."""
        if metric_name in self.config['metric_patterns']:
            pattern = self.config['metric_patterns'][metric_name]
            match = re.search(pattern, metrics_text)
            return float(match.group(1)) if match else 0.0
        return 0.0

    def evaluate_response(self, log_entry: Dict, ground_truth_entry: Dict) -> Dict[str, float]:
        """Enhanced evaluation using multiple NLP metrics."""
        metrics = {
            'routing_accuracy': self._calculate_routing_accuracy(
                log_entry.get('assigned_agents', ''),
                ground_truth_entry['AgentAssigned']
            ),
            'planning_score': ground_truth_entry['PlanningIndex'],
            'reasoning_score': ground_truth_entry['ReasoningRelevancy'],
            'tool_accuracy': self._calculate_tool_accuracy(
                log_entry.get('tool_calls', []),
                ground_truth_entry['expected_tools'].split(', ')
            )
        }
        
        # Add performance metrics from log
        if 'performance_metrics' in log_entry:
            metrics.update({
                f'log_{k}': v 
                for k, v in log_entry['performance_metrics'].items()
            })
        
        # Convert agent and tool selections to binary arrays for classification metrics
        actual_agents = set(self._normalize_agents(log_entry.get('assigned_agents', '')))
        expected_agents = set(self._normalize_agents(ground_truth_entry['AgentAssigned']))
        all_agents = actual_agents.union(expected_agents)
        
        y_true = [1 if agent in expected_agents else 0 for agent in all_agents]
        y_pred = [1 if agent in actual_agents else 0 for agent in all_agents]
        
        # Add classification metrics
        if len(y_true) > 0:
            metrics.update({
                'agent_precision': precision_score(y_true, y_pred, zero_division=0),
                'agent_recall': recall_score(y_true, y_pred, zero_division=0),
                'agent_f1': f1_score(y_true, y_pred, zero_division=0)
            })
        
        # Get response texts
        response_text = self._extract_response_text(log_entry)
        expected_text = ground_truth_entry['expected_answer']
        
        # Tokenize texts for BLEU score
        response_tokens = word_tokenize(response_text.lower())
        expected_tokens = word_tokenize(expected_text.lower())
        
        # Calculate ROUGE scores
        rouge_scores = self.rouge_scorer.score(response_text, expected_text)
        
        # Calculate BLEU score with smoothing
        bleu_score = sentence_bleu(
            [expected_tokens], 
            response_tokens,
            weights=self.config['bleu_config']['weights'],
            smoothing_function=self.smoothing
        )
        
        # Calculate semantic similarity using BERT embeddings
        response_embedding = self.sentence_model.encode([response_text])[0]
        expected_embedding = self.sentence_model.encode([expected_text])[0]
        semantic_similarity = 1 - cosine(response_embedding, expected_embedding)
        
        # Calculate Levenshtein distance (normalized)
        max_len = max(len(response_text), len(expected_text))
        levenshtein_similarity = 1 - (edit_distance(response_text, expected_text) / max_len if max_len > 0 else 0)
        
        # Calculate BERTScore
        P, R, F1 = bert_score([response_text], [expected_text], lang='en')
        bert_f1 = F1.mean().item()
        
        # Update metrics with all scores
        metrics.update({
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'bleu': bleu_score,
            'semantic_similarity': semantic_similarity,
            'levenshtein_similarity': levenshtein_similarity,
            'bert_score': bert_f1
        })

        # Calculate tool usage accuracy
        tool_accuracy = self._calculate_tool_accuracy(
            log_entry.get('tool_calls', []),
            ground_truth_entry.get('expected_tools', '')
        )
        metrics['tool_accuracy'] = tool_accuracy
        
        return metrics

    def generate_report(self, metrics: Dict[str, float], coverage: float) -> str:
        """Generate an enhanced evaluation report with all metrics."""
        report = [
            "Agent Evaluation Report",
            "======================",
            f"\nCoverage: {coverage*100:.1f}%",
            f"\nOverall Performance: {sum(metrics.values())/len(metrics):.2f}",
            "\nRouting and Planning Metrics",
            "-------------------------",
            f"Routing Accuracy: {metrics.get('routing_accuracy', 0):.2f}",
            f"Planning Score: {metrics.get('planning_score', 0):.2f}",
            f"Reasoning Score: {metrics.get('reasoning_score', 0):.2f}",
            f"Tool Accuracy: {metrics.get('tool_accuracy', 0):.2f}",
            "\nText Similarity Metrics",
            "---------------------",
            f"BLEU Score: {metrics.get('bleu', 0):.2f}",
            f"Semantic Similarity: {metrics.get('semantic_similarity', 0):.2f}",
            f"Levenshtein Similarity: {metrics.get('levenshtein_similarity', 0):.2f}",
            f"BERTScore: {metrics.get('bert_score', 0):.2f}",
            "\nROUGE Scores",
            "------------",
            f"ROUGE-1: {metrics.get('rouge1', 0):.2f}",
            f"ROUGE-2: {metrics.get('rouge2', 0):.2f}",
            f"ROUGE-L: {metrics.get('rougeL', 0):.2f}",
            "\nAnalysis",
            "--------",
            self._generate_analysis(metrics)
        ]
        
        return "\n".join(report)

    def _extract_response_text(self, log_entry: Dict) -> str:
        """Extract response text from log entry."""
        # First try to get from the response field
        if log_entry.get('response'):
            return log_entry['response']
            
        # If not found, try to extract from raw content using configured patterns
        raw_content = log_entry.get('raw_content', '')
        patterns = self.config['log_patterns']
        
        # Try all configured response patterns
        for pattern_key in ['response', 'assistant']:
            if pattern_key in patterns:
                match = re.search(patterns[pattern_key], raw_content, re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        # If still not found, try timestamp-based extraction
        if 'timestamp' in patterns:
            timestamps = re.finditer(patterns['timestamp'], raw_content)
            last_response = ''
            for match in timestamps:
                pos = match.end()
                content = raw_content[pos:].split('\n', 1)
                if len(content) > 1:
                    last_response = content[1].strip()
            if last_response:
                return last_response
        
        return ''

    def _clean_response_text(self, text: str) -> str:
        """Clean up extracted response text."""
        if not text:
            return ""
        
        # Use configured cleanup patterns
        cleanup_patterns = self.config.get('cleanup_patterns', {
            'ansi_codes': r'\x1b\[[0-9;]*[a-zA-Z]',
            'timestamps': r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*?\|',
            'multiple_newlines': r'\n\s*\n',
            'multiple_spaces': r'\s+',
            'debug_markers': r'DEBUG\s*\*+',
            'color_codes': r'\[[\d;]+m'
        })
        
        for pattern in cleanup_patterns.values():
            text = re.sub(pattern, ' ' if 'spaces' in pattern else '', text)
        
        return text.strip()

    def evaluate_all(self) -> Tuple[Dict[str, float], float]:
        """Evaluate all responses in the log file against ground truth."""
        with open(self.logs_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Split log content into interactions using configured pattern
        interaction_pattern = self.config['log_patterns'].get('interaction_split', 'NEW INTERACTION')
        interactions = [x for x in log_content.split(interaction_pattern) if x.strip()]
        
        evaluated_count = 0
        total_metrics = defaultdict(float)
        
        for interaction in interactions:
            log_entry = self.parse_log_entry(interaction)
            if not log_entry.get('query'):
                continue
            
            # Match with ground truth using configured matching method
            matching_config = self.config.get('ground_truth_matching', {
                'method': 'exact',
                'case_sensitive': False,
                'similarity_threshold': 0.9
            })
            
            if matching_config['method'] == 'exact':
                gt_entries = self.ground_truth[
                    self.ground_truth['Query'].str.lower().str.strip() == 
                    log_entry['query'].lower().strip()
                ].to_dict('records')
            else:  # fuzzy matching
                query = log_entry['query'].lower().strip()
                gt_entries = [
                    row for _, row in self.ground_truth.iterrows()
                    if fuzz.ratio(query, row['Query'].lower().strip()) >= 
                    (matching_config['similarity_threshold'] * 100)
                ]
            
            if gt_entries:
                metrics = self.evaluate_response(log_entry, gt_entries[0])
                for key, value in metrics.items():
                    total_metrics[key] += value
                evaluated_count += 1
        
        # Calculate averages and coverage
        coverage = evaluated_count / len(interactions) if interactions else 0
        
        # Average the metrics
        averaged_metrics = {
            key: value / evaluated_count if evaluated_count > 0 else 0
            for key, value in total_metrics.items()
        }
        
        return averaged_metrics, coverage


    def _calculate_routing_accuracy(self, actual_agents: str, expected_agents: str) -> float:
        """Calculate routing accuracy using fuzzy string matching."""
        if not actual_agents or not expected_agents:
            return 0.0
        
        actual_set = self._normalize_agents(actual_agents)
        expected_set = self._normalize_agents(expected_agents)
        
        if not actual_set or not expected_set:
            return 0.0
            
        # Calculate fuzzy match scores for each pair
        match_scores = []
        for actual in actual_set:
            scores = [fuzz.ratio(actual, expected) / 100.0 for expected in expected_set]
            match_scores.append(max(scores) if scores else 0.0)
            
        # Calculate weighted accuracy
        accuracy = sum(match_scores) / max(len(actual_set), len(expected_set))
        
        # Apply minimum threshold from config
        min_accuracy = self.config.get('evaluation_thresholds', {}).get('min_routing_accuracy', 0.0)
        return max(accuracy, min_accuracy)

    def _normalize_tool_name(self, tool_name: str) -> str:
        """Normalize tool name using configuration rules."""
        if not tool_name:
            return ''
            
        matching_config = self.config['tool_evaluation']['matching']
        normalized = tool_name.lower() if not matching_config['case_sensitive'] else tool_name
        
        # Strip configured prefixes and suffixes
        for prefix in matching_config['strip_prefixes']:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        
        for suffix in matching_config['strip_suffixes']:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)]
                
        return normalized.strip()

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using sentence transformers."""
        embeddings = self.sentence_model.encode([str1, str2])
        return 1 - cosine(embeddings[0], embeddings[1])

    def _calculate_tool_accuracy(self, actual_tools: List[Dict], expected_tools: List[str]) -> float:
        """Calculate tool usage accuracy using configured patterns and thresholds."""
        if not actual_tools or not expected_tools:
            return 0.0

        tool_config = self.config['tool_evaluation']
        match_threshold = tool_config['match_threshold']
        matching_config = tool_config['matching']
        
        correct_tools = 0
        total_tools = len(expected_tools)

        for expected_tool in expected_tools:
            expected_normalized = self._normalize_tool_name(expected_tool)
            
            for actual_tool in actual_tools:
                if not all(field in actual_tool.get('function', {}) for field in tool_config['required_fields']):
                    continue

                actual_name = actual_tool['function']['name']
                actual_normalized = self._normalize_tool_name(actual_name)
                
                # Calculate similarity score
                if matching_config['use_fuzzy_matching']:
                    similarity = fuzz.ratio(actual_normalized, expected_normalized) / 100.0
                else:
                    similarity = self._calculate_similarity(actual_normalized, expected_normalized)
                
                if similarity >= match_threshold:
                    correct_tools += 1
                    break

        return correct_tools / total_tools if total_tools > 0 else 0.0

