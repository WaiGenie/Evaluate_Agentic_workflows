from typing import Dict, Any

class EvaluatorConfig:
    """Configuration class for AgentEvaluator."""
    
    DEFAULT_CONFIG: Dict[str, Any] = {
        'rouge_metrics': ['rouge1', 'rouge2', 'rougeL'],
        'semantic_threshold': 0.7,
        'routing_threshold': 0.8,
        'response_quality_threshold': 0.6,
        'bleu_config': {
            'weights': (0.25, 0.25, 0.25, 0.25),  # 4-gram weights
            'smoothing_function': True,  # Enable smoothing
            'max_order': 2  # Limit to bigrams to avoid warnings
        },
        'ground_truth_matching': {
            'method': 'exact',  # or 'fuzzy'
            'case_sensitive': False,
            'similarity_threshold': 0.9
        },
        'log_patterns': {
            'query': r'QUERY:\s*(.*?)(?=\n|$)',
            'response': r'RESPONSE:\s*(.*?)(?=\n\d{4}-\d{2}-\d{2}|$)',
            'assistant': r'ASSISTANT:\s*(.*?)(?=\n\d{4}-\d{2}-\d{2}|$)',
            'assigned_agents': r'Included function transfer_task_to_(\w+)_teacher',
            'agent_id': r'Agent ID:\s*\[93m(.*?)\s*\*',
            'session_id': r'Session ID:\s*\[93m(.*?)\s*\*',
            'agent_run': r'Agent Run Start:\s*\[93m(.*?)\s*\*',
            'tool_calls': {
                'block_start': r'Tool Calls:\s*\[1m\[',
                'block_end': r'\](?=\n)',
                'function_call': {
                    'type': r'"type":\s*"([^"]+)"',
                    'name': r'"name":\s*"([^"]+)"',
                    'arguments': r'"arguments":\s*"({.*?})"'
                }
            },
            'tool_function': r'"name":\s*"(transfer_task_to_\w+_teacher)"',
            'tool_arguments': r'"arguments":\s*"({.*?})"',
            'tool_execution_status': r'Running:\s*\n\s*-\s*(transfer_task_to_\w+_teacher)',
            'timestamp': r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            'interaction_split': 'NEW INTERACTION',
            'metrics': {
                'input_tokens': r'Input tokens:\s*\[1;36m(\d+)',
                'output_tokens': r'Output tokens:\s*\[1;36m(\d+)',
                'total_tokens': r'Total tokens:\s*\[1;36m(\d+)',
                'time': r'Time:\s*\[1;36m([\d.]+)s',
                'tokens_per_second': r'Tokens per second:\s*\[1;36m([\d.]+)',
                'time_to_first_token': r'Time to first token:\s*\[1;36m([\d.]+)s'
            }
        },
        'model_config': {
            'sentence_transformer': 'all-MiniLM-L6-v2',
            'bert_score_model': 'roberta-large'
        },

        'cleanup_patterns': {
            'ansi_codes': r'\x1b\[[0-9;]*[a-zA-Z]',
            'timestamps': r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*?\|',
            'multiple_newlines': r'\n\s*\n',
            'multiple_spaces': r'\s+',
            'debug_markers': r'DEBUG\s*\*+',
            'color_codes': r'\[[\d;]+m'
        },
        'report_format': {
            'title': "Agent Evaluation Report",
            'sections': {
                'overview': {
                    'title': "Overview",
                    'metrics': ['coverage', 'overall_performance']
                },
                'routing': {
                    'title': "Routing and Planning Metrics",
                    'metrics': ['routing_accuracy', 'planning_score', 'reasoning_score', 'tool_accuracy']
                },
                'similarity': {
                    'title': "Text Similarity Metrics",
                    'metrics': ['bleu', 'semantic_similarity', 'levenshtein_similarity', 'bert_score']
                },
                'rouge': {
                    'title': "ROUGE Scores",
                    'metrics': ['rouge1', 'rouge2', 'rougeL']
                }
            }
        },
        'analysis_thresholds': {
            'routing': {
                'threshold': 0.7,
                'success_message': "✓ Strong agent routing performance",
                'improvement_message': "⚠ Agent routing needs improvement"
            },
            'semantic': {
                'threshold': 0.7,
                'success_message': "✓ Good semantic understanding",
                'improvement_message': "⚠ Semantic understanding could be improved"
            },
            'response_quality': {
                'threshold': 0.6,
                'success_message': "✓ High-quality response generation",
                'improvement_message': "⚠ Response quality needs improvement"
            }
        },
        'metric_patterns': {
            'input_tokens': r'Input tokens:\s*(\d+)',
            'output_tokens': r'Output tokens:\s*(\d+)',
            'response_time': r'Response time:\s*([\d.]+)s'
        },
        'normalization': {
            'agent_replacements': [
                ('_teacher', ''),
                (' teacher', ''),
                (' ', '_')
            ],
            'separators': ['+', ',', '|']
        },
        'evaluation': {
            'text_preprocessing': {
                'lowercase': True,
                'remove_punctuation': True,
                'remove_extra_whitespace': True
            },
            'similarity_weights': {
                'bleu': 0.25,
                'semantic': 0.35,
                'rouge': 0.25,
                'bert': 0.15
            },
            'evaluation_thresholds': {
                'min_routing_accuracy': 0.0,
                'min_tool_accuracy': 0.0,
                'fuzzy_match_threshold': 80  # minimum fuzzy match score (0-100)
            },
            'fuzzy_matching': {
                'use_token_sort': True,  # use token_sort_ratio instead of basic ratio
                'case_sensitive': False
            },
            'tool_evaluation': {
                'match_threshold': 0.8,
                'argument_similarity_threshold': 0.7,
                'required_fields': ['name', 'arguments'],
                'matching': {
                    'case_sensitive': False,
                    'strip_prefixes': ['transfer_task_to_', 'function_'],
                    'strip_suffixes': ['_teacher', '_agent'],
                    'use_fuzzy_matching': True,
                    'fuzzy_threshold': 85  # minimum fuzzy match score (0-100)
                }
            },
        }
    }
    
    @classmethod
    def load(cls, custom_config: Dict = None) -> Dict:
        """Load configuration with custom overrides."""
        config = cls.DEFAULT_CONFIG.copy()
        if custom_config:
            cls._deep_update(config, custom_config)
        return config
    
    @classmethod
    def _deep_update(cls, base: Dict, update: Dict) -> None:
        """Recursively update nested dictionary."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                cls._deep_update(base[key], value)
            else:
                base[key] = value