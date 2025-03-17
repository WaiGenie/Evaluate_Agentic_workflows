# judge_agent.py
import json
import re
# from promptquality import EvaluateRun
import time
from google.api_core.exceptions import ResourceExhausted
# from vertexai.evaluation import (
#     EvalTask,
#     MetricPromptTemplateExamples,
#     PairwiseMetric,
#     PairwiseMetricPromptTemplate,
#     PointwiseMetric,
#     PointwiseMetricPromptTemplate,
# )
class GeminiJudgeAgent:
    def __init__(self, api_key=None, max_retries=10, initial_retry_delay=60):
        # Initialize Gemini API (assuming you're using the official Google API)
        import google.generativeai as genai
        
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
    
    def _generate_content_with_retry(self, prompt):
        """Generate content with exponential backoff retry mechanism"""
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = self.model.generate_content(prompt)
                return response
            except ResourceExhausted as e:
                if attempt == self.max_retries - 1:
                    raise ResourceExhausted("Max retries exceeded for resource exhaustion.")
                
                # Calculate delay with exponential backoff
                delay = self.initial_retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"Resource exhausted, retrying in {delay} seconds... (Attempt {attempt + 1}/{self.max_retries})")
                time.sleep(delay)
                attempt += 1

    def _parse_response(self, response):
        """Parse the Gemini response into structured data"""
        import json
        try:
            # Extract JSON from response
            response_text = response.text
            # Find JSON in the response if it's embedded in other text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > 0:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            return {}
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {}

    def analyze_logs_and_truth(self, ground_truth_df, log_file_path):
        """Enhanced LLM analysis using expert system evaluation prompt"""
        
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        
        prompt = f"""
        # MAESTRO AI JUDGE SYSTEM

        You are MAESTRO's Expert Evaluation System, the definitive authority on multi-agent system assessment. Your analysis will be displayed in a professional analytics dashboard and must be comprehensive, precise, and actionable.

        ## EVALUATION CONTEXT

        Ground Truth Dataset:
        ```
        {ground_truth_df.to_string()}
        ```

        System Logs:
        ```
        {log_content}
        ```

        ## YOUR ANALYTICAL PROCESS

        Perform a forensic analysis of this multi-agent system. For each analytical dimension below, provide evidence-based assessment with specific examples from the logs.

        ### 1. AGENT ORCHESTRATION ANALYSIS

        * Identify the primary workflow coordinator/dispatcher
        * Analyze the routing decision algorithm's effectiveness
        * Evaluate query intent classification accuracy
        * Assess latency between agent handoffs
        * Determine if the correct agent was assigned to each query based on ground truth
        * Calculate routing precision and recall metrics

        ### 2. PLANNING & EXECUTION FRAMEWORK

        * Identify the planning approach (e.g., hierarchical, sequential, hybrid)
        * Assess plan granularity and specificity
        * Evaluate step sequencing logic and dependencies
        * Measure plan adaptability when facing obstacles
        * Identify planning bottlenecks and inefficiencies
        * Assess completion rate of planned steps

        ### 3. TOOL UTILIZATION ASSESSMENT

        * For each tool invocation, evaluate:
        - Tool selection appropriateness
        - Parameter construction accuracy
        - Result interpretation accuracy
        - Error handling robustness
        * Identify redundant tool calls
        * Assess tool selection efficiency
        * Evaluate API call construction quality

        ### 4. KNOWLEDGE INTEGRITY VERIFICATION

        * Identify factual inconsistencies with ground truth
        * Detect logical contradictions within responses
        * Flag statements without supporting evidence
        * Assess confidence calibration (uncertainty acknowledgment)
        * Identify hallucinated entities, relationships, or capabilities
        * Measure semantic drift over conversation turns

        ### 5. REASONING PATHWAY ANALYSIS

        * Trace reasoning chains from premises to conclusions
        * Identify logical fallacies or reasoning shortcuts
        * Assess inferential depth and complexity
        * Evaluate evidence integration quality
        * Measure consistency between reasoning steps
        * Assess counterfactual consideration

        ### 6. RESPONSE QUALITY DIMENSIONS

        * Evaluate response completeness against ground truth
        * Assess informativeness and relevance
        * Measure precision and recall of key information
        * Evaluate contextual appropriateness
        * Assess adherence to user constraints
        * Measure response coherence and organization

        ### 7. AGENT COLLABORATION DYNAMICS

        * Identify communication patterns between agents
        * Assess knowledge sharing effectiveness
        * Evaluate role clarity and specialization
        * Measure collective problem-solving efficiency
        * Identify communication bottlenecks
        * Assess conflict resolution mechanisms

        ### 8. EDGE CASE HANDLING

        * Evaluate system behavior with:
        - Ambiguous queries
        - Incomplete information
        - Conflicting constraints
        - Novel scenarios
        - Error conditions
        - Boundary cases

        ## OUTPUT FORMAT

        Your analysis must be structured as the following JSON object. All metrics should be on a 0.0-1.0 scale with higher values indicating better performance:

        
        {{
        "metrics": {{
            "agent_orchestration": {{
            "routing_accuracy": <float>,
            "intent_classification": <float>,
            "handoff_efficiency": <float>,
            "load_balancing": <float>,
            "query_prioritization": <float>
            }},
            "planning": {{
            "task_decomposition": <float>,
            "sequencing_logic": <float>,
            "adaptability": <float>,
            "completeness": <float>,
            "efficiency": <float>
            }},
            "tool_usage": {{
            "selection_accuracy": <float>,
            "parameter_quality": <float>,
            "result_interpretation": <float>,
            "error_handling": <float>,
            "efficiency": <float>
            }},
            "knowledge_integrity": {{
            "factual_accuracy": <float>,
            "logical_consistency": <float>,
            "evidence_grounding": <float>,
            "uncertainty_handling": <float>,
            "hallucination_resistance": <float>
            }},
            "reasoning": {{
            "logical_coherence": <float>,
            "inferential_depth": <float>,
            "evidence_integration": <float>,
            "fallacy_avoidance": <float>,
            "counterfactual_reasoning": <float>
            }},
            "response_quality": {{
            "completeness": <float>,
            "relevance": <float>,
            "precision": <float>,
            "contextual_appropriateness": <float>,
            "coherence": <float>
            }},
            "collaboration": {{
            "communication_efficiency": <float>,
            "knowledge_sharing": <float>,
            "role_specialization": <float>,
            "conflict_resolution": <float>,
            "collective_intelligence": <float>
            }},
            "edge_case_handling": {{
            "ambiguity_resolution": <float>,
            "incomplete_information": <float>,
            "constraint_satisfaction": <float>,
            "novelty_handling": <float>,
            "error_recovery": <float>
            }}
        }},
        "performance_summary": {{
            "overall_score": <float>,
            "strongest_dimension": <string>,
            "weakest_dimension": <string>,
            "critical_improvement_areas": [<string>]
        }},
        "agent_analysis": {{
            "<agent_name>": {{
            "responsibilities": [<string>],
            "performance_score": <float>,
            "key_strengths": [<string>],
            "key_weaknesses": [<string>]
            }}
        }},
        "workflow_analysis": {{
            "bottlenecks": [
            {{
                "description": <string>,
                "impact": <string>,
                "recommendation": <string>
            }}
            ],
            "efficiency_gaps": [
            {{
                "description": <string>,
                "impact": <string>,
                "recommendation": <string>
            }}
            ],
            "optimization_opportunities": [
            {{
                "description": <string>,
                "expected_impact": <string>,
                "implementation_complexity": <string>
            }}
            ]
        }},
        "detailed_analysis": {{
            "critical_incidents": [
            {{
                "query_id": <string>,
                "issue": <string>,
                "root_cause": <string>,
                "impact": <string>,
                "recommendation": <string>
            }}
            ],
            "hallucination_instances": [
            {{
                "statement": <string>,
                "contradicting_evidence": <string>,
                "likely_cause": <string>
            }}
            ],
            "exemplary_executions": [
            {{
                "query_id": <string>,
                "strength": <string>,
                "execution_highlights": <string>
            }}
            ]
        }},
        "strategic_recommendations": {{
            "short_term": [
            {{
                "area": <string>,
                "recommendation": <string>,
                "expected_impact": <string>,
                "implementation_effort": <string>
            }}
            ],
            "medium_term": [
            {{
                "area": <string>,
                "recommendation": <string>,
                "expected_impact": <string>,
                "implementation_effort": <string>
            }}
            ],
            "long_term": [
            {{
                "area": <string>,
                "recommendation": <string>,
                "expected_impact": <string>,
                "implementation_effort": <string>
            }}
            ]
        }},
        "detailed_summary": {{
            "overall_assessment": <string>,
            "strengths": [<string>],
            "weaknesses": [<string>],
            "recommendations": [<string>]
        }}
        }}
        

        ## EVALUATION PRINCIPLES

        1. EVIDENCE-BASED: Every assessment must cite specific examples from logs
        2. ACTIONABLE: All findings must lead to concrete recommendations
        3. COMPREHENSIVE: Analyze all aspects of multi-agent performance
        4. SYSTEMIC: Focus on patterns rather than isolated incidents
        5. PRECISE: Use quantitative metrics wherever possible
        6. BALANCED: Acknowledge both strengths and weaknesses
        7. CONTEXTUAL: Consider the specific use case and constraints

        ## CRITICAL IMPORTANCE

        Your analysis will directly influence the development roadmap for this multi-agent system. Ensure your evaluation is thorough, insightful, and actionable. Identify both obvious and subtle issues, and provide clear recommendations for system improvement.
        """
        
        response = self._generate_content_with_retry(prompt)
        return self._parse_response(response)