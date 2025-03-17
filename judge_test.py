# judge_agent.py
import json
import re

import time
from google.api_core.exceptions import ResourceExhausted

class GeminiJudgeAgent:
    def __init__(self, api_key=None, max_retries=10, initial_retry_delay=60):
        # Initialize Gemini API (assuming you're using the official Google API)
        import google.generativeai as genai
        
        if api_key:
            genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
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
        """Enhanced LLM analysis using chain-of-thought prompting"""
        
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        
        prompt = f"""
        You are an expert AI system evaluator analyzing multi-agent interactions. 
        Let's approach this analysis step by step.

        Given Data:
        1. Ground Truth Dataset:
        {ground_truth_df.to_string()}

        2. System Logs:
        {log_content}

        Think through this analysis in the following steps:

        Step 1: Initial Data Assessment
        - What patterns do you observe in the interactions?
        - How do the actual queries align with ground truth expectations?
        - What tools and agents are being utilized?

        Step 2: Behavioral Analysis
        - How does the system handle task routing?
        - What is the quality of tool selection and usage?
        - How well does the system follow reasoning patterns?

        Step 3: Performance Evaluation
        - Compare actual responses against expected outcomes
        - Assess the efficiency of task completion
        - Evaluate the coherence of system behavior

        Step 4: Critical Analysis
        - Identify key strengths in the system
        - Pinpoint areas needing improvement
        - Formulate specific recommendations

        Based on this analysis, provide a structured evaluation in this JSON format:
        {{
            "metrics": {{
                "tool_usage": {{
                    "accuracy": <float>,
                    "efficiency": <float>,
                    "appropriateness": <float>
                }},
                "reasoning": {{
                    "logical_flow": <float>,
                    "depth": <float>,
                    "coherence": <float>
                }},
                "execution": {{
                    "task_completion": <float>,
                    "response_quality": <float>,
                    "routing_accuracy": <float>
                }}
            }},
            "detailed_analysis": {{
                "key_patterns": [
                    "List observed interaction patterns"
                ],
                "strengths": [
                    "List identified strengths with specific examples"
                ],
                "weaknesses": [
                    "List areas for improvement with specific examples"
                ],
                "recommendations": [
                    "Provide actionable recommendations"
                ]
            }},
            "overall_assessment": "Provide a comprehensive evaluation that synthesizes all observations"
        }}

        Remember to:
        1. Base all scores on concrete evidence from the logs
        2. Provide specific examples for each observation
        3. Consider the full context of interactions
        4. Focus on actionable insights
        """
        
        response = self._generate_content_with_retry(prompt)
        return self._parse_response(response)
