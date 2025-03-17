import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from evaluator import AgentEvaluator
import json
from pathlib import Path
import tempfile
import os
import sys
from judge_agent import GeminiJudgeAgent
from config import EvaluatorConfig
# Add these configurations before any Streamlit commands
os.environ['STREAMLIT_SERVER_WATCH_MODULES'] = 'false'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Disable watchdog for this session
if 'watchdog' in sys.modules:
    del sys.modules['watchdog']

# Set page configuration
st.set_page_config(
    page_title="MAESTRO - Multi-Agent System Evaluation Framework",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress {
        height: 20px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎯 MAESTRO")
st.subheader("Multi-Agent System Evaluation & Testing ROadmap")
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 10px;'>
    Comprehensive analytics and performance evaluation framework for multi-agent architectures. 
    Analyze agent interactions, measure response quality, and optimize system performance with precision.
    </div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Configuration")
    
    # Load default config
    config = EvaluatorConfig.DEFAULT_CONFIG.copy()
    
    # Create UI elements based on config
    custom_config = {
        'bleu_config': {
            'max_order': st.slider(
                "BLEU N-gram Order", 
                1, 4, 
                config['bleu_config']['max_order']
            )
        },
        'semantic_threshold': st.slider(
            "Semantic Threshold", 
            0.0, 1.0, 
            config['semantic_threshold']
        ),
        'routing_threshold': st.slider(
            "Routing Threshold", 
            0.0, 1.0, 
            config['routing_threshold']
        ),
        'response_quality_threshold': st.slider(
            "Response Quality Threshold", 
            0.0, 1.0, 
            config['response_quality_threshold']
        )
    }
    
    # Set BLEU weights based on max_order
    weights = [1/custom_config['bleu_config']['max_order']] * custom_config['bleu_config']['max_order']
    custom_config['bleu_config']['weights'] = tuple(weights)
    
    # Load final config with custom overrides
    final_config = EvaluatorConfig.load(custom_config)
    
    # Add Gemini API key input
    use_gemini_judge = st.checkbox("Use Gemini Judge Agent", True)
    gemini_api_key = st.text_input("Gemini API Key", type="password") if use_gemini_judge else None

# File upload section
col1, col2 = st.columns(2)
with col1:
    ground_truth_file = st.file_uploader("Upload Ground Truth CSV", type=['csv'])
with col2:
    log_file = st.file_uploader("Upload Log File", type=['log'])

# Update evaluation section
if ground_truth_file and log_file is not None:
    # Save uploaded files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_gt:
        tmp_gt.write(ground_truth_file.getvalue())
        gt_path = tmp_gt.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as tmp_log:
        tmp_log.write(log_file.getvalue())
        log_path = tmp_log.name
    
    # Read the CSV file into a DataFrame
    ground_truth_df = pd.read_csv(gt_path)
    
    # Initialize evaluator
    # Update evaluator initialization
    evaluator = AgentEvaluator(
        ground_truth_path=gt_path,
        logs_path=log_path,
        config=final_config  # Use the final config here
    )
    
    # Run evaluation
    with st.spinner("Analyzing agent performance..."):
        metrics, coverage = evaluator.evaluate_all()
    
    # Display overall metrics
    st.header("📈 Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Coverage", f"{coverage*100:.1f}%")
    with col2:
        overall_score = (metrics.get('routing_accuracy', 0) + 
                        metrics.get('semantic_similarity', 0) + 
                        metrics.get('tool_accuracy', 0)) / 3  # Include tool_accuracy in overall score
        st.metric("Overall Score", f"{overall_score:.2f}")
    with col3:
        st.metric("Routing Accuracy", f"{metrics.get('routing_accuracy', 0):.2f}")
    with col4:
        st.metric("Tool Accuracy", f"{metrics.get('tool_accuracy', 0):.2f}")
    
    # Detailed metrics visualization
    st.header("🔍 Detailed Analysis")
    
    # Create tabs for different metric categories
    tab1, tab2, tab3, tab4 = st.tabs(["Accuracy Metrics", "Text Similarity", "ROUGE Scores", "Classification Metrics"])
    
    with tab1:
        accuracy_metrics = {
            'Routing Accuracy': metrics.get('routing_accuracy', 0),
            'Planning Score': metrics.get('planning_score', 0),
            'Reasoning Score': metrics.get('reasoning_score', 0),
            'Tool Usage': metrics.get('tool_accuracy', 0)
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(accuracy_metrics.keys()),
                y=list(accuracy_metrics.values()),
                marker_color=['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
            )
        ])
        fig.update_layout(title="Accuracy Metrics", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        # Add detailed explanations
        st.markdown("""
        ### Accuracy Metrics Explained
        - **Routing Accuracy (0-1)**: Measures how well the system directs queries to appropriate agents
          - Formula: `correct_routings / total_routings`
          - Perfect score (1.0) means all queries were routed to correct agents
        
        - **Planning Score (0-1)**: Evaluates the quality of task planning
          - Considers: Task decomposition, step sequencing, and goal alignment
          - Higher scores indicate better structured plans
        
        - **Reasoning Score (0-1)**: Assesses logical flow and decision making
          - Evaluates: Inference quality, context understanding, and conclusion validity
          - Based on ground truth reasoning patterns
        
        - **Tool Usage (0-1)**: Measures effectiveness of tool utilization
          - Formula: `correct_tool_calls / total_tool_calls`
          - Considers both tool selection and parameter usage
        """)
    
    with tab2:
        similarity_metrics = {
            'BLEU Score': metrics.get('bleu', 0),
            'Semantic Similarity': metrics.get('semantic_similarity', 0),
            'Levenshtein Similarity': metrics.get('levenshtein_similarity', 0),
            'BERTScore': metrics.get('bert_score', 0)
        }
        
        fig = px.line_polar(
            r=list(similarity_metrics.values()),
            theta=list(similarity_metrics.keys()),
            line_close=True
        )
        fig.update_layout(title="Text Similarity Analysis")
        st.plotly_chart(fig, use_container_width=True)

        # Add explanation for Text Similarity
        st.markdown("""
        ### Text Similarity Metrics Explained
        - **BLEU Score (0-1)**: Bilingual Evaluation Understudy
          - Measures: N-gram overlap between response and reference
          - Formula: `Geometric mean of n-gram precisions × brevity penalty`
          - Used for: Evaluating text generation quality
        
        - **Semantic Similarity (0-1)**: Deep learning based meaning comparison
          - Uses: Sentence transformers (all-MiniLM-L6-v2)
          - Formula: `1 - cosine_distance(embedding1, embedding2)`
          - Captures: Meaning similarity regardless of exact wording
        
        - **Levenshtein Similarity (0-1)**: Edit distance based comparison
          - Formula: `1 - (levenshtein_distance / max_length)`
          - Measures: Character-level differences
          - Useful for: Exact matching and typo detection
        
        - **BERTScore (0-1)**: Contextual embedding similarity
          - Uses: RoBERTa model for token alignment
          - Considers: Token importance and context
          - Better than: Traditional exact match metrics
        """)
    
    with tab3:
        rouge_metrics = {
            'ROUGE-1': metrics.get('rouge1', 0),
            'ROUGE-2': metrics.get('rouge2', 0),
            'ROUGE-L': metrics.get('rougeL', 0)
        }
        
        fig = go.Figure(data=[
            go.Scatter(
                x=list(rouge_metrics.keys()),
                y=list(rouge_metrics.values()),
                mode='lines+markers',
                fill='tozeroy'
            )
        ])
        fig.update_layout(title="ROUGE Scores", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Add explanation for ROUGE Scores
        st.markdown("""
        ### ROUGE Scores Explained
        - **ROUGE-1 (0-1)**: Unigram Overlap
          - Measures: Single word overlap
          - Formula: `matching_words / total_words`
          - Best for: Content coverage evaluation
        
        - **ROUGE-2 (0-1)**: Bigram Overlap
          - Measures: Two-word phrase matches
          - Formula: `matching_bigrams / total_bigrams`
          - Best for: Fluency evaluation
        
        - **ROUGE-L (0-1)**: Longest Common Subsequence
          - Measures: Longest matching word sequence
          - Formula: `LCS_length / reference_length`
          - Best for: Capturing word order and structure
        
        #### Interpretation:
        - Higher scores indicate better match with ground truth
        - ROUGE-1: Content coverage
        - ROUGE-2: Phrase accuracy
        - ROUGE-L: Sequence matching
        """)
    
    with tab4:
        classification_metrics = {
            'Agent Precision': metrics.get('agent_precision', 0),
            'Agent Recall': metrics.get('agent_recall', 0),
            'Agent F1 Score': metrics.get('agent_f1', 0)
        }
        
        # Create a radar chart for classification metrics
        fig = px.line_polar(
            r=list(classification_metrics.values()),
            theta=list(classification_metrics.keys()),
            line_close=True,
            range_r=[0, 1]
        )
        fig.update_layout(
            title="Agent Classification Performance",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add detailed explanations
        st.markdown("""
        ### Classification Metrics Explained
        - **Precision**: How many of the selected agents were correct
        - **Recall**: How many of the correct agents were selected
        - **F1 Score**: Balanced measure between precision and recall

        - **The metrics tell us:

            1. Precision (0.000) : Of the agents that were selected, none were correct
               - Formula: True Positives / (True Positives + False Positives)

            2. Recall (0.000) : Of the agents that should have been selected, none were actually selected
               - Formula: True Positives / (True Positives + False Negatives)

            3. F1 Score (0.000) : The harmonic mean of precision and recall is 0, indicating poor performance
               - Formula: 2 * (Precision * Recall) / (Precision + Recall)
            
            The negative deltas (-0.700) show we're comparing against a threshold of 0.7, which is considered a good performance baseline.
        """)
        
        # Add a metrics table
        st.markdown("### Detailed Scores")
        for metric, value in classification_metrics.items():
            st.metric(
                label=metric,
                value=f"{value:.3f}",
                delta=f"{(value - 0.7):.3f}" if value > 0.7 else f"{(value - 0.7):.3f}"
            )
    
    # Add Gemini Judge evaluation
    if use_gemini_judge and gemini_api_key:
        st.header("🧠 AI Judge Analysis")
        
        with st.spinner("Running AI evaluation..."):
            # Initialize the judge agent
            judge = GeminiJudgeAgent(api_key=gemini_api_key)
            
            # Run analysis with the DataFrame
            analysis = judge.analyze_logs_and_truth(ground_truth_df, log_path)
            
            if analysis:
                # Display metrics in a more visual way
                st.subheader("📊 Performance Metrics")
                
                # Tool Usage Metrics
                if 'metrics' in analysis and 'tool_usage' in analysis['metrics']:
                    tool_col1, tool_col2, tool_col3 = st.columns(3)
                    with tool_col1:
                        st.metric("Tool Accuracy", f"{analysis['metrics']['tool_usage'].get('accuracy', 0):.2f}")
                    with tool_col2:
                        st.metric("Tool Efficiency", f"{analysis['metrics']['tool_usage'].get('efficiency', 0):.2f}")
                    with tool_col3:
                        st.metric("Tool Appropriateness", f"{analysis['metrics']['tool_usage'].get('appropriateness', 0):.2f}")

                # Reasoning Metrics
                if 'metrics' in analysis and 'reasoning' in analysis['metrics']:
                    reason_col1, reason_col2, reason_col3 = st.columns(3)
                    with reason_col1:
                        st.metric("Logical Flow", f"{analysis['metrics']['reasoning'].get('logical_flow', 0):.2f}")
                    with reason_col2:
                        st.metric("Reasoning Depth", f"{analysis['metrics']['reasoning'].get('depth', 0):.2f}")
                    with reason_col3:
                        st.metric("Coherence", f"{analysis['metrics']['reasoning'].get('coherence', 0):.2f}")

                # Execution Metrics
                if 'metrics' in analysis and 'execution' in analysis['metrics']:
                    exec_col1, exec_col2, exec_col3 = st.columns(3)
                    with exec_col1:
                        st.metric("Task Completion", f"{analysis['metrics']['execution'].get('task_completion', 0):.2f}")
                    with exec_col2:
                        st.metric("Response Quality", f"{analysis['metrics']['execution'].get('response_quality', 0):.2f}")
                    with exec_col3:
                        st.metric("Routing Accuracy", f"{analysis['metrics']['execution'].get('routing_accuracy', 0):.2f}")

                # Detailed Analysis
                if 'detailed_analysis' in analysis:
                    st.subheader("📋 Analysis Summary")
                    
                    # Key Patterns
                    if 'key_patterns' in analysis['detailed_analysis']:
                        st.markdown("#### 🔍 Key Patterns")
                        for pattern in analysis['detailed_analysis']['key_patterns']:
                            st.markdown(f"- {pattern}")
                    
                    # Strengths
                    if 'strengths' in analysis['detailed_analysis']:
                        st.markdown("#### ✅ Strengths")
                        for strength in analysis['detailed_analysis']['strengths']:
                            st.markdown(f"- {strength}")
                    
                    # Weaknesses
                    if 'weaknesses' in analysis['detailed_analysis']:
                        st.markdown("#### ⚠️ Areas for Improvement")
                        for weakness in analysis['detailed_analysis']['weaknesses']:
                            st.markdown(f"- {weakness}")
                    
                    # Recommendations
                    if 'recommendations' in analysis['detailed_analysis']:
                        st.markdown("#### 💡 Recommendations")
                        for rec in analysis['detailed_analysis']['recommendations']:
                            st.markdown(f"- {rec}")
                
                # Overall Assessment
                if 'overall_assessment' in analysis:
                    st.subheader("📝 Overall Assessment")
                    st.write(analysis['overall_assessment'])
            else:
                st.error("No analysis results were returned. Please check the logs for errors.")
            # Display summary first
            if 'detailed_summary' in metrics:
                summary = metrics['detailed_summary']
                
                # Overall Assessment
                st.markdown("#### Overall Assessment")
                st.write(summary.get('overall_assessment', ''))
                
                # Strengths
                st.markdown("#### ✅ Strengths")
                for strength in summary.get('strengths', []):
                    st.markdown(f"- {strength}")
                
                # Weaknesses
                st.markdown("#### ⚠️ Areas for Improvement")
                for weakness in summary.get('weaknesses', []):
                    st.markdown(f"- {weakness}")
                
                # Recommendations
                st.markdown("#### 💡 Recommendations")
                for rec in summary.get('recommendations', []):
                    st.markdown(f"- {rec}")
            
    # Analysis summary
    st.header("💡 Analysis Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Strengths")
        strengths = []
        if metrics.get('routing_accuracy', 0) > 0.7:
            strengths.append("✅ Strong routing performance")
        if metrics.get('semantic_similarity', 0) > 0.7:
            strengths.append("✅ High semantic understanding")
        if metrics.get('bert_score', 0) > 0.7:
            strengths.append("✅ Strong response quality")
        
        for strength in strengths or ["No significant strengths identified"]:
            st.markdown(strength)
    
    with col2:
        st.markdown("### Areas for Improvement")
        improvements = []
        if metrics.get('routing_accuracy', 0) < 0.7:
            improvements.append("⚠️ Routing accuracy needs improvement")
        if metrics.get('tool_accuracy', 0) < 0.7:
            improvements.append("⚠️ Tool usage could be optimized")
        if metrics.get('semantic_similarity', 0) < 0.7:
            improvements.append("⚠️ Semantic understanding could be enhanced")
        
        for improvement in improvements or ["No significant areas for improvement"]:
            st.markdown(improvement)
    
    # Download results
    st.header("📥 Export Results")
    
    # Convert metrics to JSON serializable format
    def convert_to_native_types(obj):
        if isinstance(obj, dict):
            return {key: convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # For NumPy types
            return obj.item()
        elif hasattr(obj, '__float__'):  # For other numeric types
            return float(obj)
        return obj

    results_dict = {
        "metrics": convert_to_native_types(metrics),
        "coverage": float(coverage),
        "config": convert_to_native_types(custom_config),
        "timestamp": str(pd.Timestamp.now())
    }
    
    st.download_button(
        label="Download Results as JSON",
        data=json.dumps(results_dict, indent=2),
        file_name="maestro_evaluation_results.json",
        mime="application/json"
    )

else:
    # Display welcome message and instructions
    st.info("""
        👋 Welcome to MAESTRO!
        
        To get started:
        1. Upload your ground truth CSV file
        2. Upload your log file
        3. Adjust the configuration parameters in the sidebar if needed
        
        The dashboard will automatically analyze your agent's performance and
        provide detailed metrics and visualizations.
    """)
    
  
    
