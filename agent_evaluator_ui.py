import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.evaluator import AgentEvaluator
import json
from pathlib import Path
import tempfile
import os
import sys
from src.judge_agent import GeminiJudgeAgent
from src.config import EvaluatorConfig
# Add these configurations before any Streamlit commands
os.environ['STREAMLIT_SERVER_WATCH_MODULES'] = 'false'
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# Disable watchdog for this session
if 'watchdog' in sys.modules:
    del sys.modules['watchdog']

# Set page configuration
st.set_page_config(
    page_title="MAESTRO - Multi-Agent System Evaluation Framework",
    page_icon="🔄",
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
st.subheader("Multi-Agent System Evaluation & Validation Operations Repository")
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 10px;'>
    Advanced analytics and performance evaluation framework for multi-agent architectures. 
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
    
    # Add Gemini Judge evaluation
    if use_gemini_judge and gemini_api_key:
        st.header("🧠 AI Judge Analysis")
        
        with st.spinner("Running comprehensive AI evaluation..."):
            # Initialize the judge agent
            judge = GeminiJudgeAgent(api_key=gemini_api_key)
            
            # Run analysis with the DataFrame
            analysis = judge.analyze_logs_and_truth(ground_truth_df, log_path)
            
            if analysis:
                # Create tabs for different analysis categories
                judge_tabs = st.tabs([
                    "📊 Performance Overview", 
                    "🔍 Agent Analysis", 
                    "⚙️ Workflow Analysis", 
                    "🧩 Detailed Insights",
                    "📈 Strategic Roadmap"
                ])
                
                # Performance Overview Tab
                with judge_tabs[0]:
                    st.subheader("System Performance Summary")
                    
                    # Display overall score with a gauge
                    if "performance_summary" in analysis and "overall_score" in analysis["performance_summary"]:
                        overall_score = analysis["performance_summary"]["overall_score"]
                        
                        # Create a gauge chart for overall score
                        import plotly.graph_objects as go
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=overall_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Overall System Score"},
                            gauge={
                                'axis': {'range': [0, 1]},
                                'bar': {'color': "#1f77b4"},
                                'steps': [
                                    {'range': [0, 0.33], 'color': "#FF4136"},
                                    {'range': [0.33, 0.66], 'color': "#FFDC00"},
                                    {'range': [0.66, 1], 'color': "#2ECC40"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show strongest and weakest dimensions
                    if "performance_summary" in analysis:
                        summary = analysis["performance_summary"]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(f"**Strongest Dimension:** {summary.get('strongest_dimension', 'N/A')}")
                        with col2:
                            st.warning(f"**Weakest Dimension:** {summary.get('weakest_dimension', 'N/A')}")
                    
                    # Display metrics in a radar chart
                    if "metrics" in analysis:
                        st.subheader("Dimensional Performance")
                        
                        # Prepare data for radar chart
                        dimensions = []
                        scores = []
                        
                        for dimension, metrics in analysis["metrics"].items():
                            # Calculate average score for each dimension
                            if isinstance(metrics, dict) and metrics:
                                avg_score = sum(metrics.values()) / len(metrics)
                                dimensions.append(dimension.replace("_", " ").title())
                                scores.append(avg_score)
                        
                        if dimensions and scores:
                            # Create radar chart
                            fig = go.Figure()
                            
                            fig.add_trace(go.Scatterpolar(
                                r=scores,
                                theta=dimensions,
                                fill='toself',
                                name='System Performance'
                            ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 1]
                                    )
                                ),
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Critical improvement areas
                    if "performance_summary" in analysis and "critical_improvement_areas" in analysis["performance_summary"]:
                        st.subheader("Critical Improvement Areas")
                        for area in analysis["performance_summary"]["critical_improvement_areas"]:
                            st.markdown(f"- {area}")
                    
                    # Show detailed metrics as expandable sections
                    if "metrics" in analysis:
                        st.subheader("Detailed Metrics")
                        
                        for dimension, metrics in analysis["metrics"].items():
                            if isinstance(metrics, dict) and metrics:
                                with st.expander(f"{dimension.replace('_', ' ').title()} Metrics"):
                                    # Create columns for metrics
                                    cols = st.columns(3)
                                    for i, (metric, value) in enumerate(metrics.items()):
                                        with cols[i % 3]:
                                            # Determine color based on value
                                            color = "green" if value >= 0.7 else "orange" if value >= 0.4 else "red"
                                            st.markdown(f"""
                                            <div style="background-color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                                                <p style="margin-bottom: 0;">{metric.replace('_', ' ').title()}</p>
                                                <h3 style="color: {color}; margin-top: 0;">{value:.2f}</h3>
                                            </div>
                                            """, unsafe_allow_html=True)
                
                # Agent Analysis Tab
                with judge_tabs[1]:
                    if "agent_analysis" in analysis:
                        st.subheader("Agent Performance Analysis")
                        
                        # Create tabs for each agent
                        if analysis["agent_analysis"]:
                            agent_names = list(analysis["agent_analysis"].keys())
                            agent_tabs = st.tabs(agent_names)
                            
                            for i, agent_name in enumerate(agent_names):
                                agent_data = analysis["agent_analysis"][agent_name]
                                with agent_tabs[i]:
                                    # Agent score
                                    st.metric("Performance Score", f"{agent_data.get('performance_score', 0):.2f}")
                                    
                                    # Agent responsibilities
                                    if "responsibilities" in agent_data:
                                        st.subheader("Responsibilities")
                                        for resp in agent_data["responsibilities"]:
                                            st.markdown(f"- {resp}")
                                    
                                    # Strengths and weaknesses
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.subheader("🌟 Strengths")
                                        for strength in agent_data.get("key_strengths", []):
                                            st.markdown(f"- {strength}")
                                    
                                    with col2:
                                        st.subheader("🔧 Areas for Improvement")
                                    for weakness in agent_data.get("key_weaknesses", []):
                                        st.markdown(f"- {weakness}")
                    else:
                        st.info("No agent-specific analysis available")
            
            # Workflow Analysis Tab
            with judge_tabs[2]:
                st.subheader("Workflow Analysis")
                
                if "workflow_analysis" in analysis:
                    workflow = analysis["workflow_analysis"]
                    
                    # Bottlenecks section
                    if "bottlenecks" in workflow and workflow["bottlenecks"]:
                        st.subheader("🚧 System Bottlenecks")
                        for i, bottleneck in enumerate(workflow["bottlenecks"]):
                            with st.expander(f"Bottleneck {i+1}: {bottleneck.get('description', 'N/A')}"):
                                st.markdown(f"**Impact:** {bottleneck.get('impact', 'N/A')}")
                                st.markdown(f"**Recommendation:** {bottleneck.get('recommendation', 'N/A')}")
                    
                    # Efficiency gaps section
                    if "efficiency_gaps" in workflow and workflow["efficiency_gaps"]:
                        st.subheader("⚠️ Efficiency Gaps")
                        for i, gap in enumerate(workflow["efficiency_gaps"]):
                            with st.expander(f"Gap {i+1}: {gap.get('description', 'N/A')}"):
                                st.markdown(f"**Impact:** {gap.get('impact', 'N/A')}")
                                st.markdown(f"**Recommendation:** {gap.get('recommendation', 'N/A')}")
                    
                    # Optimization opportunities section
                    if "optimization_opportunities" in workflow and workflow["optimization_opportunities"]:
                        st.subheader("💡 Optimization Opportunities")
                        for i, opp in enumerate(workflow["optimization_opportunities"]):
                            with st.expander(f"Opportunity {i+1}: {opp.get('description', 'N/A')}"):
                                st.markdown(f"**Expected Impact:** {opp.get('expected_impact', 'N/A')}")
                                st.markdown(f"**Implementation Complexity:** {opp.get('implementation_complexity', 'N/A')}")
                else:
                    st.info("No workflow analysis available")
            
            # Detailed Insights Tab
            with judge_tabs[3]:
                if "detailed_analysis" in analysis:
                    detailed = analysis["detailed_analysis"]
                    
                    # Create columns for different types of insights
                    insight_tabs = st.tabs(["Critical Incidents", "Hallucination Detection", "Exemplary Executions"])
                    
                    # Critical incidents
                    with insight_tabs[0]:
                        if "critical_incidents" in detailed and detailed["critical_incidents"]:
                            for i, incident in enumerate(detailed["critical_incidents"]):
                                with st.expander(f"Incident {i+1}: Query {incident.get('query_id', 'Unknown')}"):
                                    st.markdown(f"**Issue:** {incident.get('issue', 'N/A')}")
                                    st.markdown(f"**Root Cause:** {incident.get('root_cause', 'N/A')}")
                                    st.markdown(f"**Impact:** {incident.get('impact', 'N/A')}")
                                    st.markdown(f"**Recommendation:** {incident.get('recommendation', 'N/A')}")
                        else:
                            st.info("No critical incidents identified")
                    
                    # Hallucination instances
                    with insight_tabs[1]:
                        if "hallucination_instances" in detailed and detailed["hallucination_instances"]:
                            for i, hallucination in enumerate(detailed["hallucination_instances"]):
                                with st.expander(f"Hallucination {i+1}"):
                                    st.markdown(f"**Statement:** \"{hallucination.get('statement', 'N/A')}\"")
                                    st.markdown(f"**Contradicting Evidence:** {hallucination.get('contradicting_evidence', 'N/A')}")
                                    st.markdown(f"**Likely Cause:** {hallucination.get('likely_cause', 'N/A')}")
                        else:
                            st.success("No hallucination instances detected")
                    
                    # Exemplary executions
                    with insight_tabs[2]:
                        if "exemplary_executions" in detailed and detailed["exemplary_executions"]:
                            for i, execution in enumerate(detailed["exemplary_executions"]):
                                with st.expander(f"Example {i+1}: Query {execution.get('query_id', 'Unknown')}"):
                                    st.markdown(f"**Strength:** {execution.get('strength', 'N/A')}")
                                    st.markdown(f"**Highlights:** {execution.get('execution_highlights', 'N/A')}")
                        else:
                            st.info("No exemplary executions highlighted")
                else:
                    st.info("No detailed analysis available")
            
            # Strategic Roadmap Tab
            with judge_tabs[4]:
                if "strategic_recommendations" in analysis:
                    recommendations = analysis["strategic_recommendations"]
                    
                    # Create timeline view with tabs
                    timeline_tabs = st.tabs(["Short Term", "Medium Term", "Long Term"])
                    
                    # Short term recommendations
                    with timeline_tabs[0]:
                        if "short_term" in recommendations and recommendations["short_term"]:
                            for i, rec in enumerate(recommendations["short_term"]):
                                with st.container():
                                    st.markdown(f"""
                                    <div style="background-color: white; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #1f77b4;">
                                        <h4 style="margin-top: 0;">{rec.get('area', 'General')}</h4>
                                        <p><strong>Recommendation:</strong> {rec.get('recommendation', 'N/A')}</p>
                                        <p><strong>Expected Impact:</strong> {rec.get('expected_impact', 'N/A')}</p>
                                        <p><strong>Implementation Effort:</strong> {rec.get('implementation_effort', 'N/A')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No short-term recommendations provided")
                    
                    # Medium term recommendations
                    with timeline_tabs[1]:
                        if "medium_term" in recommendations and recommendations["medium_term"]:
                            for i, rec in enumerate(recommendations["medium_term"]):
                                with st.container():
                                    st.markdown(f"""
                                    <div style="background-color: white; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #ff7f0e;">
                                        <h4 style="margin-top: 0;">{rec.get('area', 'General')}</h4>
                                        <p><strong>Recommendation:</strong> {rec.get('recommendation', 'N/A')}</p>
                                        <p><strong>Expected Impact:</strong> {rec.get('expected_impact', 'N/A')}</p>
                                        <p><strong>Implementation Effort:</strong> {rec.get('implementation_effort', 'N/A')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No medium-term recommendations provided")
                    
                    # Long term recommendations
                    with timeline_tabs[2]:
                        if "long_term" in recommendations and recommendations["long_term"]:
                            for i, rec in enumerate(recommendations["long_term"]):
                                with st.container():
                                    st.markdown(f"""
                                    <div style="background-color: white; padding: 15px; border-radius: 5px; margin-bottom: 10px; border-left: 4px solid #2ca02c;">
                                        <h4 style="margin-top: 0;">{rec.get('area', 'General')}</h4>
                                        <p><strong>Recommendation:</strong> {rec.get('recommendation', 'N/A')}</p>
                                        <p><strong>Expected Impact:</strong> {rec.get('expected_impact', 'N/A')}</p>
                                        <p><strong>Implementation Effort:</strong> {rec.get('implementation_effort', 'N/A')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No long-term recommendations provided")
                else:
                    st.info("No strategic recommendations available")
            
            # Summary section (displayed below tabs)
            st.header("Executive Summary")
            if "detailed_summary" in analysis:
                summary = analysis["detailed_summary"]
                
                # Overall assessment
                st.subheader("Overall Assessment")
                st.write(summary.get("overall_assessment", "No assessment available"))
                
                # Create columns for strengths and weaknesses
                col1, col2 = st.columns(2)
                
                # Strengths
                with col1:
                    st.markdown("### ✅ Key Strengths")
                    for strength in summary.get("strengths", []):
                        st.markdown(f"- {strength}")
                
                # Weaknesses
                with col2:
                    st.markdown("### ⚠️ Key Improvement Areas")
                    for weakness in summary.get("weaknesses", []):
                        st.markdown(f"- {weakness}")
                
                # Recommendations
                st.markdown("### 💡 Priority Recommendations")
                for i, rec in enumerate(summary.get("recommendations", [])):
                    st.markdown(f"{i+1}. {rec}")
            
            # Add download button for full report
            st.download_button(
                label="Download Full Analysis Report (JSON)",
                data=json.dumps(analysis, indent=2),
                file_name="maestro_ai_judge_analysis.json",
                mime="application/json"
            )
            
    elif not gemini_api_key:
        st.error("Please provide a valid Gemini API key to run the AI analysis.")
    else:
            st.error("Analysis could not be completed. Please check your API key and try again.")
            
    
    
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
    
  
    