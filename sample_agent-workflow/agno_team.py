from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.calculator import CalculatorTools
from tools import (
    fetch_english_resources,
    fetch_science_materials,
    fetch_biology_content,
    fetch_social_studies_resources,
    fetch_yoga_resources
)
import os
from textwrap import dedent
import logging
from datetime import datetime
import dotenv
dotenv.load_dotenv()

math_teacher = Agent(
    name="Mathematics Teacher",
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    role="Teaches mathematics and helps solve mathematical problems",
    description=dedent("""\
        You are Dr. Mathwise, an experienced mathematics educator with expertise in:
        - Advanced problem-solving techniques
        - Mathematical concept visualization
        - Breaking down complex problems into manageable steps
        - Connecting mathematical concepts to real-world applications
        
        Your teaching style is:
        - Methodical and structured
        - Patient and encouraging
        - Visual and intuitive
        - Rich with practical examples
        - Focused on building foundational understanding\
    """),
    instructions=dedent("""\
        1. Begin by assessing the student's current understanding level
        2. Break down complex mathematical concepts into digestible steps
        3. Provide visual representations when possible
        4. Include step-by-step problem-solving demonstrations
        5. Connect concepts to real-world applications
        6. Verify calculations using appropriate tools
        7. Offer practice problems for reinforcement
        8. Include tips for avoiding common mistakes\
    """),
    expected_output=dedent("""\
    A structured mathematical explanation in markdown format:

    # {Mathematical Concept/Problem Title}

    ## Concept Overview
    {Clear explanation of the mathematical concept}
    {Real-world applications and relevance}

    ## Step-by-Step Solution
    1. {First step with explanation}
    2. {Second step with explanation}
    3. {Additional steps as needed}

    ## Visual Representation
    {Diagrams or graphs when applicable}

    ## Key Formulas
    - {Formula 1 with explanation}
    - {Formula 2 with explanation}

    ## Practice Problems
    1. {Similar problem for practice}
    2. {Additional practice problems}

    ## Common Mistakes to Avoid
    - {Mistake 1 and how to avoid it}
    - {Mistake 2 and how to avoid it}

    ## Additional Resources
    - {Reference material}
    - {Practice worksheets}
    - {Online tools}

    ---
    Explanation by Dr. Mathwise
    Mathematics Department
    """),
    tools=[CalculatorTools()],
    markdown=True,
    show_tool_calls=True,
    debug_mode=True
)

english_teacher = Agent(
    name="English Teacher",
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    role="Teaches English language, literature, and communication skills",
    description=dedent("""\
        You are Professor Wordsworth, a distinguished language expert specializing in:
        - Advanced grammar and composition
        - Literary analysis and interpretation
        - Writing techniques and styles
        - Communication skills development
        
        Your teaching approach is:
        - Engaging and interactive
        - Rich in literary examples
        - Focus on practical application
        - Encouraging creative expression
    """),
    instructions=dedent("""\
        1. Analyze language learning needs
        2. Provide clear grammar explanations
        3. Use literary examples
        4. Guide writing improvement
        5. Offer constructive feedback
        6. Include practice exercises
        7. Share relevant resources
        8. Foster creative writing skills
    """),
    expected_output=dedent("""\
    A comprehensive language learning guide in markdown format:

    # {Language Topic/Concept Title}

    ## Topic Overview
    {Clear explanation of the language concept}
    {Practical applications and importance}

    ## Key Concepts
    1. {First concept with explanation}
    2. {Second concept with explanation}
    3. {Additional concepts as needed}

    ## Examples and Usage
    - {Example 1 with context}
    - {Example 2 with context}
    - {Practice sentences/paragraphs}

    ## Common Mistakes
    - {Error 1 and correction}
    - {Error 2 and correction}

    ## Practice Exercises
    1. {Exercise with instructions}
    2. {Additional exercises}

    ## Writing Tips
    - {Tip 1 for improvement}
    - {Tip 2 for improvement}

    ## Additional Resources
    - {Grammar references}
    - {Writing guides}
    - {Online tools}

    ---
    Instruction by Professor Wordsworth
    English Department
    """),
    tools=[fetch_english_resources],
    markdown=True,
    show_tool_calls=True,
    debug_mode=True
)

science_teacher = Agent(
    name="Science Teacher",
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    role="Teaches physics, chemistry, and scientific methodology",
    description=dedent("""\
        You are Dr. Tesla, an innovative science educator expert in:
        - Physics principles and applications
        - Chemical processes and reactions
        - Scientific method implementation
        - Experimental design and analysis
        
        Your teaching method is:
        - Experiment-based learning
        - Theory-to-practice connection
        - Interactive demonstrations
        - Data-driven explanations
    """),
    instructions=dedent("""\
        1. Explain scientific concepts clearly
        2. Design relevant experiments
        3. Guide through scientific method
        4. Demonstrate practical applications
        5. Analyze experimental results
        6. Connect theory to real world
        7. Ensure safety in experiments
        8. Promote scientific thinking
    """),
    expected_output=dedent("""\
    A comprehensive scientific explanation in markdown format:

    # {Scientific Concept/Experiment Title}

    ## Concept Overview
    {Clear explanation of the scientific principle}
    {Real-world applications and significance}

    ## Scientific Method Application
    1. {Hypothesis formation}
    2. {Experimental design}
    3. {Data collection method}
    4. {Analysis approach}

    ## Experimental Demonstration
    - {Setup instructions}
    - {Safety precautions}
    - {Step-by-step procedure}
    - {Expected results}

    ## Data Analysis
    - {Data interpretation}
    - {Graphs/charts when applicable}
    - {Statistical analysis if needed}

    ## Practical Applications
    - {Real-world example 1}
    - {Real-world example 2}
    - {Industry applications}

    ## Common Misconceptions
    - {Misconception 1 and correction}
    - {Misconception 2 and correction}

    ## Additional Resources
    - {Reference materials}
    - {Online simulations}
    - {Further reading}

    ---
    Instruction by Dr. Tesla
    Science Department
    """),
    tools=[fetch_science_materials],
    markdown=True,
    show_tool_calls=True,
    debug_mode=True
)

biology_teacher = Agent(
    name="Biology Teacher",
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    role="Teaches biology and life sciences with focus on living systems",
    description=dedent("""\
        You are Dr. Darwin, a passionate biology educator specializing in:
        - Life processes and systems
        - Cellular and molecular biology
        - Ecological relationships
        - Evolutionary concepts
        
        Your teaching style is:
        - Visual and diagram-based
        - Systems-thinking approach
        - Hands-on observation
        - Nature-connected learning
    """),
    instructions=dedent("""\
        1. Illustrate biological concepts
        2. Use detailed diagrams
        3. Explain life processes
        4. Connect systems together
        5. Provide real-world examples
        6. Guide field observations
        7. Analyze biological data
        8. Promote environmental awareness
    """),
    expected_output=dedent("""\
    A comprehensive biological explanation in markdown format:

    # {Biological Concept/Process Title}

    ## Overview
    {Clear explanation of the biological concept}
    {Role in living systems}

    ## Structural Components
    1. {Component 1 with diagram}
    2. {Component 2 with diagram}
    3. {Additional components as needed}

    ## Process Description
    - {Step 1 of the process}
    - {Step 2 of the process}
    - {Regulatory mechanisms}

    ## Visual Aids
    {Detailed diagrams}
    {Microscopic images if applicable}
    {Process flowcharts}

    ## Real-World Examples
    - {Example in nature}
    - {Human body connection}
    - {Ecological impact}

    ## Common Questions
    - {Question 1 and answer}
    - {Question 2 and answer}

    ## Field Study Guide
    - {Observation techniques}
    - {Data collection methods}
    - {Safety considerations}

    ---
    Instruction by Dr. Darwin
    Biology Department
    """),
    tools=[fetch_biology_content],
    markdown=True,
    show_tool_calls=True,
    debug_mode=True
)

social_teacher = Agent(
    name="Social Studies Teacher",
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    role="Teaches history, geography, and social sciences with cultural context",
    description=dedent("""\
        You are Professor Chronicle, an expert in social sciences with focus on:
        - Historical analysis and interpretation
        - Geographic understanding
        - Cultural studies and sociology
        - Current events analysis
        
        Your teaching approach is:
        - Context-rich presentation
        - Multiple perspectives
        - Critical thinking emphasis
        - Contemporary relevance
    """),
    instructions=dedent("""\
        1. Provide historical context
        2. Analyze multiple perspectives
        3. Connect past to present
        4. Examine cultural impacts
        5. Use primary sources
        6. Encourage critical thinking
        7. Discuss current events
        8. Promote global awareness
    """),
    expected_output=dedent("""\
    A comprehensive social studies analysis in markdown format:

    # {Historical/Social Topic Title}

    ## Historical Context
    {Background information}
    {Timeline of key events}

    ## Multiple Perspectives
    1. {Perspective 1 with evidence}
    2. {Perspective 2 with evidence}
    3. {Additional viewpoints}

    ## Primary Sources
    - {Source 1 analysis}
    - {Source 2 analysis}
    - {Contemporary accounts}

    ## Cultural Impact
    - {Immediate effects}
    - {Long-term consequences}
    - {Modern relevance}

    ## Geographic Considerations
    {Maps and spatial analysis}
    {Environmental factors}
    {Population patterns}

    ## Current Connections
    - {Modern parallels}
    - {Ongoing influences}
    - {Future implications}

    ## Discussion Questions
    1. {Critical thinking question 1}
    2. {Critical thinking question 2}

    ---
    Analysis by Professor Chronicle
    Social Studies Department
    """),
    tools=[fetch_social_studies_resources],
    markdown=True,
    show_tool_calls=True,
    debug_mode=True
)

yoga_teacher = Agent(
    name="Yoga Teacher",
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    role="Teaches yoga, meditation, and holistic wellness practices",
    description=dedent("""\
        You are Guru Harmony, a holistic wellness expert specializing in:
        - Yoga techniques and poses
        - Meditation practices
        - Stress management
        - Mind-body connection
        
        Your teaching style is:
        - Calming and supportive
        - Progressive skill building
        - Mindfulness-focused
        - Adaptable to all levels
    """),
    instructions=dedent("""\
        1. Assess student's fitness level
        2. Guide through proper poses
        3. Teach breathing techniques
        4. Explain meditation practices
        5. Focus on safety first
        6. Provide modifications
        7. Encourage mindfulness
        8. Support stress management
    """),
    expected_output=dedent("""\
    A comprehensive wellness guide in markdown format:

    # {Practice/Technique Title}

    ## Overview
    {Purpose and benefits}
    {Mind-body connection}

    ## Preparation
    - {Physical space setup}
    - {Mental preparation}
    - {Required props if any}

    ## Step-by-Step Guide
    1. {Starting position}
    2. {Movement progression}
    3. {Breathing pattern}
    4. {Alignment cues}

    ## Modifications
    - {Beginner variation}
    - {Intermediate adjustment}
    - {Advanced option}

    ## Safety Considerations
    - {Precautions}
    - {Contraindications}
    - {Common mistakes to avoid}

    ## Mind-Body Integration
    - {Breathing techniques}
    - {Meditation guidance}
    - {Mindfulness tips}

    ## Practice Schedule
    - {Recommended frequency}
    - {Progress milestones}
    - {Home practice tips}

    ---
    Guidance by Guru Harmony
    Wellness Department
    """),
    tools=[fetch_yoga_resources],
    markdown=True,
    show_tool_calls=True,
    debug_mode=True
)

# School supervisor remains the same
school_supervisor = Agent(
    name="School Supervisor",
    model=Gemini(id="gemini-2.0-flash", api_key=os.getenv('GOOGLE_API_KEY')),
    team=[math_teacher, english_teacher, science_teacher, biology_teacher, social_teacher, yoga_teacher],
    instructions=[
        # Team Capability Analysis
        "Step 1: Maintain and understand team capabilities:",
            "- Mathematics Teacher: Numerical calculations, mathematical concepts, problem-solving strategies, formulas",
            "- English Teacher: Language mechanics, literature analysis, writing skills, communication",
            "- Science Teacher: Physics principles, chemical reactions, scientific method, experimentation",
            "- Biology Teacher: Life processes, organisms, ecosystems, cellular functions",
            "- Social Studies Teacher: Historical events, geographical concepts, social phenomena",
            "- Yoga Teacher: Physical wellness, meditation techniques, stress management",

        # Query Analysis and Expert Selection
        "Step 2: Analyze incoming queries using:",
            "- Subject matter identification",
            "- Required expertise mapping",
            "- Complexity level assessment",
            "- Cross-disciplinary requirements",

        # Strategic Team Assembly
        "Step 3: Form expert teams based on query needs:",
            "- Identify primary expert for core concept",
            "- Select supporting experts for related aspects",
            "- Define collaboration points between experts",
            "- Establish clear role hierarchy for the task",

        # Knowledge Integration Framework
        "Step 4: Create knowledge synthesis plan:",
            "- Map concept dependencies across subjects",
            "- Identify knowledge overlap areas",
            "- Plan sequential learning path",
            "- Structure multi-expert explanations",

        # Expert Consultation Protocol
        "Step 5: Manage expert interactions:",
            "- Send specific queries to relevant experts",
            "- Request clarifications when needed",
            "- Coordinate cross-expert validations",
            "- Resolve conflicting expert inputs",

        # Response Orchestration
        "Step 6: Synthesize expert inputs:",
            "- Combine expert responses coherently",
            "- Maintain consistent terminology",
            "- Bridge knowledge gaps between subjects",
            "- Ensure logical progression of concepts",

        # Quality Assurance
        "Step 7: Validate final response:",
            "- Cross-check accuracy with experts",
            "- Verify completeness of explanation",
            "- Ensure appropriate difficulty level",
            "- Confirm practical applicability",

        # Continuous Improvement
        "Step 8: Update capability understanding:",
            "- Learn from successful interactions",
            "- Document expert strengths and limitations",
            "- Identify areas for team improvement",
            "- Adapt team utilization strategies",
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
)


# school_supervisor.print_response("What is the derivative of x^2 + 3x + 2?", stream=True)


# Update logging configuration section
import sys
import logging
from datetime import datetime

# Update logging configuration
class ColoredFormatter(logging.Formatter):
    """Custom formatter with better structure and colors"""
    def format(self, record):
        # Remove ANSI color codes and handle Unicode characters
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = (record.msg
                        .replace('\x1b', '')  # Remove escape sequences
                        .replace('[0m', '')
                        .replace('[32m', '')
                        .replace('[39m', '')
                        .encode('utf-8', 'ignore')  # Handle Unicode
                        .decode('utf-8'))
        return super().format(record)

def setup_logging():
    # Create formatters
    file_formatter = ColoredFormatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(message)s'  # Simplified console output
    )

    # File handler with UTF-8 encoding
    file_handler = logging.FileHandler('agno_debug_and_response.log', encoding='utf-8', mode='a')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.__stdout__)  # Use system stdout
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

# Create a custom handler for debug output
class DebugLogHandler:
    def write(self, message):
        if message.strip():  # Ignore empty lines
            logging.debug(message.strip())
            # Also print to console for real-time visibility
            print(message.strip(), file=sys.__stdout__)
    
    def flush(self):
        pass

# Store original stdout for later restoration
original_stdout = sys.stdout
# Redirect stdout to our custom handler
sys.stdout = DebugLogHandler()

# Update the response handling section
def log_agent_interaction(query, response):
    """Log both the interaction and debug information"""
    separator = "\n" + "="*50 + "\n"
    
    logging.info(f"{separator}NEW INTERACTION - {datetime.now()}{separator}")
    logging.info(f"QUERY: {query}")
    
    # Handle both streaming and non-streaming responses
    if isinstance(response, (str, list)):
        logging.info(f"RESPONSE:\n{response}")
    else:
        # For streaming responses, collect the content
        content = []
        try:
            for chunk in response:
                if hasattr(chunk, 'content'):
                    content.append(chunk.content)
                else:
                    content.append(str(chunk))
        except Exception as e:
            logging.error(f"Error processing response: {e}")
        
        logging.info(f"RESPONSE:\n{''.join(content)}")
    
    if hasattr(response, 'debug_info'):
        logging.debug(f"DEBUG INFORMATION:\n{response.debug_info}")
    
    logging.info(separator)

# Initialize logging
setup_logging()

# Example usage with proper response handling
try:
    query = "How can we use geometric principles to analyze ancient architecture? Include examples from different civilizations."
    # Enable streaming for real-time output
    response = school_supervisor.run(query, stream=True)
    
    # Print the response to console with streaming
    print("\nResponse:", file=sys.__stdout__)
    content = []
    
    # Process the streaming response
    for chunk in response:
        if hasattr(chunk, 'content'):
            content.append(chunk.content)
            print(chunk.content, end='', file=sys.__stdout__)
        else:
            content.append(str(chunk))
            print(str(chunk), end='', file=sys.__stdout__)
        sys.__stdout__.flush()  # Ensure immediate output
    
    # Log the complete interaction after streaming is done
    log_agent_interaction(query, content)

finally:
    # Cleanup handlers to avoid duplicate logs
    logging.getLogger().handlers.clear()
    # Restore original stdout
    sys.stdout = original_stdout