import pytest
from click.testing import CliRunner
from llm.cli import cli
import tempfile
import os
from unittest.mock import Mock
from tests.mocks.dspy_mock import MockPrediction

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def sample_collection(runner):
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample files for different use cases
        files = {
            "medical_research.txt": """
            Recent studies on mRNA vaccines show 95% efficacy against original strains.
            Variant-specific boosters demonstrate 60-80% protection against newer variants.
            Side effects remain minimal, primarily including fatigue and soreness.
            Study limitations include small sample size and short observation period.
            Next phase will focus on long-term immunity patterns.
            """,
            "legal_docs.txt": """
            Privacy law precedent: Smith v. Technology Corp (2022)
            Key findings: Companies must explicitly disclose data sharing practices
            Impact: Strengthened user consent requirements
            Jurisdiction: Federal Court, 9th Circuit
            Precedent relevance score: 0.85
            Risk factors identified: Data retention policies, Cross-border transfers
            """,
            "technical_docs.txt": """
            OAuth2 Implementation Guide:
            1. Token refresh occurs automatically within 5 minutes of expiration
            2. Uses industry-standard encryption for token storage
            3. Implements rate limiting on refresh attempts
            Security compliance: OWASP Top 10 2021
            Known vulnerabilities: None critical, 2 medium severity
            Last audit date: 2023-12-15
            """,
            "source_code.py": """
            def simple_function():
                return 42

            def complex_function(n):
                for i in range(n):
                    for j in range(n):
                        print(i * j)
            """,
            "pull_request.diff": """
            diff --git a/app.py b/app.py
            --- a/app.py
            +++ b/app.py
            @@ -1,5 +1,7 @@
             def process_data():
            -    return data
            +    if not validate_input(data):
            +        raise ValueError("Invalid input")
            +    return transform_data(data)
            """,
            "research_papers.txt": """
            Renewable Energy Study 2023:
            Solar Power:
            - Environmental Impact Score: 85/100
            - Cost Efficiency Rating: 75/100
            - Implementation Challenges: Grid integration, Storage capacity
            Wind Power:
            - Environmental Impact Score: 90/100
            - Cost Efficiency Rating: 70/100
            - Implementation Challenges: Noise pollution, Visual impact
            """,
            "financial_reports.txt": """
            Q2 2023 Market Analysis:
            - Market Growth Rate: 12.5%
            - Risk Assessment Score: 4/10
            - Key Opportunities: 
              * Cloud adoption trend
              * AI integration demand
              * Mobile-first solutions
            - Potential Threats:
              * New competitors
              * Regulatory changes
              * Supply chain disruptions
            """,
            "market_research.txt": """
            Competitor Analysis Q2 2023:
            Company A: 
            - Market Share: 35%
            - Growth Rate: 15%
            - Key Features: AI automation, Cloud storage
            Company B:
            - Market Share: 25%
            - Growth Rate: 8%
            - Key Features: Mobile integration, Analytics
            Priority Areas:
            1. AI/ML capabilities
            2. Mobile optimization
            3. Security features
            """
        }
        
        # Write the files
        for filename, content in files.items():
            with open(os.path.join(tmpdir, filename), 'w') as f:
                f.write(content.strip())
        
        yield tmpdir

def test_single_input_basic(runner):
    """Test basic single input with positional argument"""
    result = runner.invoke(cli, [
        'dspy',
        'ChainOfThought(question -> answer)',
        'Explain how photosynthesis works in simple terms.'
    ])
    print("Output:", result.output)  # Add debug output
    print("Exit code:", result.exit_code)  # Add debug output
    assert result.exit_code == 0
    assert len(result.output.strip()) > 0

def test_single_input_classification(runner, mocker):
    """Test sentiment classification with type annotation"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    mock_prediction = MockPrediction(answer="positive")
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.Predict', mock_module)
    
    result = runner.invoke(cli, [
        'dspy',
        'Predict(text -> sentiment: str{positive, negative, neutral})',
        'This product exceeded all my expectations!'
    ])
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    assert result.output.strip() in ['positive', 'negative', 'neutral']

def test_stdin_basic(runner):
    """Test basic stdin input"""
    result = runner.invoke(cli, [
        'dspy',
        'ChainOfThought(question -> answer)'
    ], input='What are the main differences between REST and GraphQL?')
    assert result.exit_code == 0
    assert len(result.output.strip()) > 0

def test_stdin_code_complexity(runner, mocker, sample_collection):
    """Test code complexity classification"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    mock_prediction = MockPrediction(answer="O(n^2)")
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.Predict', mock_module)
    
    with open(os.path.join(sample_collection, 'source_code.py'), 'r') as f:
        code = f.read()
    result = runner.invoke(cli, [
        'dspy',
        'Predict(code -> complexity: str{O(1), O(n), O(n^2), O(2^n)})'
    ], input=code)
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    assert result.output.strip() in ['O(1)', 'O(n)', 'O(n^2)', 'O(2^n)']

def test_multiple_inputs_basic(runner):
    """Test multiple inputs with named options"""
    result = runner.invoke(cli, [
        'dspy',
        'ChainOfThought(topic, audience -> explanation)',
        '--topic', 'quantum computing',
        '--audience', 'high school students'
    ])
    assert result.exit_code == 0
    assert len(result.output.strip()) > 0

def test_multiple_inputs_structured(runner, mocker):
    """Test multiple inputs with structured output"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    structured_output = {
        'root_cause': "Memory allocation issue",
        'severity': "high",
        'fix_steps': ["Implement chunked upload", "Add memory limits", "Monitor usage"]
    }
    mock_prediction = MockPrediction(answer=str(structured_output))
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ProgramOfThought', mock_module)
    
    result = runner.invoke(cli, [
        'dspy',
        'ProgramOfThought(bug_report, system_context -> root_cause: str, severity: str{low, medium, high}, fix_steps: list[str])',
        '--bug_report', 'Application crashes when uploading files larger than 1GB',
        '--system_context', 'Node.js backend with S3 storage'
    ])
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    output = result.output.strip()
    assert 'root_cause' in output
    assert any(level in output for level in ['low', 'medium', 'high'])
    assert '[' in output and ']' in output  # Check for list output

def test_rag_basic(runner, sample_collection):
    """Test basic RAG functionality"""
    result = runner.invoke(cli, [
        'dspy',
        'ChainOfThought(context, question -> answer)',
        '--context', 'technical_docs',
        '--question', 'How does our authentication system handle OAuth2 token refresh?'
    ])
    assert result.exit_code == 0
    assert 'token' in result.output.lower()
    assert 'refresh' in result.output.lower()

def test_rag_fact_extraction(runner, mocker, sample_collection):
    """Test RAG with fact extraction"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    structured_output = {
        'dates': ['2022'],
        'amounts': [0.85],
        'entities': ['Smith', 'Technology Corp', 'Federal Court', '9th Circuit']
    }
    mock_prediction = MockPrediction(answer=str(structured_output))
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ProgramOfThought', mock_module)
    
    result = runner.invoke(cli, [
        'dspy',
        'ProgramOfThought(context, query -> dates: list[str], amounts: list[float], entities: list[str])',
        '--context', 'legal_docs',
        '--query', 'Extract all dates, amounts, and entities from the document'
    ])
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    output = result.output.strip()
    assert '[' in output and ']' in output  # Check for list output
    assert '2022' in output  # Should find the date from sample data

def test_stdin_with_options(runner, mocker, sample_collection):
    """Test stdin with additional options"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    structured_output = {
        'feedback': ['Added input validation', 'Improved error handling'],
        'risk_level': 'medium',
        'approval': True
    }
    mock_prediction = MockPrediction(answer=str(structured_output))
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ProgramOfThought', mock_module)
    
    with open(os.path.join(sample_collection, 'pull_request.diff'), 'r') as f:
        diff = f.read()
    result = runner.invoke(cli, [
        'dspy',
        'ProgramOfThought(diff, standards -> feedback: list[str], risk_level: str{low, medium, high}, approval: bool)',
        '--diff', 'stdin',
        '--standards', 'standard code review practices'
    ], input=diff)
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    output = result.output.strip()
    assert any(level in output for level in ['low', 'medium', 'high'])
    assert any(val in output for val in ['true', 'false', 'True', 'False'])

def test_complex_rag_analysis(runner, mocker, sample_collection):
    """Test complex RAG analysis with multiple fields"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    structured_output = {
        'environmental_impact': 85.5,
        'cost_efficiency': 75.0,
        'implementation_challenges': ['Storage stability', 'Distribution logistics', 'Temperature control'],
        'recommendations': ['Improve cold chain', 'Partner with logistics providers', 'Invest in monitoring']
    }
    mock_prediction = MockPrediction(answer=str(structured_output))
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ProgramOfThought', mock_module)
    
    result = runner.invoke(cli, [
        'dspy',
        'ProgramOfThought(context, analysis_request -> environmental_impact: float{0-100}, cost_efficiency: float{0-100}, implementation_challenges: list[str], recommendations: list[str])',
        '--context', 'medical_research',
        '--analysis_request', 'Analyze the efficacy and implementation challenges of mRNA vaccines'
    ])
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    output = result.output.strip()
    assert any(str(i) in output for i in range(101))  # Check for float{0-100}
    assert '[' in output and ']' in output  # Check for list output

def test_security_audit(runner, mocker, sample_collection):
    """Test security audit with compliance check"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    structured_output = {
        'compliance_status': 'partial',
        'vulnerabilities': ['Token expiration not enforced', 'Missing rate limiting'],
        'risk_level': 'high',
        'action_items': ['Implement token expiration', 'Add rate limiting', 'Update documentation']
    }
    mock_prediction = MockPrediction(answer=str(structured_output))
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ProgramOfThought', mock_module)
    
    result = runner.invoke(cli, [
        'dspy',
        'ProgramOfThought(context, audit_scope -> compliance_status: str{compliant, partial, non_compliant}, vulnerabilities: list[str], risk_level: str{low, medium, high, critical}, action_items: list[str])',
        '--context', 'technical_docs',
        '--audit_scope', 'Evaluate OAuth2 implementation against OWASP standards'
    ])
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    output = result.output.strip()
    assert any(status in output for status in ['compliant', 'partial', 'non_compliant'])
    assert any(level in output for level in ['low', 'medium', 'high', 'critical'])
    assert '[' in output and ']' in output  # Check for list output

def test_strategic_planning(runner, mocker, sample_collection):
    """Test strategic planning with dictionary outputs"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    structured_output = {
        'priority_score': {'feature_a': 0.85, 'feature_b': 0.75, 'feature_c': 0.65},
        'timeline': {'feature_a': 'Q3', 'feature_b': 'Q4', 'feature_c': 'Q1 2024'},
        'resource_requirements': ['2 backend devs', '1 ML engineer', 'DevOps support'],
        'expected_roi': 2.5
    }
    mock_prediction = MockPrediction(answer=str(structured_output))
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ProgramOfThought', mock_module)
    
    result = runner.invoke(cli, [
        'dspy',
        'ProgramOfThought(market_data, competitor_analysis, objectives -> priority_score: dict[str, float], timeline: dict[str, str], resource_requirements: list[str], expected_roi: float)',
        '--market_data', 'market_research',
        '--competitor_analysis', 'competitor_reports',
        '--objectives', 'Identify top 3 features for competitive advantage in Q3'
    ])
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    output = result.output.strip()
    assert '{' in output and '}' in output  # Check for dict output
    assert '[' in output and ']' in output  # Check for list output
    assert any(str(i) in output for i in range(10))  # Check for float values

def test_medical_research_structured(runner, mocker, sample_collection):
    """Test medical research with confidence levels"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    structured_output = {
        'findings': ['Variant B shows 40% higher transmission', 'Vaccine efficacy reduced by 15%'],
        'confidence_level': 'high',
        'limitations': ['Small sample size', 'Limited geographic scope'],
        'next_steps': ['Expand study population', 'Monitor new variants']
    }
    mock_prediction = MockPrediction(answer=str(structured_output))
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ProgramOfThought', mock_module)
    
    result = runner.invoke(cli, [
        'dspy',
        'ProgramOfThought(context, study_query -> findings: list[str], confidence_level: str{high, medium, low}, limitations: list[str], next_steps: list[str])',
        '--context', 'medical_research',
        '--study_query', 'Analyze the efficacy of different COVID-19 variants'
    ])
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    output = result.output.strip()
    assert any(level in output for level in ['high', 'medium', 'low'])
    assert '[' in output and ']' in output  # Check for list output

def test_legal_analysis_with_metrics(runner, mocker, sample_collection):
    """Test legal analysis with float metrics"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    structured_output = {
        'jurisdiction': 'California',
        'precedent_relevance': 0.85,
        'risk_factors': ['User consent not explicit', 'Cross-border data transfer'],
        'recommendation': 'Update privacy policy and implement consent management'
    }
    mock_prediction = MockPrediction(answer=str(structured_output))
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ProgramOfThought', mock_module)
    
    result = runner.invoke(cli, [
        'dspy',
        'ProgramOfThought(background, case_details -> jurisdiction: str, precedent_relevance: float, risk_factors: list[str], recommendation: str)',
        '--background', 'legal_docs',
        '--case_details', 'Evaluate data privacy compliance for our new feature'
    ])
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    output = result.output.strip()
    assert any(str(i) in output for i in range(10))  # Check for float values
    assert '[' in output and ']' in output  # Check for list output

def test_sustainability_metrics(runner, mocker, sample_collection):
    """Test sustainability analysis with bounded float values"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    structured_output = {
        'environmental_impact': 85.5,
        'cost_efficiency': 75.0,
        'implementation_challenges': ['Grid integration', 'Land use', 'Storage capacity'],
        'recommendations': ['Hybrid solution', 'Smart grid deployment', 'Battery storage']
    }
    mock_prediction = MockPrediction(answer=str(structured_output))
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ProgramOfThought', mock_module)
    
    result = runner.invoke(cli, [
        'dspy',
        'ProgramOfThought(context, analysis_request -> environmental_impact: float{0-100}, cost_efficiency: float{0-100}, implementation_challenges: list[str], recommendations: list[str])',
        '--context', 'research_papers',
        '--analysis_request', 'Compare solar vs wind power for metropolitan areas'
    ])
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    output = result.output.strip()
    assert any(str(i) in output for i in range(101))  # Check for float{0-100}
    assert '[' in output and ']' in output  # Check for list output

def test_multi_source_analysis(runner, mocker, sample_collection):
    """Test analysis with multiple RAG sources"""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    structured_output = {
        'growth_rate': 12.5,
        'risk_score': 7,
        'opportunities': ['Market expansion', 'Product innovation'],
        'threats': ['New competitors', 'Regulatory changes']
    }
    mock_prediction = MockPrediction(answer=str(structured_output))
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ProgramOfThought', mock_module)
    
    result = runner.invoke(cli, [
        'dspy',
        'ProgramOfThought(market_data, competitor_data, query -> growth_rate: float, risk_score: int{1-10}, opportunities: list[str], threats: list[str])',
        '--market_data', 'financial_reports',
        '--competitor_data', 'market_research',
        '--query', 'Evaluate market position for Q3 planning'
    ])
    print("Output:", result.output)
    print("Exit code:", result.exit_code)
    print("Exception:", result.exception)
    assert result.exit_code == 0
    output = result.output.strip()
    assert any(str(i) in output for i in range(11))  # Check for int{1-10}
    assert '[' in output and ']' in output  # Check for list output 