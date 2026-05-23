#!/usr/bin/env python3
"""
LS-DYNA Crash Analysis Intelligence System
Complete implementation for CAE data analysis with Claude AI integration
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from anthropic import Anthropic
# PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    _REPORTLAB_AVAILABLE = True
except Exception:
    _REPORTLAB_AVAILABLE = False

# Load environment variables from .env (if present)
load_dotenv()

# Initialize Anthropic client using API key from environment (if present)
_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if _anthropic_api_key:
    client = Anthropic(api_key=_anthropic_api_key)
else:
    client = None
MODEL = "claude-opus-4-6"

@dataclass
class SimulationMetrics:
    """Container for LS-DYNA simulation metrics"""
    simulation_id: str
    timestamp: str
    
    # Acceleration metrics
    peak_acceleration_x: float
    peak_acceleration_y: float
    peak_acceleration_z: float
    peak_acceleration_resultant: float
    time_to_peak_ms: float
    hic_value: float
    
    # Energy metrics
    kinetic_energy_initial: float
    kinetic_energy_final: float
    kinetic_energy_dissipated_percent: float
    internal_energy: float
    hourglass_energy_percent: float
    
    # Deformation metrics
    max_displacement: float
    max_plastic_strain: float
    element_deletion_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LS_DYNA_Analyzer:
    """Main analyzer class coordinating parsing, aggregation, and AI analysis"""
    
    def __init__(self):
        self.conversation_history = []
        self.simulation_data = None
        self.analysis_results = {}
    
    def load_simulation_data(self, metrics_json: str) -> SimulationMetrics:
        """
        Load simulation metrics from JSON (in real usage, parse from files)
        
        Example JSON format:
        {
            "simulation_id": "CRASH_FRONTAL_NHTSA",
            "peak_acceleration_resultant": 82.3,
            "peak_acceleration_x": 45.2,
            ... etc
        }
        """
        data = json.loads(metrics_json)
        
        self.simulation_data = SimulationMetrics(
            simulation_id=data.get("simulation_id", "UNKNOWN"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            peak_acceleration_x=data.get("peak_acceleration_x", 0),
            peak_acceleration_y=data.get("peak_acceleration_y", 0),
            peak_acceleration_z=data.get("peak_acceleration_z", 0),
            peak_acceleration_resultant=data.get("peak_acceleration_resultant", 0),
            time_to_peak_ms=data.get("time_to_peak_ms", 0),
            hic_value=data.get("hic_value", 0),
            kinetic_energy_initial=data.get("kinetic_energy_initial", 0),
            kinetic_energy_final=data.get("kinetic_energy_final", 0),
            kinetic_energy_dissipated_percent=data.get("kinetic_energy_dissipated_percent", 0),
            internal_energy=data.get("internal_energy", 0),
            hourglass_energy_percent=data.get("hourglass_energy_percent", 0),
            max_displacement=data.get("max_displacement", 0),
            max_plastic_strain=data.get("max_plastic_strain", 0),
            element_deletion_count=data.get("element_deletion_count", 0)
        )
        
        return self.simulation_data
    
    def _build_context_prompt(self) -> str:
        """Build comprehensive context prompt with all metrics"""
        if not self.simulation_data:
            return "No simulation data loaded"
        
        metrics = self.simulation_data.to_dict()
        
        context = f"""
CRASH SIMULATION DATA FOR ANALYSIS:

Simulation ID: {metrics['simulation_id']}
Timestamp: {metrics['timestamp']}

=== ACCELERATION METRICS ===
Peak Acceleration (X): {metrics['peak_acceleration_x']:.2f} g
Peak Acceleration (Y): {metrics['peak_acceleration_y']:.2f} g
Peak Acceleration (Z): {metrics['peak_acceleration_z']:.2f} g
Peak Acceleration (Resultant): {metrics['peak_acceleration_resultant']:.2f} g
Time to Peak: {metrics['time_to_peak_ms']:.1f} ms
Head Injury Criterion (HIC): {metrics['hic_value']:.0f}

=== ENERGY METRICS ===
Initial Kinetic Energy: {metrics['kinetic_energy_initial']:.0f} J
Final Kinetic Energy: {metrics['kinetic_energy_final']:.0f} J
Energy Dissipated: {metrics['kinetic_energy_dissipated_percent']:.1f}%
Internal Energy: {metrics['internal_energy']:.0f} J
Hourglass Energy: {metrics['hourglass_energy_percent']:.1f}%

=== DEFORMATION METRICS ===
Maximum Displacement: {metrics['max_displacement']:.3f} m
Maximum Plastic Strain: {metrics['max_plastic_strain']:.4f}
Deleted Elements: {metrics['element_deletion_count']}

=== SAFETY STANDARDS CONTEXT ===
- NHTSA HIC Limit: 1000 (15ms window)
- Peak acceleration safe limit: ~80g
- Energy dissipation target: >60%
- Maximum plastic strain acceptable: <0.25
"""
        return context
    
    def initial_assessment(self) -> str:
        """
        First turn: Claude reviews all metrics and provides initial assessment
        """
        if not self.simulation_data:
            return "Error: No simulation data loaded"
        
        prompt = f"""
{self._build_context_prompt()}

You are an expert automotive crash analysis engineer with 20+ years of experience in LS-DYNA FEA.

Provide a comprehensive INITIAL ASSESSMENT:

1. **COMPLIANCE STATUS**
   - Does this design PASS or FAIL NHTSA standards?
   - HIC value assessment (safe/marginal/critical)
   - Acceleration levels assessment
   
2. **KEY FINDINGS** (top 5, in order of importance)
   - Each finding with specific values
   - Engineering interpretation
   
3. **ENERGY ANALYSIS**
   - How is kinetic energy being dissipated?
   - Is energy balance healthy?
   - Hourglass energy assessment
   
4. **CRITICAL ZONES**
   - Where are the problem areas?
   - What's failing and why?
   
5. **OVERALL ASSESSMENT** (score 1-10)
   - Quality of design
   - Confidence in results
   
Be specific, quantitative, and technical. This will be the foundation for optimization work.
"""
        
        return self._chat(prompt)
    
    def identify_failure_modes(self, area: str = "primary") -> str:
        """
        Second turn: Deep dive into failure modes
        """
        prompt = f"""
Based on the initial assessment, let's analyze the failure modes in detail.

Focus on: {area} failure modes

Provide:

1. **ROOT CAUSE ANALYSIS**
   - Physics-based explanation of why failures are occurring
   - Which materials/zones are most affected
   - Progression of failure (how does it develop over time?)

2. **CONTRIBUTING DESIGN FACTORS**
   - What about the current design allows these failures?
   - Geometric weaknesses
   - Material choice issues
   - Connectivity/reinforcement gaps

3. **SENSITIVITY ANALYSIS**
   - Which parameters have the biggest influence on failures?
   - If we changed thickness by ±20%, what would happen?
   - How does geometry modification affect the failure zone?

4. **FAILURE MECHANICS**
   - Local stress concentrations
   - Buckling or instability factors
   - Energy absorption inefficiencies

Be as specific as possible with numbers and locations.
"""
        
        return self._chat(prompt)
    
    def generate_optimization_recommendations(self) -> str:
        """
        Third turn: Generate ranked recommendations for improvement
        """
        prompt = """
Based on the failure mode analysis, generate a PRIORITIZED OPTIMIZATION STRATEGY:

Format each recommendation as:

**[TIER 1/2/3] - [Title]**
- Specific change (e.g., "Increase A-pillar thickness from 1.2 to 1.4 mm")
- Expected improvement: ±X% for [metric]
- Implementation complexity: (High/Medium/Low)
- Engineering effort: ~X hours
- Risk factors: [any concerns]
- Cost/weight impact: [qualitative assessment]

TIER 1 (CRITICAL - Required for compliance):
[At least 2-3 must-fix items with high impact]

TIER 2 (IMPORTANT - Significant performance gains):
[4-5 items that meaningfully improve design]

TIER 3 (NICE-TO-HAVE - Incremental improvements):
[2-3 lower-effort optimizations]

Also provide:
1. **Testing Plan** - What simulations should we run to validate improvements?
2. **Timeline** - How long to implement each tier? (realistic schedule)
3. **Expected Outcome** - If all recommendations are implemented, what improvements do you expect?
4. **Benchmarking** - How will optimized design compare to competitors?
5. **Risks** - Any unintended consequences to watch for?

Rank by impact/effort ratio (highest impact per unit effort first).
"""
        
        return self._chat(prompt)
    
    def compare_with_baseline(self, baseline_metrics: Dict[str, float]) -> str:
        """
        Compare current design with baseline design
        """
        baseline_json = json.dumps(baseline_metrics, indent=2)
        
        prompt = f"""
BASELINE DESIGN METRICS:
{baseline_json}

CURRENT DESIGN METRICS:
{json.dumps(self.simulation_data.to_dict(), indent=2)}

Perform a detailed COMPARATIVE ANALYSIS:

1. **METRIC-BY-METRIC COMPARISON**
   Format: Metric | Baseline | Current | Delta | Winner
   - Be quantitative for all comparisons
   - Calculate percentage differences
   
2. **PERFORMANCE TRADE-OFFS**
   - What got better?
   - What got worse?
   - Is the trade-off worth it?
   
3. **COMPLIANCE COMPARISON**
   - Which design is safer?
   - Which has more margin?
   - Any regulatory concerns?
   
4. **DESIGN EFFICIENCY**
   - Which design does more with less (weight, material)?
   - Which has better energy management?
   
5. **RECOMMENDATION**
   - Which design should we pursue?
   - Should we hybrid-combine best features?
   - What validation tests would confirm the choice?

Provide specific reasoning, not just conclusions.
"""
        
        return self._chat(prompt)
    
    def _chat(self, user_message: str) -> str:
        """
        Internal method: Send message and maintain conversation history
        """
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        # If Anthropic client is not configured, return a clear error message
        if client is None:
            error_msg = (
                "Error: Anthropic API key is not configured.\n"
                "Set the environment variable ANTHROPIC_API_KEY or create a .env file with that key.\n"
                "See .env.example for the variable name. The analysis requiring Claude will be skipped."
            )
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg

        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system="""You are an expert automotive crash analysis engineer with 25+ years of experience.
You have deep knowledge of:
- LS-DYNA finite element analysis
- Vehicle safety standards (NHTSA, Euro NCAP, IIHS)
- Structural optimization and design
- Material science and failure mechanics
- Crash test physics and injury criteria

Provide detailed, specific, quantitative analysis. Always explain the physics.
Use engineering terminology confidently. Provide specific numbers and metrics.
When making recommendations, always explain the rationale and expected impact.""",
            messages=self.conversation_history
        )

        assistant_message = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message
    
    def generate_summary_report(self) -> str:
        """
        Generate a text summary report of all analysis performed
        """
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║           LS-DYNA CRASH ANALYSIS - EXECUTIVE SUMMARY              ║
╚════════════════════════════════════════════════════════════════════╝

Simulation ID: {self.simulation_data.simulation_id}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

─────────────────────────────────────────────────────────────────────
KEY METRICS SNAPSHOT
─────────────────────────────────────────────────────────────────────

Acceleration Performance:
  • Peak Acceleration (Resultant): {self.simulation_data.peak_acceleration_resultant:.1f} g
  • HIC Value: {self.simulation_data.hic_value:.0f}
  • NHTSA Limit: 1000
  • Status: {'✓ PASS' if self.simulation_data.hic_value < 1000 else '✗ FAIL'}

Energy Metrics:
  • Initial Kinetic Energy: {self.simulation_data.kinetic_energy_initial:.0f} J
  • Energy Dissipated: {self.simulation_data.kinetic_energy_dissipated_percent:.1f}%
  • Hourglass Energy: {self.simulation_data.hourglass_energy_percent:.1f}% (target: <5%)

Deformation:
  • Max Displacement: {self.simulation_data.max_displacement:.3f} m
  • Max Plastic Strain: {self.simulation_data.max_plastic_strain:.4f}
  • Element Deletions: {self.simulation_data.element_deletion_count}

─────────────────────────────────────────────────────────────────────
ANALYSIS CONVERSATION LOG
─────────────────────────────────────────────────────────────────────

{self._format_conversation()}

─────────────────────────────────────────────────────────────────────
NEXT STEPS
─────────────────────────────────────────────────────────────────────

1. Review the detailed recommendations above
2. Prioritize Tier 1 items for next design iteration
3. Run simulations on recommended parameter changes
4. Track improvements in metrics across iterations
5. Validate against physical testing when available

Report generated using Claude AI-powered crash analysis system
"""
        return report

    def generate_pdf_report(self, output_path: str) -> Optional[str]:
        """
        Generate a PDF file of the summary report at `output_path`.
        Returns the output path on success, or None if ReportLab is not installed.
        """
        if not _REPORTLAB_AVAILABLE:
            return None

        summary_text = self.generate_summary_report()
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        flow = []

        for line in summary_text.splitlines():
            if not line.strip():
                flow.append(Spacer(1, 6))
                continue
            para = Paragraph(line.replace(' ', '&nbsp;'), styles['Normal'])
            flow.append(para)
            flow.append(Spacer(1, 4))

        doc.build(flow)
        return output_path
    
    def _format_conversation(self) -> str:
        """Format conversation history for readability"""
        formatted = ""
        for i, msg in enumerate(self.conversation_history):
            if msg["role"] == "user":
                formatted += f"\n>>> USER QUERY {(i//2)+1}:\n{msg['content'][:500]}...\n"
            else:
                formatted += f"\n<< AI RESPONSE {(i//2)+1}:\n{msg['content'][:500]}...\n"
        return formatted


def run_example_analysis():
    """
    Run a complete example analysis workflow
    """
    
    # Example simulation data (in production, this comes from parsing d3plot/nodout/etc.)
    example_metrics = {
        "simulation_id": "FRONTAL_CRASH_TEST_NHTSA_V4_2",
        "timestamp": datetime.now().isoformat(),
        "peak_acceleration_x": 45.2,
        "peak_acceleration_y": 8.3,
        "peak_acceleration_z": 32.1,
        "peak_acceleration_resultant": 82.3,
        "time_to_peak_ms": 125,
        "hic_value": 980,
        "kinetic_energy_initial": 455000,
        "kinetic_energy_final": 125000,
        "kinetic_energy_dissipated_percent": 72.5,
        "internal_energy": 250000,
        "hourglass_energy_percent": 3.2,
        "max_displacement": 0.342,
        "max_plastic_strain": 0.145,
        "element_deletion_count": 342
    }
    
    # Initialize analyzer
    analyzer = LS_DYNA_Analyzer()
    analyzer.load_simulation_data(json.dumps(example_metrics))
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║  LS-DYNA CRASH ANALYSIS - INTELLIGENT SYSTEM          ║")
    print("╚════════════════════════════════════════════════════════╝\n")
    
    # Step 1: Initial Assessment
    print("Step 1: INITIAL ASSESSMENT")
    print("-" * 60)
    assessment = analyzer.initial_assessment()
    print(assessment)
    print("\n" + "="*60 + "\n")
    
    # Step 2: Failure Mode Analysis
    print("Step 2: FAILURE MODE ANALYSIS")
    print("-" * 60)
    failure_modes = analyzer.identify_failure_modes("acceleration and energy dissipation")
    print(failure_modes)
    print("\n" + "="*60 + "\n")
    
    # Step 3: Optimization Recommendations
    print("Step 3: OPTIMIZATION RECOMMENDATIONS")
    print("-" * 60)
    recommendations = analyzer.generate_optimization_recommendations()
    print(recommendations)
    print("\n" + "="*60 + "\n")
    
    # Step 4: Comparison with Baseline (if available)
    baseline_metrics = {
        "peak_acceleration_resultant": 95.0,
        "hic_value": 1150,
        "kinetic_energy_dissipated_percent": 62.0,
        "max_displacement": 0.395,
        "max_plastic_strain": 0.185,
        "element_deletion_count": 520
    }
    
    print("Step 4: BASELINE COMPARISON")
    print("-" * 60)
    comparison = analyzer.compare_with_baseline(baseline_metrics)
    print(comparison)
    print("\n" + "="*60 + "\n")
    
    # Generate summary
    print("ANALYSIS SUMMARY REPORT")
    print("-" * 60)
    summary = analyzer.generate_summary_report()
    print(summary)


if __name__ == "__main__":
    run_example_analysis()
