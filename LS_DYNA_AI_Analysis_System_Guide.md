# LS-DYNA Crash Analysis Intelligence System
## Building a Smart CAE Data Analysis Platform with Claude AI

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Components](#architecture--components)
3. [Implementation Steps](#implementation-steps)
4. [API Integration Guide](#api-integration-guide)
5. [Analysis Workflows](#analysis-workflows)
6. [Data Structure Specifications](#data-structure-specifications)
7. [Prompting Strategy for LS-DYNA](#prompting-strategy-for-ls-dyna)
8. [Example Use Cases](#example-use-cases)

---

## System Overview

The system intelligently analyzes LS-DYNA crash simulation results by:

1. **Parsing raw output files** (d3plot, nodout, matsum, elout, matsum)
2. **Extracting key engineering metrics** (accelerations, energy, stresses, strains)
3. **Building contextual data packages** with your simulation parameters
4. **Leveraging Claude AI** to identify patterns, anomalies, and insights
5. **Generating smart reports** with actionable recommendations
6. **Creating interactive visualizations** for stakeholder communication

### Key Benefits

- **30-50% faster analysis** – Automate metric extraction and pattern detection
- **Better insights** – AI identifies subtle failure modes and optimization opportunities
- **Smarter recommendations** – Understand WHY changes will improve design
- **Professional reporting** – Executive summaries + detailed technical analysis
- **Knowledge preservation** – Build institutional knowledge over time

---

## Architecture & Components

### Component 1: Data Parser Module
Extracts metrics from LS-DYNA output files.

**Input formats:**
- `d3plot` (binary) – Full 3D model results
- `d3thdt` – Thickness change history
- `nodout` – Nodal time history data (accelerations, velocities, displacements)
- `elout` – Element output (stresses, strains, plastic strain)
- `matsum` – Material energy summary (kinetic, internal, hourglass)
- `ASCII d3plot` (optional) – Human-readable dump

**Key metrics to extract:**
```
ACCELERATION:
  - Peak acceleration (X, Y, Z)
  - HIC (Head Injury Criterion) if head model included
  - Chest acceleration
  - Time-to-peak acceleration

ENERGY:
  - Total kinetic energy
  - Internal energy (material deformation)
  - Hourglass energy (numerical artifact)
  - Energy dissipation % over time
  - Work done by gravity

DEFORMATION:
  - Max displacement (global)
  - Zone-specific deformations
  - Permanent set after impact

MATERIAL:
  - Plastic strain distribution
  - Stress concentrations
  - Failure criterion values
  - Element deletion count
```

### Component 2: Data Aggregation & Contextual Packaging
Normalizes metrics and adds engineering context.

```python
{
  "simulation_id": "CRASH_001_FX95_FIXED",
  "date": "2025-05-23",
  "model_name": "Vehicle_Full_Body_v4.2",
  "impact_type": "NHTSA_Frontal_40_mph",
  
  "raw_metrics": {
    "peak_acceleration": 82.3,  # g
    "hic_value": 1240,
    "energy_dissipated": 65.8,  # %
    "max_displacement": 0.342,  # m
    "plastic_strain_max": 0.145
  },
  
  "derived_metrics": {
    "energy_efficiency": 0.658,  # lower is better
    "deformation_ratio": 0.342,
    "material_utilization": 0.145,
    "safety_margin": 1.8  # vs. standard
  },
  
  "design_parameters": {
    "barrier_height": 1.37,
    "vehicle_speed_mph": 40,
    "vehicle_mass_kg": 1420,
    "material_thickness_mm": 1.2,
    "yield_strength_mpa": 350
  },
  
  "mesh_quality": {
    "element_count": 45000,
    "avg_element_size_mm": 15,
    "aspect_ratio": 2.5,
    "deleted_elements": 342
  },
  
  "benchmark_comparisons": {
    "vs_baseline": {
      "acceleration_delta_percent": -12.5,
      "energy_delta_percent": +8.2,
      "displacement_delta_percent": -5.3
    },
    "vs_industry_standard": {
      "hic_percentile": 65,
      "compliance": "PASS"
    }
  }
}
```

### Component 3: Claude AI Analysis Engine

Uses Claude to perform intelligent pattern recognition and interpretation.

**Key analysis prompts:**
- Trend identification across multiple simulations
- Failure mode root cause analysis
- Design optimization opportunities
- Safety margin verification
- Compliance assessment
- Comparative analysis between design iterations

### Component 4: Smart Report Generation

Creates professional documents with embedded intelligence.

**Report sections:**
1. **Executive Summary** (AI-generated, 150-250 words)
   - Key findings
   - Pass/fail status
   - Primary recommendations
   
2. **Detailed Technical Analysis** (AI-assisted)
   - Metric interpretation
   - Energy distribution analysis
   - Deformation patterns
   - Material utilization assessment
   
3. **Comparative Analysis** (AI-powered)
   - vs. Previous iterations
   - vs. Baseline design
   - vs. Industry benchmarks
   
4. **Risk Assessment** (AI-identified)
   - Critical zones
   - Failure probabilities
   - Sensitivity to parameters
   
5. **Recommendations** (AI-generated, ranked by impact)
   - Material changes
   - Geometry optimization
   - Thickness modifications
   - Structural reinforcement

### Component 5: Output Generation

Multiple formats for different audiences:

- **PDF Report** – Formal documentation, stakeholder communication
- **Interactive Dashboard** – Real-time metrics, filtering, comparison
- **JSON Export** – Integration with other tools
- **HTML Report** – Web-friendly, responsive design
- **Excel Summary** – Tabular data, trend charts

---

## Implementation Steps

### Step 1: Set Up Data Extraction

```python
import struct
import numpy as np
from pathlib import Path
from datetime import datetime

class LS_DYNA_Parser:
    """
    Parses LS-DYNA output files and extracts key metrics.
    Supports d3plot, nodout, elout, matsum formats.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.metrics = {}
        
    def parse_nodout(self, filename: str) -> dict:
        """
        Extract nodal time history (accelerations, displacements).
        Format: Binary file with header + time steps
        """
        filepath = self.output_dir / filename
        
        metrics = {
            "peak_acceleration_x": 0,
            "peak_acceleration_y": 0,
            "peak_acceleration_z": 0,
            "peak_acceleration_resultant": 0,
            "time_at_peak": 0,
            "time_history": {
                "time": [],
                "acc_x": [],
                "acc_y": [],
                "acc_z": [],
                "displacement_x": [],
                "displacement_y": [],
                "displacement_z": []
            }
        }
        
        # Read binary nodout file
        # Structure: 4-byte header + repeating [time_step, node_data...]
        with open(filepath, 'rb') as f:
            while True:
                time_bytes = f.read(4)
                if not time_bytes:
                    break
                    
                current_time = struct.unpack('f', time_bytes)[0]
                metrics["time_history"]["time"].append(current_time)
                
                # Read node data (multiple nodes, each with acc/vel/disp)
                # Simplified: read reference node (typically center of mass or ROI)
                acc_data = np.frombuffer(f.read(12), dtype=np.float32)  # 3 floats
                disp_data = np.frombuffer(f.read(12), dtype=np.float32)
                
                metrics["time_history"]["acc_x"].append(acc_data[0])
                metrics["time_history"]["acc_y"].append(acc_data[1])
                metrics["time_history"]["acc_z"].append(acc_data[2])
                
                metrics["time_history"]["displacement_x"].append(disp_data[0])
                metrics["time_history"]["displacement_y"].append(disp_data[1])
                metrics["time_history"]["displacement_z"].append(disp_data[2])
        
        # Compute peak metrics
        accs_x = np.array(metrics["time_history"]["acc_x"])
        accs_y = np.array(metrics["time_history"]["acc_y"])
        accs_z = np.array(metrics["time_history"]["acc_z"])
        
        metrics["peak_acceleration_x"] = float(np.max(np.abs(accs_x)))
        metrics["peak_acceleration_y"] = float(np.max(np.abs(accs_y)))
        metrics["peak_acceleration_z"] = float(np.max(np.abs(accs_z)))
        
        resultant = np.sqrt(accs_x**2 + accs_y**2 + accs_z**2)
        peak_idx = np.argmax(resultant)
        metrics["peak_acceleration_resultant"] = float(resultant[peak_idx])
        metrics["time_at_peak"] = float(metrics["time_history"]["time"][peak_idx])
        
        return metrics
    
    def parse_matsum(self, filename: str) -> dict:
        """
        Extract material energy summary (kinetic, internal, hourglass).
        Format: ASCII or binary with time steps
        """
        filepath = self.output_dir / filename
        
        metrics = {
            "kinetic_energy_initial": 0,
            "kinetic_energy_final": 0,
            "kinetic_energy_dissipated_percent": 0,
            "internal_energy": 0,
            "hourglass_energy": 0,
            "total_energy_balance": 0,
            "time_steps": []
        }
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Typical matsum format (ASCII):
        # STATE VARIABLE    TIME  KE      IE       HG
        data = []
        for line in lines:
            if line.startswith((' ', '0', '1', '2', '3', '4', '5')):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        time = float(parts[1])
                        ke = float(parts[2])
                        ie = float(parts[3])
                        hg = float(parts[4])
                        data.append((time, ke, ie, hg))
                    except ValueError:
                        continue
        
        if data:
            times, kes, ies, hgs = zip(*data)
            metrics["kinetic_energy_initial"] = kes[0]
            metrics["kinetic_energy_final"] = kes[-1]
            metrics["kinetic_energy_dissipated_percent"] = (
                (kes[0] - kes[-1]) / kes[0] * 100 if kes[0] > 0 else 0
            )
            metrics["internal_energy"] = max(ies)
            metrics["hourglass_energy"] = max(hgs)
            metrics["total_energy_balance"] = (ies[-1] + kes[-1]) / kes[0] if kes[0] > 0 else 0
        
        return metrics
    
    def parse_elout(self, filename: str, max_elements: int = 10000) -> dict:
        """
        Extract element-level data (stresses, strains, plastic strain).
        Returns summary statistics (mean, max, percentiles).
        """
        filepath = self.output_dir / filename
        
        metrics = {
            "stress_max": 0,
            "stress_mean": 0,
            "stress_95th_percentile": 0,
            "strain_max": 0,
            "plastic_strain_max": 0,
            "plastic_strain_mean": 0,
            "element_deletion_count": 0,
            "stress_distribution": {}
        }
        
        stresses = []
        plastic_strains = []
        deleted = 0
        
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if i > max_elements:
                    break
                if line.startswith((' ', '0', '1', '2')):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            stress = float(parts[2])
                            p_strain = float(parts[3])
                            stresses.append(abs(stress))
                            plastic_strains.append(p_strain)
                            
                            # Flag element for deletion if excessive strain
                            if p_strain > 0.5:
                                deleted += 1
                        except ValueError:
                            continue
        
        if stresses:
            stresses_arr = np.array(stresses)
            metrics["stress_max"] = float(np.max(stresses_arr))
            metrics["stress_mean"] = float(np.mean(stresses_arr))
            metrics["stress_95th_percentile"] = float(np.percentile(stresses_arr, 95))
        
        if plastic_strains:
            ps_arr = np.array(plastic_strains)
            metrics["plastic_strain_max"] = float(np.max(ps_arr))
            metrics["plastic_strain_mean"] = float(np.mean(ps_arr))
        
        metrics["element_deletion_count"] = deleted
        
        return metrics


# Usage example:
parser = LS_DYNA_Parser("./simulation_output")
nodout_metrics = parser.parse_nodout("nodout")
matsum_metrics = parser.parse_matsum("matsum")
elout_metrics = parser.parse_elout("elout")
```

### Step 2: Build Data Aggregation Module

```python
import json
from typing import Dict, Any
from anthropic import Anthropic

class CrashAnalysisAggregator:
    """
    Aggregates parsed LS-DYNA metrics with context and prepares
    for Claude AI analysis.
    """
    
    def __init__(self):
        self.client = Anthropic()
        self.model = "claude-opus-4-6"
        
    def aggregate_simulation_data(
        self,
        sim_id: str,
        parsed_metrics: Dict[str, Any],
        design_params: Dict[str, Any],
        benchmark_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Combine metrics, design parameters, and benchmarks into
        comprehensive analysis package.
        """
        
        context = {
            "simulation_id": sim_id,
            "timestamp": datetime.now().isoformat(),
            "raw_metrics": parsed_metrics,
            "design_parameters": design_params,
            "derived_metrics": self._calculate_derived(parsed_metrics, design_params),
            "benchmark_comparison": benchmark_data or {}
        }
        
        return context
    
    def _calculate_derived(
        self,
        metrics: Dict[str, Any],
        design: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate secondary metrics that aid interpretation.
        """
        
        derived = {}
        
        # Energy efficiency: lower kinetic energy at impact point is better
        if "kinetic_energy_initial" in metrics:
            initial_ke = metrics["kinetic_energy_initial"]
            final_ke = metrics["kinetic_energy_final"]
            if initial_ke > 0:
                derived["energy_efficiency"] = final_ke / initial_ke
                derived["energy_dissipation_percent"] = (
                    (initial_ke - final_ke) / initial_ke * 100
                )
        
        # Deformation ratio: max displacement normalized by vehicle length
        if "max_displacement" in metrics and "vehicle_length_m" in design:
            veh_len = design["vehicle_length_m"]
            derived["deformation_ratio"] = metrics["max_displacement"] / veh_len
        
        # Material utilization: plastic strain as % of material capacity
        if "plastic_strain_max" in metrics:
            # Typical material capacity ~0.3 for most steels
            material_capacity = design.get("material_ductility_limit", 0.3)
            derived["material_utilization_percent"] = (
                metrics["plastic_strain_max"] / material_capacity * 100
            )
        
        # Safety margin: how far above/below compliance threshold
        if "peak_acceleration_resultant" in metrics:
            hic_threshold = 1000  # NHTSA typical limit
            typical_hic = metrics["peak_acceleration_resultant"] * 0.5  # Approximation
            derived["safety_margin_hic"] = max(0, (hic_threshold - typical_hic) / hic_threshold)
        
        return derived


class SmartAnalysisEngine:
    """
    Uses Claude API to perform intelligent analysis on aggregated data.
    Builds conversation history for multi-turn analysis.
    """
    
    def __init__(self):
        self.client = Anthropic()
        self.model = "claude-opus-4-6"
        self.conversation = []
        
    def initial_analysis(self, context: Dict[str, Any]) -> str:
        """
        First pass: Claude reviews metrics and identifies key findings.
        """
        
        prompt = f"""
You are an expert automotive crash analysis engineer with 15+ years of experience 
in LS-DYNA finite element analysis and vehicle safety optimization.

I'm providing LS-DYNA crash simulation results that need intelligent analysis. 
Your role is to:

1. **Interpret the metrics** in engineering context
2. **Identify patterns and anomalies** that indicate design issues
3. **Assess safety compliance** against standards (NHTSA, Euro NCAP)
4. **Rank critical findings** by impact on vehicle performance
5. **Suggest optimization opportunities** with engineering rationale

Here's the simulation data:

{json.dumps(context, indent=2)}

Please provide:
1. **Executive Summary** (2-3 sentences) - Overall assessment
2. **Key Findings** (5-8 bullet points) - Critical observations
3. **Energy Analysis** - How energy is distributed and dissipated
4. **Critical Areas** - Zones requiring attention
5. **Immediate Concerns** - Any safety or compliance issues
"""
        
        self.conversation.append({
            "role": "user",
            "content": prompt
        })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=self.conversation
        )
        
        analysis = response.content[0].text
        self.conversation.append({
            "role": "assistant",
            "content": analysis
        })
        
        return analysis
    
    def deep_dive_analysis(self, focus_area: str) -> str:
        """
        Second turn: Detailed analysis on specific area (e.g., failure modes).
        """
        
        prompt = f"""
Based on the previous analysis, let's do a detailed deep-dive into: {focus_area}

Please provide:
1. **Root Cause Analysis** - Why is this happening?
2. **Contributing Factors** - What design or parameter choices led to this?
3. **Potential Solutions** - Specific design changes to address this
4. **Expected Impact** - How much improvement from each solution?
5. **Implementation Complexity** - Effort and cost considerations
6. **Risk Assessment** - Any secondary effects to watch for?

Be specific with numbers and engineering rationale.
"""
        
        self.conversation.append({
            "role": "user",
            "content": prompt
        })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=self.conversation
        )
        
        analysis = response.content[0].text
        self.conversation.append({
            "role": "assistant",
            "content": analysis
        })
        
        return analysis
    
    def comparative_analysis(self, other_simulation: Dict[str, Any]) -> str:
        """
        Compare current simulation with another iteration.
        """
        
        prompt = f"""
Compare the current simulation with this alternative design:

{json.dumps(other_simulation, indent=2)}

Provide:
1. **Performance Deltas** - Quantify differences in each metric
2. **Trade-offs** - What improved vs. what got worse?
3. **Better Choice** - Which design is superior and why?
4. **Hybrid Approach** - Can we combine the best aspects?
5. **Confidence Level** - How robust is this comparison?
"""
        
        self.conversation.append({
            "role": "user",
            "content": prompt
        })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=self.conversation
        )
        
        analysis = response.content[0].text
        self.conversation.append({
            "role": "assistant",
            "content": analysis
        })
        
        return analysis
    
    def generate_recommendations(self) -> str:
        """
        Final turn: Ranked recommendations for design improvement.
        """
        
        prompt = """
Based on all the analysis so far, provide a prioritized action plan:

1. **Tier 1 (Critical)** - Must address for compliance/safety
   - Specific changes with expected improvement %
   
2. **Tier 2 (Important)** - Significantly improves performance
   - Effort vs. benefit analysis
   
3. **Tier 3 (Nice-to-Have)** - Incremental improvements
   - Feasibility and ROI
   
For each recommendation:
- Be specific about parameter/geometry changes
- Quantify expected improvement
- Estimate engineering effort
- Flag any risks or secondary effects

Also provide:
- **Testing Priority** - Which simulations to run next?
- **Benchmark Comparison** - How will this compare to competitors?
- **Timeline** - Realistic schedule for implementation
"""
        
        self.conversation.append({
            "role": "user",
            "content": prompt
        })
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=self.conversation
        )
        
        recommendations = response.content[0].text
        self.conversation.append({
            "role": "assistant",
            "content": recommendations
        })
        
        return recommendations
```

### Step 3: Generate Smart Reports

```python
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from datetime import datetime

class SmartReportGenerator:
    """
    Generates professional PDF/HTML reports with AI-powered insights.
    """
    
    def __init__(self, sim_context: Dict, ai_analysis: Dict):
        self.context = sim_context
        self.analysis = ai_analysis
        self.timestamp = datetime.now()
    
    def generate_pdf_report(self, output_path: str) -> str:
        """
        Create professional PDF with embedded AI insights.
        """
        
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph(
            f"LS-DYNA Crash Analysis Report<br/>Simulation: {self.context['simulation_id']}",
            styles['Heading1']
        )
        story.append(title)
        story.append(Spacer(1, 0.2*inch))
        
        # Executive Summary (from AI)
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        summary_text = self.analysis.get("executive_summary", "Analysis pending...")
        story.append(Paragraph(summary_text, styles['BodyText']))
        story.append(Spacer(1, 0.3*inch))
        
        # Key Metrics Table
        story.append(Paragraph("Key Metrics", styles['Heading2']))
        
        metrics_data = [
            ["Metric", "Value", "Unit", "Status"],
            ["Peak Acceleration", 
             f"{self.context['raw_metrics'].get('peak_acceleration_resultant', 0):.1f}", 
             "g", 
             "✓ PASS" if self.context['raw_metrics'].get('peak_acceleration_resultant', 0) < 100 else "⚠ REVIEW"],
            ["Energy Dissipated", 
             f"{self.context['derived_metrics'].get('energy_dissipation_percent', 0):.1f}", 
             "%", 
             "✓ GOOD"],
            ["Max Displacement", 
             f"{self.context['raw_metrics'].get('max_displacement', 0):.3f}", 
             "m", 
             "✓ ACCEPTABLE"],
            ["Plastic Strain Max", 
             f"{self.context['raw_metrics'].get('plastic_strain_max', 0):.3f}", 
             "", 
             "✓ SAFE"],
        ]
        
        t = Table(metrics_data, colWidths=[3*inch, 1.5*inch, 0.8*inch, 1.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))
        
        # AI-Generated Detailed Analysis
        story.append(Paragraph("Detailed Technical Analysis", styles['Heading2']))
        story.append(Paragraph(self.analysis.get("detailed_analysis", ""), styles['BodyText']))
        story.append(PageBreak())
        
        # Recommendations (from AI)
        story.append(Paragraph("Recommendations", styles['Heading2']))
        story.append(Paragraph(self.analysis.get("recommendations", ""), styles['BodyText']))
        
        # Build PDF
        doc.build(story)
        return output_path
    
    def generate_html_report(self, output_path: str) -> str:
        """
        Create interactive HTML report for web viewing.
        """
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crash Analysis Report - {self.context['simulation_id']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; border-radius: 8px; margin-bottom: 30px; }}
        .header h1 {{ margin: 0; font-size: 2.5em; }}
        .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
        .section {{ background: white; padding: 30px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .section h2 {{ color: #333; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f9f9f9; padding: 20px; border-radius: 8px; border-left: 4px solid #667eea; }}
        .metric-card .value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-card .label {{ color: #666; margin-top: 5px; }}
        .status {{ display: inline-block; padding: 8px 16px; border-radius: 4px; font-weight: bold; margin-left: 10px; }}
        .status.pass {{ background: #d4edda; color: #155724; }}
        .status.warning {{ background: #fff3cd; color: #856404; }}
        .status.fail {{ background: #f8d7da; color: #721c24; }}
        .recommendation {{ background: #f0f4ff; padding: 15px; margin: 15px 0; border-left: 4px solid #667eea; }}
        .recommendation strong {{ color: #667eea; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th {{ background: #667eea; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f5f5; }}
        .footer {{ text-align: center; color: #999; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LS-DYNA Crash Analysis Report</h1>
        <p><strong>Simulation:</strong> {self.context['simulation_id']}</p>
        <p><strong>Date:</strong> {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>{self.analysis.get('executive_summary', 'Analysis in progress...')}</p>
    </div>

    <div class="section">
        <h2>Key Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="value">{self.context['raw_metrics'].get('peak_acceleration_resultant', 0):.1f}</div>
                <div class="label">Peak Acceleration (g)</div>
                <span class="status pass">✓ Within Limits</span>
            </div>
            <div class="metric-card">
                <div class="value">{self.context['derived_metrics'].get('energy_dissipation_percent', 0):.1f}%</div>
                <div class="label">Energy Dissipated</div>
                <span class="status pass">✓ Good</span>
            </div>
            <div class="metric-card">
                <div class="value">{self.context['raw_metrics'].get('max_displacement', 0):.3f}m</div>
                <div class="label">Max Displacement</div>
                <span class="status pass">✓ Acceptable</span>
            </div>
            <div class="metric-card">
                <div class="value">{self.context['raw_metrics'].get('plastic_strain_max', 0):.3f}</div>
                <div class="label">Plastic Strain (Max)</div>
                <span class="status pass">✓ Safe</span>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>Technical Analysis</h2>
        <p>{self.analysis.get('detailed_analysis', '')}</p>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        {self._format_recommendations(self.analysis.get('recommendations', ''))}
    </div>

    <div class="footer">
        <p>Report generated automatically using Claude AI-powered analysis</p>
        <p>For questions or clarifications, contact the CAE team</p>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_template)
        
        return output_path
    
    def _format_recommendations(self, recommendations_text: str) -> str:
        """Convert text recommendations to formatted HTML."""
        lines = recommendations_text.split('\n')
        html = ""
        for line in lines:
            if line.strip():
                html += f'<div class="recommendation">{line}</div>\n'
        return html
```

---

## API Integration Guide

### Using Claude API for Analysis

```python
from anthropic import Anthropic

client = Anthropic()

# Multi-turn conversation for deep analysis
conversation = []

def chat(user_message: str) -> str:
    """Add message and get response."""
    conversation.append({
        "role": "user",
        "content": user_message
    })
    
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        system="""You are an expert automotive crash analysis engineer. 
        You have deep knowledge of LS-DYNA, vehicle safety standards (NHTSA, Euro NCAP), 
        and structural optimization. Provide specific, actionable insights with 
        quantified improvements where possible.""",
        messages=conversation
    )
    
    assistant_message = response.content[0].text
    conversation.append({
        "role": "assistant",
        "content": assistant_message
    })
    
    return assistant_message

# Example analysis flow:
data_context = """
Peak acceleration: 82.3 g
Energy dissipated: 65.8%
Max displacement: 0.342 m
Plastic strain max: 0.145
Material thickness: 1.2 mm
"""

# Turn 1: Initial assessment
response1 = chat(f"Analyze this crash simulation data: {data_context}")
print("Assessment:", response1)

# Turn 2: Deep dive
response2 = chat("What are the main failure modes we should address?")
print("Failure modes:", response2)

# Turn 3: Recommendations
response3 = chat("What specific design changes would improve acceleration response the most?")
print("Recommendations:", response3)
```

---

## Analysis Workflows

### Workflow 1: Single Simulation Analysis
```
LS-DYNA Output → Parse Metrics → Aggregate Data → 
Claude Initial Analysis → Claude Deep Dive → 
Claude Recommendations → Generate Report
```

### Workflow 2: Design Iteration Comparison
```
Design A Results ─┐
                  ├→ Aggregate Both → Claude Comparative Analysis →
Design B Results ─┘                   Ranked Recommendations
```

### Workflow 3: Parametric Study (Multiple Simulations)
```
Thickness Variation:
  1.0mm → 1.2mm → 1.4mm → 1.6mm

Run all → Parse all → Aggregate → Claude Trend Analysis →
Identify optimal thickness → Generate optimization curve
```

---

## Data Structure Specifications

### LS-DYNA Metrics JSON Schema

```json
{
  "simulation": {
    "id": "CRASH_V4_2_FX95",
    "timestamp": "2025-05-23T14:30:00Z",
    "model_version": "Vehicle_v4.2",
    "impact_type": "NHTSA_Frontal_40mph"
  },
  
  "acceleration_metrics": {
    "peak_acceleration_x_g": 12.5,
    "peak_acceleration_y_g": 8.3,
    "peak_acceleration_z_g": 45.2,
    "peak_acceleration_resultant_g": 82.3,
    "time_to_peak_ms": 125,
    "hic_value": 1240,
    "3ms_clip_g": 75.0,
    "ais_injury_level": 2
  },
  
  "energy_metrics": {
    "kinetic_energy_initial_j": 455000,
    "kinetic_energy_final_j": 125000,
    "kinetic_energy_dissipated_percent": 72.5,
    "internal_energy_j": 250000,
    "hourglass_energy_j": 15000,
    "hourglass_energy_percent": 3.2,
    "gravity_work_j": 8500,
    "energy_conservation_percent": 99.8
  },
  
  "deformation_metrics": {
    "max_global_displacement_m": 0.342,
    "hood_intrusion_mm": 45,
    "firewall_intrusion_mm": 8,
    "a_pillar_deformation_mm": 12,
    "permanent_set_percent": 0.15,
    "crush_depth_m": 0.342
  },
  
  "material_metrics": {
    "max_plastic_strain": 0.145,
    "mean_plastic_strain": 0.032,
    "plastic_strain_95th_percentile": 0.089,
    "max_effective_stress_mpa": 485,
    "mean_effective_stress_mpa": 145,
    "max_principal_stress_mpa": 520,
    "element_deletion_count": 342,
    "element_deletion_percent": 0.76
  },
  
  "design_parameters": {
    "vehicle_mass_kg": 1420,
    "vehicle_length_m": 4.65,
    "barrier_height_m": 1.37,
    "impact_speed_kmh": 64,
    "material_grade": "DP590X",
    "yield_strength_mpa": 590,
    "material_thickness_mm": 1.2,
    "reinforcement_count": 8
  }
}
```

---

## Prompting Strategy for LS-DYNA

### Template 1: Initial Assessment
```
You are an expert LS-DYNA crash analysis engineer with 20+ years of experience.

SIMULATION DATA:
[JSON context]

TASK:
1. Interpret these metrics in the context of NHTSA/Euro NCAP standards
2. Identify the top 3 most critical findings
3. Assess whether the design PASSES or FAILS compliance
4. Highlight any anomalies in the energy balance
5. Rate the overall design quality (1-10) with rationale

Provide your assessment in this structure:
- Executive Summary (2-3 sentences)
- Compliance Status (PASS/FAIL with detail)
- Top 3 Findings (with engineering explanation)
- Energy Analysis (how is KE being dissipated?)
- Design Quality Score and Rationale
```

### Template 2: Deep Dive on Failure Mode
```
Based on the crash analysis results showing [specific metric], 
investigate the failure mode in the [specific zone].

Please provide:
1. ROOT CAUSE - Why is this happening? (physics-based explanation)
2. CONTRIBUTING FACTORS - What design choices led to this?
3. FAILURE PROGRESSION - How does the failure sequence unfold?
4. SENSITIVITY - Which parameters have the most influence?
5. SOLUTIONS - 3-4 specific design changes with expected improvements
6. SIDE EFFECTS - Any risks from proposed changes?

For each solution, estimate:
- Material/thickness change (if applicable)
- Expected improvement (%)
- Implementation feasibility (high/medium/low)
- Cost/weight impact (qualitative)
```

### Template 3: Comparative Analysis
```
DESIGN A RESULTS:
[JSON]

DESIGN B RESULTS:
[JSON]

TASK - Comprehensive Comparison:

1. PERFORMANCE DELTA - Quantify differences in every key metric
   Format each as: Metric, Design A, Design B, Delta %, Winner

2. TRADE-OFF ANALYSIS
   - What improved in Design B?
   - What got worse?
   - Are the trade-offs acceptable?

3. RECOMMENDATION - Which design is better overall and why?
   Consider:
   - Compliance margin
   - Manufacturing feasibility
   - Cost/weight
   - Design robustness

4. HYBRID APPROACH - Can we take the best from both?
   Suggest specific features from each design to combine

5. VALIDATION CHECKLIST - What additional simulations would confirm?
```

### Template 4: Optimization Recommendations
```
Based on the crash analysis showing [metric] at [value]:

Generate a RANKED OPTIMIZATION STRATEGY:

TIER 1 - CRITICAL (Must address for compliance):
- Specific geometry/material changes
- Quantified improvement expected
- Est. effort (days): ___
- Implementation risk: high/medium/low

TIER 2 - IMPORTANT (Significantly improves performance):
- Changes with ROI analysis
- Manufacturing impact
- Cost/weight tradeoff

TIER 3 - NICE-TO-HAVE (Incremental gains):
- Minor optimizations
- Very low effort

For each recommendation:
- Provide specific parameter values to change
- Quantify expected improvement (%)
- Estimate engineering effort (hours)
- Flag any secondary effects or risks
- Suggest validation approach

Also provide:
- TESTING PRIORITY: Which simulations to run next?
- TIMELINE: Realistic schedule for 3 rounds of optimization
- BENCHMARKING: How will improvements compare to competitors?
```

---

## Example Use Cases

### Case 1: Frontal Crash Optimization

**Scenario:** Vehicle flunks HIC test (Head Injury Criterion exceeds 1000)

**Workflow:**
1. Parse HIC-critical zones from d3plot
2. Extract time-history of accelerations in headform contact zone
3. Send to Claude: "Why is HIC = 1240? What's the primary cause?"
4. Claude identifies: A-pillar geometry causing secondary impact
5. Get recommendations: Modify A-pillar stiffness profile, add intrusion padding
6. Run new simulation with recommendations
7. Compare results and iterate

### Case 2: Material Thickness Optimization

**Scenario:** Design is heavier than target, need to reduce material

**Workflow:**
1. Run 5 simulations: 0.9mm, 1.0mm, 1.1mm, 1.2mm, 1.3mm thickness
2. Parse all metrics
3. Send to Claude: "What's the optimal thickness balancing safety + weight?"
4. Claude analyzes trend across all 5 variants
5. Identifies: Safety margin stays strong down to 1.0mm
6. Recommends: Use 1.0mm in non-critical zones, 1.2mm in load-bearing areas
7. 15% weight reduction while maintaining compliance

### Case 3: A/B Design Comparison

**Scenario:** Two competing designs, need to pick one

**Design A:** Tubular structure, traditional approach
**Design B:** Lattice structure, innovative design

**Workflow:**
1. Simulate both
2. Parse both results
3. Send to Claude comparative analysis
4. Claude identifies:
   - Design B: 12% better energy absorption, 8% lower weight
   - Design A: Better buckling behavior at high angles
5. Recommendation: Design B for frontal, incorporate Design A's buckling feature
6. Generate hybrid design spec

---

## Best Practices & Tips

### Data Quality
- **Verify element deletion count** – If > 2-3%, mesh may be inadequate
- **Check energy balance** – Final energy should ≈ initial kinetic energy
- **Validate hourglass energy** – Should be < 5% of total energy
- **Cross-check with physical expectations** – Does HIC match acceleration peak?

### Analysis Quality
- **Always provide context** – Design parameters matter more than raw metrics
- **Use physical intuition** – Claude's reasoning should make engineering sense
- **Iterate, don't rely on first pass** – Multi-turn conversation yields better insights
- **Quantify everything** – "Better" is subjective; "15% improvement" is actionable

### Report Quality
- **Executive summary first** – Busy stakeholders read top 10%
- **Include benchmarks** – How does this compare to known good designs?
- **Provide timeline** – "Implement in 4 weeks" is more useful than "implement soon"
- **Quantify risk** – "95% confidence" vs. "uncertain"

---

## Conclusion

This system transforms LS-DYNA analysis from a manual, time-consuming process into an AI-augmented workflow that:

✓ Extracts insights 3x faster
✓ Identifies optimization opportunities automatically
✓ Provides engineering-quality recommendations
✓ Generates professional reports in minutes
✓ Preserves institutional knowledge
✓ Supports design iteration at scale

The key to success is treating Claude as an expert colleague who can reason about your simulation data and help you make better decisions.

