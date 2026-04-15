import base64
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

def encode_image(filepath):
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def build_html():
    print("Reading data assets...")
    metrics_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'metrics_table.csv'))
    params_df = pd.read_csv(os.path.join(PROCESSED_DIR, 'learned_params.csv'))
    
    comp_img_b64 = encode_image(os.path.join(PROCESSED_DIR, 'comparison.png'))
    r0_img_b64 = encode_image(os.path.join(PROCESSED_DIR, 'r0_trajectory.png'))
    
    best_mae = metrics_df['MAE'].min()
    best_rmse = metrics_df['RMSE'].min()
    dataset_size = len(params_df)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural ODE Epidemic Simulator</title>
    <style>
        :root {{
            --bg-color: #ffffff;
            --text-color: #333333;
            --muted-text: #666666;
            --card-bg: #f9f9f9;
            --border-color: #eeeeee;
            --accent: #2563eb;
            --hover: #1e40af;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #121212;
                --text-color: #e0e0e0;
                --muted-text: #a0a0a0;
                --card-bg: #1e1e1e;
                --border-color: #333333;
                --accent: #3b82f6;
                --hover: #60a5fa;
            }}
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }}
        
        .container {{ max-width: 900px; margin: 0 auto; padding: 2rem 1rem; }}
        
        h1, h2, h3 {{ color: var(--text-color); margin-top: 2rem; }}
        h1 {{ font-size: 2.5rem; text-align: center; margin-bottom: 0.5rem; }}
        .subtitle {{ text-align: center; color: var(--muted-text); font-size: 1.2rem; margin-bottom: 2rem; }}
        
        .stats {{ display: flex; justify-content: center; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }}
        .stat-pill {{ background-color: var(--card-bg); border: 1px solid var(--border-color); padding: 0.5rem 1rem; border-radius: 999px; font-weight: 600; font-size: 0.9rem; }}
        
        .btn {{ display: block; width: max-content; margin: 0 auto 3rem auto; background-color: var(--accent); color: white; text-decoration: none; padding: 0.75rem 1.5rem; border-radius: 6px; font-weight: bold; transition: background 0.2s; }}
        .btn:hover {{ background-color: var(--hover); }}
        
        .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 3rem; }}
        @media (max-width: 768px) {{ .panels {{ grid-template-columns: 1fr; }} }}
        .panel {{ background-color: var(--card-bg); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-color); }}
        
        .chart {{ width: 100%; border-radius: 8px; margin: 1rem 0; border: 1px solid var(--border-color); }}
        
        table {{ width: 100%; border-collapse: collapse; margin: 2rem 0; background-color: var(--card-bg); font-size: 0.95rem; }}
        th, td {{ padding: 1rem; text-align: left; border-bottom: 1px solid var(--border-color); }}
        th {{ background-color: var(--card-bg); font-weight: 600; border-bottom: 2px solid var(--border-color);}}
        
        details {{ background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 6px; margin-bottom: 1rem; padding: 0.5rem 1rem; }}
        summary {{ font-weight: bold; cursor: pointer; outline: none; }}
        details p {{ margin-top: 1rem; color: var(--muted-text); }}
        
        pre {{ background-color: var(--card-bg); padding: 1rem; border-radius: 6px; overflow-x: auto; border: 1px solid var(--border-color); }}
        code {{ font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <!-- SECTION 1: HERO -->
        <h1>Neural ODE Epidemic Simulator</h1>
        <p class="subtitle">A scientific machine learning system forecasting epidemic spread using Universal Differential Equations.</p>
        
        <div class="stats">
            <span class="stat-pill">Best MAE: {best_mae:.6f}</span>
            <span class="stat-pill">Best RMSE: {best_rmse:.6f}</span>
            <span class="stat-pill">Dataset: {dataset_size} Weeks</span>
        </div>
        
        <a href="https://github.com/angadjg07/neural-ode-epidemic" class="btn" target="_blank">View on GitHub</a>
        
        <!-- SECTION 2: THE SCIENCE -->
        <h2>The Science, Briefly</h2>
        <div class="panels">
            <div class="panel">
                <h3>Classical SIR Model</h3>
                <p>The classical mathematical SIR model segregates a population into Susceptible, Infected, and Recovered compartments. It relies on strictly fixed transition parameters: &beta; (transmission rate) and &gamma; (recovery rate). While excellent for theoretical analysis, assuming these parameters never naturally fluctuate drastically limits the model's accuracy on real datasets spanning multiple virus variants.</p>
            </div>
            <div class="panel">
                <h3>Neural ODE + Hybrid UDE</h3>
                <p>A Neural ODE replaces layer transitions with continuous-time integration via PyTorch's <code>torchdiffeq</code> solvers. Expanding upon this, our Hybrid Universal Differential Equation (UDE) keeps the physical equations explicitly intact while delegating localized time-varying corrections &mdash; &beta;(t) and &gamma;(t) &mdash; to a fully integrated, dynamically trained neural network.</p>
            </div>
        </div>
        
        <!-- SECTION 3: MAIN RESULTS -->
        <h2>Model Performance Comparison</h2>
        <img src="data:image/png;base64,{comp_img_b64}" alt="Forecast Comparison Chart" class="chart">
        
        {metrics_df.to_html(index=False).replace('<table border="1" class="dataframe">', '<table>')}
        <p style="text-align:center; color:var(--muted-text);"><em>The Hybrid UDE systematically mitigates test set trajectory deviation by implicitly modeling population behavioral variations over time.</em></p>
        
        <!-- SECTION 4: WHAT THE MODEL LEARNED -->
        <h2>What The Network Learned</h2>
        <img src="data:image/png;base64,{r0_img_b64}" alt="R0 Trajectory Chart" class="chart">
        <p>The parameter <strong>R&#8320;(t)</strong> indicates the instantaneous Basic Reproduction Number. When R&#8320;(t) > 1.0, the epidemic curve technically experiences exponential momentum and expands. When it drops beneath 1.0, cases are declining. Notably, our parameterized model implicitly discovered this periodic wave threshold dynamic strictly from chronological case fractions, bypassing any manual heuristics or explicit timeline dates.</p>
        
        <!-- SECTION 5: SCIML CONCEPTS -->
        <h2>SciML Concepts Explained</h2>
        <details>
            <summary>What is a Neural ODE?</summary>
            <p>Instead of defining hidden network steps iteratively, Neural Ordinary Differential Equations utilize a continuous differential equation solver to evaluate how a hidden state organically evolves. To backpropagate through an ODE solver without explosive memory overhead, researchers employ the adjoint sensitivity method.</p>
        </details>
        <details>
            <summary>What is a Physics-Informed model?</summary>
            <p>Physics-Informed Neural Networks (PINNs) guide structural learning using established scientific principles. Instead of granting a neural model blanket predictive freedom, we strictly enforce constraints via the architecture graph. For example, our custom implementation inherently limits populations from crossing 0, preventing physically impossible calculations.</p>
        </details>
        <details>
            <summary>What is a Universal Differential Equation?</summary>
            <p>Authored by Chris Rackauckas et al., Universal Differential Equations constitute a "gray-box" workflow. Known scientific differential interactions are mapped identically inside the computational tree, but any unknown interaction residual terms are modeled by a neural network. This allows researchers to interpret the physics perfectly while achieving Deep Learning extrapolation capacities.</p>
        </details>
        
        <!-- SECTION 6: REPRODUCE -->
        <h2>Reproduce Locally</h2>
        <p>Estimated complete training cycle: ~5-10 minutes on CPU sequentially.</p>
        <pre><code>git clone https://github.com/angadjg07/neural-ode-epidemic.git
cd neural-ode-epidemic
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/train.py --model hybrid --epochs 500 --run_name final_hybrid</code></pre>
    </div>
</body>
</html>"""

    output_path = os.path.join(RESULTS_DIR, 'index.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
        
    print(f"Built results/index.html — open in browser to preview")

if __name__ == '__main__':
    build_html()
